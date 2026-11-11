import base64
from dataclasses import asdict
from io import BytesIO
import json
from pathlib import Path
import socket
import threading

import httpx
from PIL import Image
import pytest

import studio_worker as service
from nemotron_parse import PostprocessOptions, postprocess_document


def block(text, cls="Text", box=(0.1, 0.1, 0.9, 0.9)):
    return f"<x_{box[0]}><y_{box[1]}>{text}<x_{box[2]}><y_{box[3]}><class_{cls}>"


def body(raw, **options):
    return {"raw": raw, "width": 800, "height": 600, "options": options}


def image_body(settings=None):
    output = BytesIO()
    Image.new("RGB", (80, 100), "white").save(output, "PNG")
    return {"image_base64": base64.b64encode(output.getvalue()).decode(), "width": 80, "height": 100,
            "settings": {"backend": "transformers", **(settings or {})}}


def test_defaults_exact_pinned():
    assert asdict(service.parse_options({})) == {
        "infer_headers": True, "render_charts": True, "split_chart_value_percent": False,
        "relocate_chart_captions": False, "normalize_text": True, "deduplicate": True,
    }
    with pytest.raises(ValueError):
        service.parse_options({"infer_headers": "false"})


def test_raw_reversible_postprocessing_and_stable_ids():
    raw = block("soft\u00adhyphen\n• bullet", "Text") + block("未来 Ω café", "FutureUnknownRole", (0.2, 0.3, 0.7, 0.8))
    default = service.postprocess_request(body(raw))
    plain = service.postprocess_request(body(raw, normalize_text=False))
    assert default["raw"] == raw == plain["raw"]
    assert default["blocks"][0]["id"] == plain["blocks"][0]["id"]
    assert default["blocks"][1]["semanticClass"] == "FutureUnknownRole"
    assert default["blocks"][1]["bboxPixels"] == [160.0, 180.0, 560.0, 480.0]
    assert default["markdown"] == postprocess_document(raw, image_size=(800,600))["markdown"]


def test_dedup_is_reversible():
    raw = block("Repeat") * 2
    assert len(service.postprocess_request(body(raw))["blocks"]) == 1
    result = service.postprocess_request(body(raw, deduplicate=False))
    assert len(result["blocks"]) == 2
    assert result["blocks"][0]["id"] != result["blocks"][1]["id"]
    assert result["mergedRaw"] == raw


def test_empty_and_malformed_predictions_are_preserved():
    assert service.postprocess_request(body(""))["blocks"] == []
    raw = "<s><x_0.1>unfinished prediction"
    result = service.postprocess_request(body(raw))
    assert result["raw"] == raw and result["warnings"]
    table = block(r"\begin{tabular}{cc} A & B", "Table")
    result = service.postprocess_request(body(table))
    assert result["raw"] == table
    assert "original" in " ".join(result["warnings"]).lower()


def test_native_table_spans_empty_cells_inline_and_values():
    raw = block('<table><tr><th colspan="2">Same</th></tr><tr><td rowspan="2"><b>0032</b><br>16.6%</td><td></td></tr><tr><td>末尾</td></tr></table>', "Table")
    result = service.postprocess_request(body(raw, infer_headers=False))
    table = result["blocks"][0]["table"]
    assert table["rowCount"] == 3 and table["columnCount"] == 2
    assert table["cells"][0]["columnSpan"] == 2
    assert table["cells"][0]["isHeader"]
    assert table["cells"][1]["rowSpan"] == 2
    assert table["cells"][1]["text"] == "0032\n16.6%"
    assert table["cells"][2]["text"] == ""
    assert json.loads(table["recordsJSON"])[0]["Same"] == "0032 16.6%"
    assert "Same_1" in json.loads(table["recordsJSON"])[0]
    assert json.loads(table["hierarchicalJSON"])["body_cells"][0][0]["rowspan"] == 2


def test_malformed_spans_never_discard_raw():
    raw = block('<table><tr><td rowspan="999999999">bad</td></tr></table>', "Table")
    result = service.postprocess_request(body(raw))
    assert result["raw"] == raw
    assert result["blocks"][0]["rawText"] in raw
    assert result["warnings"]


def test_postprocess_keeps_original_fullpage_and_stages():
    original, merged = block("original"), block("crop improvement")
    result = service.postprocess_request({**body(merged), "originalRaw": original,
                                          "stages": [{"name": "fullpage", "raw": original}]})
    assert result["raw"] == original
    assert result["mergedRaw"] == merged
    assert result["stages"][0]["raw"] == original


def test_private_network_and_public_rejection(monkeypatch):
    assert service.local_server_url("http://127.0.0.1:8000/v1") == "http://127.0.0.1:8000/v1"
    assert service.local_server_url("http://192.168.1.2:8000/v1")
    for url in ["https://8.8.8.8/v1", "http://user:password@localhost/v1", "file:///tmp/input", "http://0.0.0.0/v1"]:
        with pytest.raises(ValueError):
            service.local_server_url(url)


def test_parse_exact_official_prompt_and_metadata(monkeypatch):
    calls = []
    def backend(image, *, prompt, stage, settings):
        calls.append((prompt, stage, settings))
        return {"raw_out": "<s>" + block("A title", "Title") + "</s>", "finish_reason": "stop", "usage": {"completion_tokens": 12}}
    engine = service.Engine()
    monkeypatch.setattr(engine, "load_transformers", lambda settings: (backend, 0.0))
    result = engine.parse(image_body())
    assert calls[0][0] == "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
    assert result["backend"] == "transformers"
    assert result["raw"].startswith("<s>")
    assert result["metrics"]["request_count"] == 1
    assert result["stages"][0]["raw"] == result["raw"]
    assert result["revision"] == service.REVISION
    assert result["inference"]["postprocessing"] == asdict(PostprocessOptions())


def test_prompt_control_is_control_token_not_chat(monkeypatch):
    prompts = []
    def backend(image, *, prompt, stage, settings):
        prompts.append(prompt)
        return block("text")
    engine = service.Engine()
    monkeypatch.setattr(engine, "load_transformers", lambda settings: (backend, 0.0))
    engine.parse(image_body({"predict_text_in_pic": False}))
    assert prompts == ["</s><s><predict_bbox><predict_classes><output_markdown><predict_no_text_in_pic>"]


def test_double_triple_stages_use_official_crops(monkeypatch):
    calls = []
    table = r"\begin{tabular}{cc} A & 1 \\ B & 2 \\ \end{tabular}"
    def backend(image, *, prompt, stage, settings):
        calls.append((stage, image.size, settings))
        return {"raw_out": block(table if stage != "detection" else "", "Table"), "finish_reason": "stop"}
    engine = service.Engine()
    monkeypatch.setattr(engine, "load_transformers", lambda settings: (backend, 0.0))
    for mode, expected in [("double", ["fullpage", "crop"]), ("triple", ["fullpage", "detection", "crop"])]:
        calls.clear()
        result = engine.parse(image_body({"mode": mode}))
        assert [c[0] for c in calls] == expected
        assert result["stages"][-1]["cropBBoxPixels"]
        assert calls[-1][2].temperature == 0.1
        assert calls[-1][2].repetition_penalty == 1.0
        assert result["metrics"]["request_count"] == len(expected)


def test_token_limit_retains_failed_response(monkeypatch):
    engine = service.Engine()
    monkeypatch.setattr(engine, "load_transformers", lambda settings: (lambda *a, **kw: {"raw_out": "partial", "finish_reason": "length"}, 0.0))
    with pytest.raises(service.RequestFailure) as failure:
        engine.parse(image_body())
    assert failure.value.stages[0]["raw"] == "partial"
    assert "length" in str(failure.value)


def test_cancel_and_retry(monkeypatch):
    engine = service.Engine()
    started, released = threading.Event(), threading.Event()
    def backend(*args, **kwargs):
        started.set()
        released.wait(2)
        return block("okay")
    monkeypatch.setattr(engine, "load_transformers", lambda settings: (backend, 0.0))
    failures = []
    def parse():
        try:
            engine.parse({**image_body(), "request_id": "cancel-test"})
        except service.RequestFailure as error:
            failures.append(error)
    thread = threading.Thread(target=parse)
    thread.start()
    assert started.wait(2)
    assert engine.cancel("cancel-test")["cancelled"]
    released.set()
    thread.join(2)
    assert failures and not engine.jobs
    result = engine.parse({**image_body(), "request_id": "cancel-test"})
    assert result["blocks"][0]["rawText"] == "okay"


def test_reject_mismatched_image_dimensions(monkeypatch):
    with pytest.raises(ValueError, match="dimensions"):
        service.Engine().parse({**image_body(), "width": 81})


def test_transformers_cancellation_uses_decoder_stopping_criteria():
    cancelled = threading.Event()
    class Model:
        device = "cpu"
        def generate(self, *args, stopping_criteria, **kwargs):
            assert not stopping_criteria[0](None, None)
            cancelled.set()
            assert stopping_criteria[0](None, None)
            return "stopped"
    assert service.CancellableModelProxy(Model(), cancelled).generate() == "stopped"


def test_inline_html_retains_breaks_and_formatting_separately():
    raw = block('<table><tr><td><b>bold</b><br><sup>2</sup></td></tr></table>', 'Table')
    result = service.postprocess_request(body(raw))
    cell = result['blocks'][0]['table']['cells'][0]
    assert cell['text'] == 'bold\n2'
    assert '<b>bold</b>' in cell['html'] and '<br' in cell['html'] and '<sup>2</sup>' in cell['html']


def test_pinned_reference_fixtures_match_original_helpers():
    fixtures = Path(__file__).resolve().parents[1] / 'fixtures'
    cases = json.loads((fixtures / 'reference_cases.json').read_text())
    expected = json.loads((fixtures / 'reference_expected.json').read_text())
    for name, raw in cases.items():
        assert postprocess_document(raw, image_size=(800, 600)) == expected[name]
        wrapped = service.postprocess_request(body(raw))
        assert wrapped['raw'] == raw
        assert wrapped['markdown'] == expected[name]['markdown']


def test_chart_caption_context_and_multi_table_warning():
    raw = block('<chart type="bar">bind: Value ; ; y\nNote: original values<table><caption>Quarterly share</caption><tr><td>A</td><td>60%</td></tr></table></chart>', 'Chart')
    result = service.postprocess_request(body(raw))
    table = result['blocks'][0]['table']
    assert table['caption'] == 'Quarterly share'
    assert table['contextText'] == 'Note: original values'
    assert 'bind: Value' in table['html']
    raw = block('<table><tr><td>1</td></tr></table><table><tr><td>2</td></tr></table>', 'Table')
    result = service.postprocess_request(body(raw))
    assert 'multiple tables' in ' '.join(result['blocks'][0]['warnings'])
    assert result['blocks'][0]['table']['html'].count('<table>') == 2


def test_pinned_model_identity_rejects_changed_file(tmp_path, monkeypatch):
    import hashlib
    payload = b'weights fixture'
    (tmp_path / 'model.safetensors').write_bytes(payload)
    (tmp_path / 'MODEL_IDENTITY.json').write_text(json.dumps({'files': {
        'model.safetensors': {'size': len(payload), 'algorithm': 'sha256', 'digest': hashlib.sha256(payload).hexdigest()}
    }}))
    monkeypatch.setattr(service, 'HERE', tmp_path)
    engine = service.Engine()
    engine.verify_model_directory(tmp_path)
    assert engine.verified_files
    (tmp_path / 'model.safetensors').write_bytes(b'Weights fixture')
    with pytest.raises(ValueError, match='hash differs'):
        engine.verify_model_directory(tmp_path)


def test_auto_and_explicit_device_share_model_cache(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.backends.mps, 'is_available', lambda: True)
    engine = service.Engine()
    model = object()
    engine.models[(str(tmp_path.resolve()), service.REVISION, 'mps')] = model
    automatic, duration = engine.load_transformers({'model_directory': str(tmp_path), 'device': 'auto'})
    explicit, duration2 = engine.load_transformers({'model_directory': str(tmp_path), 'device': 'mps'})
    assert automatic is model is explicit
    assert duration == duration2 == 0
    assert len(engine.models) == 1
