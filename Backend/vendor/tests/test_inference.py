import base64
import json
from io import BytesIO

import httpx
from PIL import Image
import pytest

from nemotron_parse.inference import (
    DocumentParser,
    GenerationError,
    GenerationSettings,
    merge_crops,
)
from nemotron_parse import PostprocessOptions
from test_postprocessing import block, tabular

OLD = tabular(r"\thead{Name} & \thead{Value}", "A & old")
NEW = tabular(r"\thead{Name} & \thead{Value}", "A & new")


def test_double_refines_table_and_keeps_text_order_and_geometry():
    calls = []

    def backend(image, *, stage, **kwargs):
        calls.append((stage, image.size))
        return {
            "raw_out": (
                (
                    block("Before", box=(0, 0, 1, 0.05))
                    + block(OLD, "Table")
                    + block("After", box=(0, 0.95, 1, 1))
                )
                if stage == "fullpage"
                else block(NEW, "Table", (0, 0, 1, 1))
            ),
            "finish_reason": "stop",
        }

    with DocumentParser(
        backend=backend, crop_margin_px=0, crop_margin_ratio=0
    ) as parser:
        doc = parser.parse_double(Image.new("RGB", (100, 200)))
    assert [c[0] for c in calls] == ["fullpage", "crop"]
    assert calls[1][1] == (80, 160)
    assert doc["inference"]["requests"] == 2
    assert [b["class"] for b in doc["blocks"]] == ["Text", "Table", "Text"]
    assert doc["blocks"][1]["bbox"] == [0.1, 0.1, 0.9, 0.9]
    assert "new" in doc["markdown"] and "old" not in doc["markdown"]
    assert "old" in doc["stages"]["fullpage"]["raw_out"]


def test_triple_detects_missing_table_and_maps_crop_coordinates():
    def backend(image, *, stage, **kwargs):
        if stage == "fullpage":
            return block("Before", box=(0, 0, 1, 0.05))
        if stage == "detection":
            return block("", "Table", (0.2, 0.3, 0.8, 0.7))
        return block(NEW, "Table", (0, 0, 1, 1))

    parser = DocumentParser(backend=backend, crop_margin_px=0, crop_margin_ratio=0)
    doc = parser.parse_triple(Image.new("RGB", (100, 100)))
    assert doc["inference"]["requests"] == 3
    assert doc["blocks"][1]["bbox"] == [0.2, 0.3, 0.8, 0.7]
    assert "new" in doc["markdown"]


def test_union_adds_parser_regions_but_detector_only_does_not():
    def backend(image, *, stage, **kwargs):
        if stage == "fullpage":
            return block(OLD, "Table", (0.1, 0.1, 0.4, 0.4))
        if stage == "detection":
            return block("", "Chart", (0.6, 0.6, 0.9, 0.9))
        return block("chart data", "Chart", (0, 0, 1, 1))

    parser = DocumentParser(backend=backend, crop_margin_px=0, crop_margin_ratio=0)
    image = Image.new("RGB", (100, 100))
    union = parser.parse_triple(image)
    detector = parser.parse_triple(image, triple_regions="detector")
    assert len(union["stages"]["crops"]) == 2
    assert len(detector["stages"]["crops"]) == 1


def test_region_limit_applies_to_union():
    def backend(image, *, stage, **kwargs):
        return (
            block("", "Table", (0.1, 0.1, 0.4, 0.4))
            if stage == "detection"
            else block(OLD, "Table", (0.6, 0.6, 0.9, 0.9))
        )

    parser = DocumentParser(backend=backend, max_regions=1)
    assert (
        len(parser.parse_triple(Image.new("RGB", (100, 100)))["inference"]["regions"])
        == 1
    )


def test_no_regions_does_not_make_crop_request():
    calls = []

    def backend(image, *, stage, **kwargs):
        calls.append(stage)
        return block("hello") if stage == "fullpage" else "</s>"

    parser = DocumentParser(backend=backend)
    doc = parser.parse_triple(Image.new("RGB", (20, 20)))
    assert calls == ["fullpage", "detection"]
    assert doc["markdown"] == "hello"


@pytest.mark.parametrize("stage", ["fullpage", "detection", "crop"])
@pytest.mark.parametrize(
    "response",
    [
        {"raw_out": "partial", "finish_reason": "length"},
        {"raw_out": ""},
        {"raw_out": "bad", "error": "failed"},
    ],
)
def test_failed_generations_are_not_reported_as_success(stage, response):
    def backend(image, *, stage: str, **kwargs):
        if stage == target:
            return response
        return block(OLD, "Table")

    target = stage
    parser = DocumentParser(backend=backend)
    with pytest.raises(GenerationError) as error:
        parser.parse_triple(Image.new("RGB", (100, 100)))
    assert error.value.stage == stage
    assert error.value.response["raw_out"] == response["raw_out"]


def test_guard_rejects_incompatible_column_count():
    wrong = tabular(
        r"\thead{Name} & \thead{Value} & \thead{Extra}", "A & new & extra", cols="ccc"
    )
    row = {
        "raw_out": block(wrong, "Table", (0, 0, 1, 1)),
        "detector_label": "table",
        "detector_score": 1.0,
        "detector_bbox_norm": (0.1, 0.1, 0.9, 0.9),
        "crop_bbox_pixels": (10, 10, 90, 90),
        "page_size": (100, 100),
    }
    merged = merge_crops(block(OLD, "Table"), [row])
    assert "old" in merged and "extra" not in merged


def test_batch_preserves_order(tmp_path):
    paths = []
    for width in [10, 20, 30]:
        path = tmp_path / f"{width}.png"
        Image.new("RGB", (width, 10)).save(path)
        paths.append(path)
    parser = DocumentParser(backend=lambda image, **kwargs: block(str(image.width)))
    docs = parser.parse_many(paths, concurrency=3)
    assert [d["markdown"] for d in docs] == ["10", "20", "30"]


def test_http_endpoint_protocol_auth_and_stage_settings():
    calls = []

    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["Authorization"] == "Bearer test-key"
        assert str(request.url) == "http://example/v1/chat/completions"
        assert body["skip_special_tokens"] is False
        uri = body["messages"][0]["content"][1]["image_url"]["url"]
        image = Image.open(BytesIO(base64.b64decode(uri.split(",")[1])))
        assert image.mode == "RGB"
        prompt = body["messages"][0]["content"][0]["text"]
        raw = block("", "Table") if "output_no_text" in prompt else block(OLD, "Table")
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": raw}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 10},
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handle))
    with DocumentParser(
        "http://example/v1/", api_key="test-key", client=client
    ) as parser:
        doc = parser.parse_triple(Image.new("RGB", (100, 100)))
    assert [x["max_tokens"] for x in calls] == [8744, 4096, 8744]
    assert [x["temperature"] for x in calls] == [0.2, 0.2, 0.1]
    assert doc["stages"]["fullpage"]["usage"]["completion_tokens"] == 10
    assert not client.is_closed
    client.close()


@pytest.mark.parametrize("mode", ["bad", "all"])
def test_invalid_mode_does_not_issue_request(mode):
    parser = DocumentParser(backend=lambda *a, **k: pytest.fail("unexpected inference"))
    with pytest.raises(ValueError):
        parser.parse(Image.new("RGB", (10, 10)), mode=mode)


def test_generation_options_validate_limits():
    with pytest.raises(ValueError):
        GenerationSettings(max_tokens=0)
    with pytest.raises(ValueError):
        GenerationSettings(temperature=float("nan"))
    with pytest.raises(ValueError):
        DocumentParser(backend=lambda **k: "", max_regions=0)


def test_transformers_backend_cpu_settings_rng_and_truncation():
    import torch
    from types import SimpleNamespace
    from nemotron_parse.inference import TransformersBackend

    class Inputs(dict):
        def to(self, device):
            return self

    class Processor:
        def __call__(self, **kwargs):
            assert kwargs["add_special_tokens"] is False
            return Inputs(input_ids=torch.tensor([[4, 5]]))

        def batch_decode(self, sequences, **kwargs):
            assert kwargs["skip_special_tokens"] is False
            return [block("hello")]

    class Model:
        device = "cpu"
        generation_config = SimpleNamespace(max_new_tokens=999)

        def generate(self, **kwargs):
            config = kwargs["generation_config"]
            assert config.do_sample is False and config.max_new_tokens == 3
            torch.rand(1)
            return SimpleNamespace(sequences=torch.tensor([[4, 5, 7, 2]]))

    model = Model()
    backend = TransformersBackend(model, Processor())
    before = torch.random.get_rng_state().clone()
    response = backend(
        Image.new("RGB", (10, 10)),
        prompt="test",
        stage="fullpage",
        settings=GenerationSettings(temperature=0, max_tokens=3),
    )
    assert response["usage"]["completion_tokens"] == 2
    assert response["finish_reason"] == "stop"
    assert torch.equal(before, torch.random.get_rng_state())
    assert model.generation_config.max_new_tokens == 999

    class Capped(Model):
        def generate(self, **kwargs):
            return SimpleNamespace(sequences=torch.tensor([[4, 5, 7, 8, 2]]))

    backend = TransformersBackend(Capped(), Processor())
    parser = DocumentParser(
        backend=backend, fullpage=GenerationSettings(temperature=0, max_tokens=3)
    )
    with pytest.raises(GenerationError):
        parser.parse(Image.new("RGB", (10, 10)))


def test_chart_crop_refinement_is_guarded():
    original = "<chart>\ntitle: Sales\n| Item | Value |\n| --- | --- |\n| A | 10 |\n| B | 20 |\n</chart>"
    updated = original.replace("10", "11")
    row = {
        "raw_out": block(updated, "Chart", (0, 0, 1, 1)),
        "detector_label": "chart",
        "detector_score": 1.0,
        "detector_bbox_norm": (0.1, 0.1, 0.9, 0.9),
        "crop_bbox_pixels": (10, 10, 90, 90),
        "page_size": (100, 100),
    }
    merged = merge_crops(block(original, "Chart"), [row])
    assert merged.count("<class_Chart>") == 1 and "| A | 11 |" in merged
    row["raw_out"] = block(updated.replace("</chart>", ""), "Chart", (0, 0, 1, 1))
    merged = merge_crops(block(original, "Chart"), [row])
    assert "| A | 10 |" in merged and merged.count("<class_Chart>") == 1


def test_cli_writes_markdown_and_stage_json(tmp_path, monkeypatch, capsys):
    from nemotron_parse import inference

    image = tmp_path / "page.png"
    Image.new("RGB", (20, 20)).save(image)
    output = tmp_path / "results"
    original = inference.DocumentParser
    monkeypatch.setattr(
        inference,
        "DocumentParser",
        lambda *a, **k: original(
            backend=lambda image, **kw: block("CLI text"), options=k.get("options")
        ),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "nemotron-parse",
            "--image",
            str(image),
            "--mode",
            "double",
            "--output",
            str(output),
        ],
    )
    inference.main()
    assert (output / "0000_page.md").read_text() == "CLI text"
    saved = json.loads((output / "0000_page.json").read_text())
    assert saved["inference"]["mode"] == "double"
    assert saved["stages"]["fullpage"]["raw_out"] == block("CLI text")
    with pytest.raises(SystemExit):
        inference.main()
