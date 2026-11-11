#!/usr/bin/env python3
"""Persistent private-pipe worker for pinned NVIDIA inference and postprocessing.

This module never substitutes OCR or mock output for Nemotron predictions.
Raw generations remain immutable and all formatting runs through the vendored
NVIDIA revision. The worker creates no listening socket. stdout is newline-delimited JSON only.
"""
from __future__ import annotations

import argparse
import base64
import copy
from dataclasses import asdict, fields, replace
import hashlib
from io import BytesIO
import ipaddress
import importlib.util
import json
import os
import re
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlparse
import uuid
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
from bs4 import BeautifulSoup
from PIL import Image, UnidentifiedImageError
from nemotron_parse import PostprocessOptions, postprocess_document
from nemotron_parse import tables
from nemotron_parse.postprocessing import clean_raw_output, postprocess_text, extract_classes_bboxes
from nemotron_parse.inference import (
    DocumentParser, GenerationError, GenerationSettings, TransformersBackend, merge_crops,
)

REVISION = "a9a1febb1b9bf497fc67bd1d23fa33e170b249db"
MODEL = "nvidia/Nemotron-3.5-Parse"
MAX_BODY = 64 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 40_000_000


class Cancelled(RuntimeError):
    pass


class RequestFailure(RuntimeError):
    def __init__(self, message, *, stages=None, raw=None):
        super().__init__(message)
        self.stages = stages or []
        self.raw = raw


def parse_options(value):
    value = value or {}
    allowed = {f.name for f in fields(PostprocessOptions)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError("Unknown postprocessing options: " + ", ".join(sorted(unknown)))
    if any(type(v) is not bool for v in value.values()):
        raise ValueError("Postprocessing options must be Boolean")
    return PostprocessOptions(**value)


def local_server_url(value):
    """Prevent accidental document transmission outside loopback/private LAN."""
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("Use an http(s) loopback or private LAN server URL without embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Server URL cannot contain query parameters or fragments")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    for item in addresses:
        address = ipaddress.ip_address(item[4][0])
        if not (address.is_loopback or address.is_private) or address.is_unspecified or address.is_multicast:
            raise ValueError("Only loopback and private LAN inference servers are permitted")
    return value.rstrip("/")


def make_table(html, raw, options):
    soup = BeautifulSoup(html, "html.parser")
    found = soup.find_all("table")
    if not found:
        return None
    if len(found) > 1:
        # The complete HTML/exports retain all tables; the native grid is first table.
        warning = "This block contains multiple tables. Native grid and corrected table exports show the first table. Complete model content and official conversions are retained in Raw and structured JSON."
    else:
        warning = None
    table = found[0]
    caption_node = table.find("caption", recursive=False)
    caption = caption_node.get_text(" ", strip=True) if caption_node else ""
    context_soup = BeautifulSoup(html, "html.parser")
    for context_table in context_soup.find_all("table"):
        context_table.decompose()
    for inactive in context_soup.find_all(["script", "style"]):
        inactive.decompose()
    context_text = context_soup.get_text("\n", strip=True)
    # NVIDIA chart bind directives are structural metadata; original HTML/raw retain them.
    if context_soup.find("chart"):
        context_text = re.sub(r"(?im)^\s*bind\s*:[^\n]*\n?", "", context_text).strip()
    grid, _ = tables._expand_spans(table)  # official validation, including malformed spans
    cells, occupied = [], set()
    rows = table.find_all("tr")
    for r, row in enumerate(rows):
        c = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while (r, c) in occupied:
                c += 1
            rs, cs = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            # Preserve line breaks in native cells, and raw inline markup separately.
            original_inline_html = cell.decode_contents()
            for br in cell.find_all("br"):
                br.replace_with("\n")
            cells.append({"row": r, "column": c, "rowSpan": rs, "columnSpan": cs,
                          "text": cell.get_text(), "isHeader": cell.name == "th",
                          "html": original_inline_html})
            for rr in range(r, r + rs):
                for cc in range(c, c + cs):
                    occupied.add((rr, cc))
            c += cs
    result = {
        "cells": cells, "rowCount": len(grid), "columnCount": len(grid[0]) if grid else 0,
        "html": html, "markdown": tables.convert_html_tables_to_markdown(html), "latex": raw,
        "csv": tables.convert_html_table_to_csv(html),
        "recordsJSON": tables.convert_html_table_to_json(html),
        "hierarchicalJSON": tables.convert_html_table_to_json_hierarchical(html),
        "caption": caption, "contextText": context_text,
    }
    if warning:
        result["warnings"] = [warning]
    return result


def normalize_result(document, *, raw, merged_raw, options, backend, settings, stages=None, metrics=None, size=None):
    warnings = [str(w.get("message", w)) if isinstance(w, dict) else str(w)
                for w in document.get("warnings", [])]
    blocks, seen = [], {}
    for index, item in enumerate(document.get("blocks", [])):
        source_text = item["text"]
        seed = json.dumps([item["class"], item["bbox"], source_text], ensure_ascii=False)
        digest = hashlib.sha256(seed.encode()).hexdigest()[:24]
        occurrence = seen.get(digest, 0)
        seen[digest] = occurrence + 1
        warning = []
        table = None
        if item["class"] in {"Table", "Chart"}:
            try:
                table = make_table(item["content"], source_text, options)
                if table is not None:
                    warning.extend(table.get("warnings", []))
                if table is None and (item["class"] == "Table" or "tabular" in source_text):
                    warning.append("Table prediction could not be converted. Original prediction retained.")
            except (ValueError, IndexError, TypeError, OverflowError) as error:
                warning.append("Malformed table retained: " + str(error))
        box = item["bbox"]
        box = [min(1.0, max(0.0, float(v))) for v in (min(box[0], box[2]), min(box[1], box[3]), max(box[0], box[2]), max(box[1], box[3]))]
        block = {"id": f"{digest}-{occurrence}", "rawText": source_text,
                 "text": item["content"], "semanticClass": item["class"], "bbox": box,
                 "bboxPixels": item.get("bbox_pixels", []), "readingOrder": index,
                 "provenance": "merged" if settings.get("mode", "single") != "single" else "fullpage",
                 "warnings": warning}
        if table is not None:
            block["table"] = table
        blocks.append(block)
        warnings.extend(f"Block {index + 1}: {w}" for w in warning)
    if not blocks and merged_raw.strip():
        warnings.append("No complete structured blocks recognized; raw output is preserved for inspection.")
    return {"raw": raw, "mergedRaw": merged_raw, "markdown": document["markdown"],
            "blocks": blocks, "warnings": list(dict.fromkeys(warnings)), "stages": stages or [],
            "metrics": metrics or {}, "backend": backend, "revision": REVISION,
            "settings": settings, "options": asdict(options),
            "imageSize": list(size or []), "coordinateFrame": "content_normalized",
            "inference": document.get("inference", {})}


def safe_postprocess(raw, *, image_size, options):
    try:
        return postprocess_document(raw, image_size=image_size, options=options)
    except (ValueError, IndexError, TypeError, OverflowError) as error:
        # A malformed table must not make the user's immutable prediction disappear.
        classes, boxes, texts = extract_classes_bboxes(raw)
        blocks = []
        for cls, box, text in zip(classes, boxes, texts):
            try:
                content = postprocess_text(text, cls, table_format="HTML", options=options)
            except (ValueError, IndexError, TypeError, OverflowError):
                content = text
            blocks.append({"class": cls, "bbox": list(box), "text": text, "content": content,
                           "bbox_pixels": [float(v) * image_size[i % 2] for i, v in enumerate(box)]})
        return {"raw_out": raw, "markdown": "\n\n".join(b["content"] for b in blocks) or raw,
                "blocks": blocks, "warnings": [{"message": "Official postprocessor rejected malformed markup; original block content retained: " + str(error)}]}


def postprocess_request(body):
    start = time.perf_counter()
    raw = body.get("raw")
    if not isinstance(raw, str):
        raise ValueError("raw must be a string")
    original_raw = body.get("originalRaw", raw)
    if not isinstance(original_raw, str):
        raise ValueError("originalRaw must be a string")
    options = parse_options(body.get("options"))
    size = (int(body["width"]), int(body["height"]))
    if min(size) <= 0:
        raise ValueError("Image width and height must be positive")
    document = safe_postprocess(raw, image_size=size, options=options)
    return normalize_result(document, raw=original_raw, merged_raw=raw, options=options,
                            backend=body.get("backend", "postprocessing-only"),
                            settings=body.get("settings") or {}, stages=body.get("stages") or [],
                            metrics={"postprocess_seconds": time.perf_counter() - start}, size=size)


class CancellableModelProxy:
    """Keep NVIDIA's adapter intact while adding Transformers' standard stop hook."""
    def __init__(self, model, cancelled):
        self._model, self._cancelled = model, cancelled

    def __getattr__(self, name):
        return getattr(self._model, name)

    def generate(self, *args, **kwargs):
        from transformers import StoppingCriteria, StoppingCriteriaList
        event = self._cancelled
        class StopWhenCancelled(StoppingCriteria):
            def __call__(self, input_ids, scores, **extra):
                return event.is_set()
        criteria = list(kwargs.pop("stopping_criteria", None) or [])
        criteria.append(StopWhenCancelled())
        return self._model.generate(*args, stopping_criteria=StoppingCriteriaList(criteria), **kwargs)


class PromptBackend:
    """A narrow control-token override, without changing NVIDIA orchestration."""
    def __init__(self, backend, *, predict_text_in_pic=True):
        self.backend = backend
        self.predict_text_in_pic = predict_text_in_pic

    def __call__(self, image, *, prompt, stage, settings):
        if not self.predict_text_in_pic and stage != "detection":
            prompt = prompt.replace("<predict_text_in_pic>", "<predict_no_text_in_pic>")
        return self.backend(image, prompt=prompt, stage=stage, settings=settings)


class HTTPBackend:
    """Pinned DocumentParser HTTP request contract with explicit transport settings."""
    def __init__(self, client, base_url, model):
        self.client, self.base_url, self.model = client, base_url, model

    def __call__(self, image, *, prompt, stage, settings):
        stream = BytesIO()
        image.save(stream, format="PNG")
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        response = self.client.post(self.base_url + "/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}},
            ]}], **asdict(settings), "skip_special_tokens": False,
        }, headers={"Authorization": "Bearer EMPTY"})
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        return {"raw_out": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"), "usage": body.get("usage", {})}


class RecordingParser(DocumentParser):
    def __init__(self, *args, cancelled, **kwargs):
        super().__init__(*args, **kwargs)
        self.cancelled = cancelled
        self.records = []
        self.inference_seconds = 0.0

    def _generate(self, image, stage):
        if self.cancelled.is_set():
            raise Cancelled("Parsing cancelled")
        start = time.perf_counter()
        try:
            response = super()._generate(image, stage)
        except GenerationError as error:
            response = error.response
            self.records.append({"name": stage, "raw": response.get("raw_out", ""),
                                 "finishReason": response.get("finish_reason") or "error",
                                 "usage": response.get("usage", {}), "seconds": time.perf_counter() - start})
            raise
        elapsed = time.perf_counter() - start
        self.inference_seconds += elapsed
        self.records.append({"name": stage, "raw": response["raw_out"],
                             "finishReason": response.get("finish_reason") or "stop",
                             "usage": response.get("usage", {}), "seconds": elapsed})
        if self.cancelled.is_set():
            raise Cancelled("Parsing cancelled after active generation completed")
        return response


class Engine:
    def __init__(self):
        self.model_lock = threading.Lock()
        self.models = {}
        self.jobs_lock = threading.Lock()
        self.jobs = {}
        self.cancelled_before_start = {}
        self.verified_files = {}

    def verify_model_directory(self, directory):
        manifest = json.loads((HERE / "MODEL_IDENTITY.json").read_text())
        for filename, expected in manifest["files"].items():
            path = Path(directory) / filename
            if not path.is_file():
                raise ValueError("Pinned model installation is missing " + filename)
            stat = path.stat()
            fingerprint = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
            if fingerprint in self.verified_files:
                continue
            if stat.st_size != expected["size"]:
                raise ValueError("Model file differs from pinned NVIDIA revision: " + filename)
            digest = hashlib.sha256() if expected["algorithm"] == "sha256" else hashlib.sha1()
            if expected["algorithm"] == "git-blob-sha1":
                digest.update(f"blob {stat.st_size}\0".encode())
            with path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected["digest"]:
                raise ValueError("Model file hash differs from pinned NVIDIA revision: " + filename)
            self.verified_files[fingerprint] = True

    def cancel(self, request_id):
        with self.jobs_lock:
            event = self.jobs.get(request_id)
            if event:
                event.set()
            else:
                # Handle cancellation racing pool scheduling, with a bounded
                # tombstone map. A retry receives a fresh job ID in the app.
                self.cancelled_before_start[request_id] = time.monotonic()
                if len(self.cancelled_before_start) > 128:
                    self.cancelled_before_start.pop(next(iter(self.cancelled_before_start)))
        return {"cancelled": True,
                "message": "Local Transformers stops at the next decoder step. vLLM stops subsequent stages; its active request may finish first."}

    def load_transformers(self, settings):
        model_path = settings.get("model_directory") or MODEL
        revision = settings.get("revision") or REVISION
        if revision != REVISION:
            raise ValueError("This build supports only the pinned NVIDIA revision " + REVISION)
        device = settings.get("device", "auto")
        if Path(model_path).is_dir():
            model_path = str(Path(model_path).resolve())
        start = time.perf_counter()
        with self.model_lock:
            try:
                import torch
                from transformers import AutoModel, AutoProcessor, GenerationConfig
            except ImportError as error:
                raise RuntimeError("Transformers backend dependencies are missing; run Backend/setup.sh --model") from error
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
            if device not in {"cpu", "mps", "cuda"}:
                raise ValueError("Device must be auto, cpu, mps or cuda")
            key = (model_path, revision, device)
            if key in self.models:
                return self.models[key], 0.0
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
            # Explicit installation is required: never surprise-download multi-GB weights.
            local = Path(model_path).is_dir()
            if local:
                self.verify_model_directory(model_path)
            kwargs = {"trust_remote_code": True, "local_files_only": True}
            if not local:
                kwargs["revision"] = revision
            # Release the previous cached model before loading another configuration.
            self.models.clear()
            try:
                model = AutoModel.from_pretrained(model_path, dtype=dtype, **kwargs).to(device).eval()
                processor = AutoProcessor.from_pretrained(model_path, **kwargs)
                generation = GenerationConfig.from_pretrained(model_path, local_files_only=True,
                                                               **({} if local else {"revision": revision}))
            except Exception as error:
                raise RuntimeError("Nemotron model initialization failed. Install pinned weights with Backend/download_model.py and select that directory. " + str(error)) from error
            adapter = TransformersBackend(model, processor, generation_config=generation)
            # Keep one installed model cached; changing device/path cannot leak old models.
            self.models.clear()
            self.models[key] = adapter
            return adapter, time.perf_counter() - start

    def parse(self, body):
        request_id = str(body.get("request_id") or uuid.uuid4())
        cancelled = threading.Event()
        with self.jobs_lock:
            cancelled_at = self.cancelled_before_start.pop(request_id, None)
            if cancelled_at is not None and time.monotonic() - cancelled_at < 300:
                raise Cancelled("Request cancelled before inference started")
            if request_id in self.jobs:
                raise ValueError("Request ID is already active")
            self.jobs[request_id] = cancelled
        try:
            return self._parse(body, cancelled)
        finally:
            with self.jobs_lock:
                self.jobs.pop(request_id, None)

    def _parse(self, body, cancelled):
        start = time.perf_counter()
        settings = dict(body.get("settings") or {})
        options = parse_options(body.get("options"))
        backend = settings.get("backend", "vllm")
        if backend not in {"vllm", "transformers"}:
            raise ValueError("Python inference backend must be vllm or transformers; native Core AI executes in Swift")
        revision = settings.get("revision") or REVISION
        if revision != REVISION:
            raise ValueError("The selected revision does not match the vendored reference helpers")
        image_data = base64.b64decode(body["image_base64"], validate=True)
        with Image.open(BytesIO(image_data)) as source:
            image = source.convert("RGB")
        if tuple((int(body["width"]), int(body["height"]))) != image.size:
            raise ValueError("Declared dimensions do not match the rendered RGB page")
        mode = settings.get("mode", "single")
        common = {"top_p": float(settings.get("top_p", 1.0)),
                  "top_k": int(settings.get("top_k", 50)), "seed": int(settings.get("seed", 0))}
        fullpage = GenerationSettings(**common, max_tokens=int(settings.get("max_new_tokens", 8744)),
                                      temperature=float(settings.get("temperature", 0.2)),
                                      repetition_penalty=float(settings.get("repetition_penalty", 1.05)))
        detection = replace(fullpage, max_tokens=int(settings.get("detection_max_tokens", 4096)))
        crop = replace(fullpage, temperature=float(settings.get("crop_temperature", 0.1)),
                       repetition_penalty=float(settings.get("crop_repetition_penalty", 1.0)))
        model_load = 0.0
        args = {"options": replace(options, deduplicate=False), "fullpage": fullpage,
                "detection": detection, "crop": crop, "max_regions": int(settings.get("max_crops", 80)),
                "crop_margin_ratio": float(settings.get("crop_margin", 0.04)),
                "crop_margin_px": int(settings.get("crop_margin_pixels", 12)), "cancelled": cancelled}
        client = None
        if backend == "transformers":
            loaded, model_load = self.load_transformers(settings)
            if isinstance(loaded, TransformersBackend):
                # Share NVIDIA's serialization lock; each request owns its stop event.
                loaded = copy.copy(loaded)
                loaded.model = CancellableModelProxy(loaded.model, cancelled)
            args["backend"] = loaded
            args["model"] = MODEL
        else:
            import httpx
            args["base_url"] = local_server_url(settings.get("server_url", "http://127.0.0.1:8000/v1"))
            args["model"] = settings.get("model") or "nemotron-parse"
            # No proxies, redirects or inherited cloud credentials; only explicit LAN target.
            client = httpx.Client(timeout=1800.0, trust_env=False, follow_redirects=False)
            args["client"] = client
            args["api_key"] = "EMPTY"
            args["backend"] = HTTPBackend(client, args["base_url"], args["model"])
        args["backend"] = PromptBackend(args["backend"], predict_text_in_pic=bool(settings.get("predict_text_in_pic", True)))
        parser = RecordingParser(**args)
        try:
            with parser:
                parsed = parser.parse(image, mode=mode, triple_regions=settings.get("triple_regions", "union"))
            raw = parsed["stages"]["fullpage"]["raw_out"]
            merged_raw = raw if mode == "single" else merge_crops(clean_raw_output(raw, deduplicate=False), parsed["stages"]["crops"])
            # Render once from reversible merge serialization with the user's options.
            pp_start = time.perf_counter()
            document = safe_postprocess(merged_raw, image_size=image.size, options=options)
            document["inference"] = parsed["inference"]
            document["inference"]["postprocessing"] = asdict(options)
            document["inference"]["backend"] = backend
            document["inference"]["base_url"] = args.get("base_url") if backend == "vllm" else None
            document["inference"]["predict_text_in_pic"] = bool(settings.get("predict_text_in_pic", True))
            pp_seconds = time.perf_counter() - pp_start
            crops = iter(parsed["stages"]["crops"])
            for stage in parser.records:
                if stage["name"] == "crop":
                    crop_row = next(crops)
                    stage["cropBBoxPixels"] = list(crop_row["crop_bbox_pixels"])
                    stage["cropID"] = crop_row["crop_id"]
            elapsed = time.perf_counter() - start
            metrics = {"model_load_seconds": model_load, "inference_seconds": parser.inference_seconds,
                       "postprocess_seconds": pp_seconds, "total_seconds": elapsed,
                       "request_count": len(parser.records), "pages_per_second": 1 / elapsed}
            try:
                import resource
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                metrics["peak_memory_bytes"] = rss if sys.platform == "darwin" else rss * 1024
            except ImportError:
                pass
            result = normalize_result(document, raw=raw, merged_raw=merged_raw, options=options,
                                      backend=backend, settings=settings, stages=parser.records, metrics=metrics, size=image.size)
            if backend == "vllm":
                result["warnings"].append("Server model revision is declared by configuration, not cryptographically verified by the vLLM API.")
            return result
        except Exception as error:
            raise RequestFailure(str(error), stages=parser.records,
                                 raw=parser.records[0]["raw"] if parser.records else None) from error
        finally:
            if client:
                client.close()

    def health(self):
        return {"status": "ok", "service": "Nemotron Parse Studio", "revision": REVISION,
                "model": MODEL, "backends": ["transformers", "vllm"], "transport": "stdio", "pid": os.getpid(),
                "loadedModels": len(self.models), "activeRequests": len(self.jobs),
                "postprocessDefaults": asdict(PostprocessOptions())}


ENGINE = Engine()


def test_connection(settings):
    if settings.get("backend") == "transformers":
        missing = [name for name in ("torch", "transformers", "timm") if importlib.util.find_spec(name) is None]
        if missing: raise ValueError("Missing model dependencies: " + ", ".join(missing))
        directory = settings.get("model_directory", "")
        if not directory or not Path(directory, "model.safetensors").is_file():
            raise ValueError("Select the installed pinned model directory containing model.safetensors")
        return {"status": "ok", "backend": "transformers", "message": "Dependencies and weights found. Model initialization occurs on the first parse."}
    if settings.get("backend") in {"coreai", "mock"}:
        return {"status": "ok", "backend": settings["backend"], "message": "Private postprocessing worker is ready. Native model validation is performed in Swift."}
    import httpx
    url = local_server_url(settings.get("server_url", "http://127.0.0.1:8000/v1"))
    with httpx.Client(timeout=15, trust_env=False, follow_redirects=False) as client:
        response = client.get(url + "/models"); response.raise_for_status()
        models = [m["id"] for m in response.json().get("data", [])]
    selected = settings.get("model") or "nemotron-parse"
    if selected not in models: raise ValueError("Configured model is not served by this endpoint. Available: " + ", ".join(models))
    return {"status": "ok", "backend": "vllm", "models": models}


def dispatch(method, body):
    if method == "health": return ENGINE.health()
    if method == "parse": return ENGINE.parse(body)
    if method == "postprocess": return postprocess_request(body)
    if method == "cancel": return ENGINE.cancel(str(body["request_id"]))
    if method == "test-connection": return test_connection(body.get("settings", body))
    raise ValueError("Unknown worker method: " + method)


def run_stdio():
    # Reserve stdout at the OS descriptor level. Even native library writes to
    # fd 1 must go to stderr, never corrupting a protocol response or deadlocking
    # the private pipe. Only the duplicate descriptor writes JSON responses.
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr
    write_lock = threading.Lock()
    inflight_lock = threading.Lock()
    inflight = set()
    slots = threading.BoundedSemaphore(8)
    pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="nemotron-worker")
    def reply(value):
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
            with write_lock: output.write(encoded + "\n"); output.flush()
        except (BrokenPipeError, OSError): pass
    def perform(request):
        identity = request["id"]
        try:
            result = dispatch(request["method"], request.get("params") or {})
            reply({"id": identity, "result": result})
        except RequestFailure as error:
            reply({"id": identity, "error": {"message": str(error), "raw": error.raw, "stages": error.stages}})
        except Exception as error:
            reply({"id": identity, "error": {"message": f"{type(error).__name__}: {error}"}})
        finally:
            with inflight_lock: inflight.discard(identity)
            slots.release()
    try:
        while True:
            line = sys.stdin.buffer.readline(MAX_BODY + 2)
            if not line: break
            identity = None
            try:
                if len(line) > MAX_BODY:
                    while line and not line.endswith(b"\n"): line = sys.stdin.buffer.readline(MAX_BODY + 2)
                    raise ValueError("Worker request exceeds the 64 MB limit")
                request = json.loads(line)
                if not isinstance(request, dict): raise ValueError("Request must be a JSON object")
                identity = request.get("id")
                if not isinstance(identity, str) or not 0 < len(identity) <= 128: raise ValueError("Request requires a string id of 1–128 characters")
                if not isinstance(request.get("method"), str) or not isinstance(request.get("params", {}), dict): raise ValueError("Request requires method and object params")
                # Control calls bypass the inference pool so cancellation always
                # remains responsive while model threads are occupied.
                if request["method"] in {"health", "cancel"}:
                    reply({"id": identity, "result": dispatch(request["method"], request.get("params", {}))}); continue
                with inflight_lock:
                    if identity in inflight: raise ValueError("Request ID is already active")
                    if not slots.acquire(blocking=False): raise ValueError("Worker queue is full; retry after current pages finish")
                    inflight.add(identity)
                pool.submit(perform, request)
            except Exception as error:
                reply({"id": identity, "error": {"message": f"{type(error).__name__}: {error}"}})
    finally:
        with ENGINE.jobs_lock:
            for event in ENGINE.jobs.values(): event.set()
        pool.shutdown(wait=True, cancel_futures=True)
        output.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stdio", action="store_true", required=True, help="Use newline-delimited JSON over private stdin/stdout pipes; never open a listening port")
    parser.parse_args()
    run_stdio()

if __name__ == "__main__": main()
