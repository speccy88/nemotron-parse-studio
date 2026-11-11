# Private NVIDIA Python worker

`studio_worker.py` runs NVIDIA's unchanged `DocumentParser`, `PostprocessOptions`,
`postprocess_document`, `postprocess_text`, and table converters from revision
`a9a1febb1b9bf497fc67bd1d23fa33e170b249db`. Their source and licenses are in
`vendor/`; `vendor/PINNED_SOURCE.json` records source SHA-256 hashes. No other OCR
model is substituted. No hosted API is required.

## Start

From the project directory:

```sh
Backend/setup.sh            # lightweight postprocessing + vLLM client
Backend/run.sh              # persistent --stdio worker, JSON over private pipes
```

For local Transformers execution, use Python 3.12 and:

```sh
PYTHON=python3.12 Backend/setup.sh --model
Backend/.venv/bin/python Backend/download_model.py --output Models
Backend/run.sh
```

The explicit model download is about 2.6 GB. Application launch and parsing do
not download weights. Select the absolute `Models` directory in Studio settings.
The tested workspace also has `.venv-model`; `run.sh` prefers that environment
when present. `NEMOTRON_PYTHON=/absolute/python Backend/run.sh` overrides selection.

The loader supports CPU, Apple MPS, and NVIDIA CUDA. Studio defaults to MPS on
Apple silicon and to `auto` on Intel Macs. The worker uses `auto` when the request
omits a device, selecting CUDA first, then MPS, then CPU. CUDA uses BF16, and CPU
and MPS use float32. Each device runs the same pinned model through Transformers.
Arithmetic and sampling can affect model output.

The connection test checks local dependencies and the model file. Actual model
initialization happens at first Parse and errors are returned explicitly. Before
loading a local model, the bridge verifies checkpoint SHA-256 and config, tokenizer
and executable model source Git blob hashes against the pinned HF revision in
`MODEL_IDENTITY.json`. Verified files are cached by path, size and modification
time. Device aliases (`auto` and resolved `mps`/`cpu`/`cuda`) share one cached model.

## A Linux vLLM server on your own LAN

Install the pinned NVIDIA repository source, including its vLLM integration;
a generic unrelated vLLM image does not establish model compatibility. In a fresh
Linux Python environment with suitable NVIDIA hardware:

```sh
hf download nvidia/Nemotron-3.5-Parse \
  --revision a9a1febb1b9bf497fc67bd1d23fa33e170b249db --local-dir nemotron-parse
cd nemotron-parse
python -m pip install '.[vllm,client]'
vllm serve . --served-model-name nemotron-parse --trust-remote-code --host 127.0.0.1
```

See `vendor/README.md` and NVIDIA's pinned `VLLM_COMPATIBILITY.md` online for the
complete serving/plugin contract. For use across your private LAN, bind vLLM to
the machine's private interface, then set Studio's vLLM URL to
`http://192.168.x.y:8000/v1`. The bridge only permits loopback/private resolved
addresses, disables environment proxy settings, and refuses redirects. It sends
an RGB page only when Parse is requested. Studio never launches vLLM or any
listening service. Select an existing private server explicitly to use this
backend. No Docker is used by these scripts.

The vLLM models endpoint cannot attest the actual checkpoint hash. Studio
preserves the configured revision and warns that server-side revision is not
cryptographically verified. Local model checkpoints in this workspace have been
verified separately in the conversion evidence.

## Private stdio contract

The executable requires `--stdio`. It has no HTTP handler, `--port`, `--host`, or
listening socket. `Backend/run.sh` supplies the flag. The Swift app starts one
child and keeps anonymous stdin/stdout pipes open across requests, retaining the
model cache. There is no localhost HTTP transport.

Each line is one UTF-8 JSON envelope: `{"id":"unique-id","method":"health","params":{}}`.
Responses echo the ID with either `result` or `error`. Responses may finish out
of order; protocol IDs match replies, while parse `request_id` identifies the job
to cancel. Errors retain partial raw/stage evidence when available. Malformed
lines return an error without preventing the next valid request.

Requests are limited to 64 MB; Swift caps buffered responses at 128 MB. The worker
reserves a dedicated stdout descriptor for JSON and redirects library diagnostics
to stderr. Health and cancellation bypass the inference queue, including cancel
requests arriving before job dispatch. EOF cancels pending jobs. Swift restarts a
crashed worker on the next request without silently replaying an interrupted parse.
`worker_client.py` is the private-pipe Python client used by validation scripts.

- `health`: pinned revision, process ID, `transport: stdio`, active jobs, defaults.
- `parse`: `image_base64` (PNG/JPEG), `width`, `height`, `settings`, `options`,
  optional `request_id`. Image dimensions must match the encoded page.
- `postprocess`: `raw` (the previous response's `mergedRaw`), `originalRaw`
  (its immutable `raw`), `width`, `height`, `options`, optional prior `stages`,
  `settings`, and `backend`. This never loads or calls a model.
- `cancel`: `request_id`. Transformers stops at its next decoder step;
  an active vLLM HTTP generation may finish, but no next crop is submitted.
- `test-connection`: `settings`. For vLLM, validates `/models` and selected
  model identity. For Transformers, validates installed dependencies/weight file.

Settings include `backend` (`transformers` or `vllm`), `model_directory`, `device`,
`model` (vLLM served name), `revision`, `server_url`, `mode` (`single`, `double`,
`triple`), `triple_regions` (`union`, `detector`), `max_new_tokens`, `temperature`,
`top_p`, `top_k`, `seed`, `repetition_penalty`, `max_crops`, `crop_margin` (ratio),
`crop_margin_pixels`, and `predict_text_in_pic`. Crop temperature/repetition and
detection max tokens may be set separately. The default is NVIDIA's full-structure
control-token prompt, including picture text. No normal chat instruction replaces
it.

Formatting defaults are exactly NVIDIA's: header inference, chart rendering,
text normalization and deduplication enabled; numeric/percentage splitting and
caption relocation disabled. Both disabled options are opt-in.

Metrics use `inference_seconds`, `postprocess_seconds`, `model_load_seconds`,
`total_seconds`, `request_count`, `pages_per_second`, and `peak_memory_bytes`.

Response fields: `raw` immutable full-page generation; `mergedRaw` reversible crop
merge serialization; `markdown`; `blocks`; `warnings`; `stages`; `metrics`;
`backend`; `revision`; `settings`; `options`; `imageSize`; `inference`.

Blocks have stable content/geometry hash IDs, source text, formatted text, unknown
semantic class names unchanged, normalized top-left boxes, pixel boxes, reading
order, provenance and warnings. Table objects include anchored native cells with
row/column spans and header roles, plus HTML, Markdown, source LaTeX, CSV, record
JSON and hierarchical JSON. The native cells retain line breaks and original
numeric strings. Flat formats follow NVIDIA's span-expansion and column-name
rules. HTML, structured cells, raw text and hierarchical JSON retain structure.
For blocks containing multiple tables, the native grid describes the first table. Corrected native table exports also use the first
grid; complete official conversions remain in extraction metadata/structured JSON,
and a warning describes this limit. Optional caption and outside-table context text
are included; chart binding directives remain only in original HTML/raw content.

The official guarded Table/Chart crop merge is retained. Raw full-page, detection,
and crop generations are saved. For reversibility, preprocessing deduplication is
deferred until final presentation; a later formatting change can recover duplicate
blocks. This is a documented orchestration difference from the default helper.
No merge thresholds or region geometry are changed. Disabling picture text is a
narrow control-token substitution. Standard Transformers `StoppingCriteria` are
injected through a model proxy, leaving NVIDIA's adapter and its shared lock intact.

Token exhaustion, filtered/empty responses and generation failures are errors;
failed responses and successful prior stage responses accompany the error JSON.
Malformed table conversion retains the original prediction and emits warnings.
No confidential document contents are logged by the bridge. Extracted HTML remains
untrusted data: the native application sanitizes rendering and export.

## Tests and real execution

`Conversion/validate_stdio.py` exercises real Transformers single/double/triple
pass, reformatting, token-limit partial output, cancellation/retry, persistent
process ID, and zero worker network sockets. Current evidence is in
`Conversion/evidence/stdio/all-report.json`. Historical `bridge/` reports predate
the migration; their HTTP fields do not describe the current implementation.

The app's operational log is
`~/Library/Application Support/NemotronParseStudio/worker.log`. It is separate
from copied document originals under
`~/Library/Application Support/Nemotron Parse Studio/Imports/`.

```sh
Backend/.venv/bin/python -m pytest -q -c Backend/pytest.ini Backend/tests Backend/vendor/tests
```

The untouched NVIDIA suite includes a small CPU Transformers stub and therefore
needs PyTorch. These tests validate the bridge contract and NVIDIA helper behavior;
model accuracy and Core AI compatibility require separate real execution evidence.
`fixtures/reference_cases.json` contains small explicit serialization examples and
`fixtures/reference_expected.json` is generated by the pinned original helpers.
`fixtures/stdio_model_response.json` is the real MPS worker response for the
synthetic mixed page with its model directory made relative.
`fixtures/native_mixed_reference.txt` preserves the exact PyTorch CPU serialization
used to verify the native mixed-page prediction.

Current helper/worker test result: 82 passed. Swift and native Core AI results
are recorded in the root README and `docs/CORE_AI.md`.
