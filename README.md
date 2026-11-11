# Nemotron Parse Studio

A native SwiftUI workspace for NVIDIA Nemotron-3.5-Parse: import documents, parse pages, inspect aligned boxes and tables, adjust NVIDIA postprocessing, make reversible corrections, and export results. Model and helper revision: `a9a1febb1b9bf497fc67bd1d23fa33e170b249db`.

The Release app was built and tested on macOS 27.2.1 with Xcode 27.0. Local Transformers supports single, double, and triple pass. Native Swift/Core AI passed end-to-end comparison with PyTorch for the **512 × 256 and 640 × 800** profiles. Select Local NVIDIA Transformers for other geometries or multipass parsing.

## Open the app after building

Open `build/Build/Products/Release/Nemotron Parse Studio.app`, or open `NemotronParseStudio.xcodeproj` and run its scheme.

This development build uses its source workspace, Python environment, and model directories. Keep them together. It is locally signed, not a notarized standalone distribution. Moving the workspace requires rebuilding to update the embedded source location.

1. Import with **⌘O** or drag files into the window. Multiple files and multipage documents are supported.
2. Set the model directory to this workspace's absolute `Models` path in Settings. **Local NVIDIA Transformers** is the default backend and uses **MPS** on Apple silicon. Choose another device in Settings if needed.
3. Use the Parse menu for a page, document, or batch. **⌘R** parses the selected page. Cancel and Retry Failed are available.
4. Click a box or select a block. Content and Tables show native editable grids; Raw preserves full model and stage output.
5. Change Postprocess options and choose **Apply Formatting**. This does not call the model again. Corrections remain separate, with undo/redo and Restore Original.
6. Export with **⌘E**. Choose page, document, or batch and use the native save dialog. The selected scope must have completed results.

For native inference, build and validate the assets using the [Core AI instructions](docs/CORE_AI.md). Import `Conversion/evidence/pages/mixed.png`, select **Native Core AI**, and click **Use Validated Native Settings** for CPU float32, single pass, greedy decoding, and a 256-token cap. The `COREAI` result badge records which backend produced the prediction. The demo backend returns synthetic output.

## Private Python worker

The app starts one persistent child process:

```text
python Backend/studio_worker.py --stdio
```

Swift and Python exchange newline-delimited JSON with request IDs over private anonymous stdin/stdout pipes. Model instances stay cached between requests. Cancellation is handled during inference; a failed worker can restart on retry. Diagnostics go to stderr; stdout is reserved for protocol responses. **The app and worker do not start an HTTP server or bind a listening port.**

Native inference runs in Swift/Core AI; NVIDIA formatting still uses this private Python worker. Local Transformers runs inference and formatting in the worker. The optional vLLM backend makes outbound requests to an explicitly configured existing loopback/private-LAN server; Studio does not launch that server. No cloud provider is required. See [worker protocol and setup](Backend/README.md).

## Dependencies and clean setup

- Xcode 27 and its SDK. Native Core AI execution requires macOS 27+. The deployment target is macOS 14; older systems have not been runtime-tested with this build.
- Python 3.12 is tested. Reference dependencies are pinned in `Backend/requirements-model.txt`; lightweight formatting dependencies are in `requirements-lite.txt`.
- The pinned NVIDIA checkpoint, approximately 2.5 GB, for Transformers and conversion. Generated native assets are separate and currently occupy approximately 5.3 GB.
- LibreOffice for DOCX/PPTX/XLSX/RTF/HTML conversion. Set `NEMOTRON_SOFFICE` for a nonstandard executable location.

From the project root:

```sh
PYTHON=python3.12 Backend/setup.sh --model
Backend/.venv/bin/python Backend/download_model.py --output Models
Scripts/build.sh
```

The downloader pins the revision. First load verifies the checkpoint SHA-256 and pinned source, tokenizer, and configuration identities before executing model code. The app compiles and opens without weights and reports missing dependencies on parse. `Backend/setup.sh` without `--model` installs only formatting and vLLM client dependencies. The app also supports an existing `.venv-model` environment.

`Scripts/build.sh` regenerates the Xcode project and builds the locally signed Release app. Swift Package Manager provides core tests and `NativeValidate`. No Docker is used.

## Import, export, and limits

PDF, scanned PDF, PNG/JPEG, ImageIO images, and multipage TIFF become white-backed RGB pages. Office documents use isolated local conversion. Originals are copied into Application Support; rendering is incremental. DPI affects PDF/converted pages, while source images retain their dimensions.

Exports: Markdown with structural HTML tables, sanitized HTML, text, complete JSON, per-table CSV with a span manifest, hierarchical table JSON, and annotated PDF. Raw predictions, stages, options, and corrections remain inspectable. JSON references original files rather than embedding them. See [import/export details](docs/IMPORT_EXPORT.md).

Limits include fixed native geometries, CPU-only native validation, no newly generated searchable-PDF layer, and one editable grid per predicted block (all compound-table source remains available). HEIC and ODF have implementation paths but no explicit fixture coverage. Actual Linux vLLM execution has not been tested here. See the [implementation status](docs/IMPLEMENTATION_STATUS.md).

## Verification

```sh
Backend/.venv/bin/python -m pytest -q -c Backend/pytest.ini Backend/tests Backend/vendor/tests
Backend/.venv/bin/python -m pytest -q Conversion/tests
swift test
Backend/.venv/bin/python Conversion/validate_stdio.py
```

Use the model setup above for the Python suites and integration script. The geometry test reads pinned source from `Models`. Run `NEMOTRON_TEST_NATIVE=1 swift test` for native integration on macOS 27 with validated `NativeModels`. `validate_stdio.py` uses real weights and checks all three modes, reformatting, token exhaustion, cancellation and retry, persistent PID, and absence of worker network sockets.

Recorded: **82 Python helper/worker tests, 21 Swift tests including real native execution, and one geometry test covering seven geometries passed.** Real stdio parsing and both native/reference comparisons passed. Synthetic fixtures establish functionality, not corpus-wide accuracy or a speed benchmark. The Release app was also exercised interactively. [Core AI results and reproduction](docs/CORE_AI.md) give the exact validation boundaries.

## Source and licenses

| Path | Purpose |
|---|---|
| `Sources/ParseCore` | Document IO, models, export, private worker, native inference |
| `Sources/ParseStudio` | SwiftUI workspace, queue, settings, corrections and export |
| `Sources/NativeValidate` | Actual Swift/Core AI validation executable |
| `Backend` | Pinned NVIDIA helpers and persistent stdio worker |
| `Conversion` | Reference runs, geometry specialization, export and evidence |
| `Tests`, `Samples` | Native tests and synthetic input fixtures |

Original NVIDIA licenses and source hashes remain in `Backend/vendor/`. The model uses OpenMDW-1.1 and associated source licenses remain authoritative. The repository includes synthetic fixtures and selected validation reports. Build products, Python environments, model weights, native assets, and local traces are generated in the workspace.
