# Implementation status

Verified November 10, 2026, on this Mac: macOS 27.2.1 (26B131), Xcode 27.0 (27A266a), Apple silicon. The Release app compiles and launches. This is a locally signed development build tied to the source workspace and installed model environments, not a notarized standalone installer.

## Delivered and verified

| Area | Implementation and evidence |
|---|---|
| Native app | Three-column SwiftUI workspace, native toolbar, keyboard shortcuts, resizable panes, document/page selection, search, failure filter, thumbnail/status list, zoom and fit, class-colored boxes and reading-order labels. Release build and interactive checks passed. |
| Import | Preserved source copy; incremental RGB rendering of PDFs/scans and ImageIO images; multipage TIFF; dimensions, points, rotation and page provenance. Tests cover PDF counts/DPI, scanned PDF, PNG/JPEG, TIFF, rotated pages, cancellation/retry. |
| Office conversion | Local LibreOffice with isolated macro-disabled profile and network denied on macOS. DOCX, PPTX, XLSX, RTF and passive HTML converted and checked for expected text. Timeout/cancel and unsafe-HTML rejection are implemented. |
| Real model | Exact pinned NVIDIA checkpoint and helpers. Local Transformers CPU/MPS, actual single/double/triple-pass runs; official crop merge and detector/parser region handling. Real test request counts were 1, 3 and 4. Token-limit errors retain partial output; cancel and retry passed. |
| Private Python transport | One persistent `studio_worker.py --stdio` child with correlated JSON lines over anonymous pipes, stderr diagnostics, cached model, responsive cancel/health, EOF shutdown and restart after failure. No local HTTP server or listening socket. Tests verify same PID across requests and zero worker network sockets. The launched app and its child also had zero network sockets during the UI check. |
| Native Core AI | Both vision operator blockers resolved by exact geometry specialization. Actual Swift image preparation, vision/adapter, prefill, explicit KV-cache decoding, token decoding and EOS. CPU float32, single pass, greedy, exact 512×256/640×800 profiles. Both native raw serializations equal pinned PyTorch CPU byte for byte. |
| Postprocessing | Pinned DocumentParser/PostprocessOptions/postprocess_document/postprocess_text/class-box/table helpers. All six controls, reset, source/result comparison, immutable raw and full stage envelopes. Reformatting calls no model. |
| Tables/charts | Native anchored grid with spans, empty cells, header roles, captions and original numeric strings. Original HTML, Markdown, LaTeX, CSV, record JSON and hierarchical JSON conversions are available. Malformed tables retain source and warnings. |
| Corrections | Text, class, reading order, inclusion, table cells and available crop candidates; separate overlay, restore, undo/redo. Cell initialization was fixed after the interactive check found an empty editor. Save/reformat/undo/redo/export were then verified in the rebuilt app. |
| Export | Page/document/batch Markdown, sanitized HTML, text, complete JSON, table CSV plus span manifest, hierarchical JSON, annotated PDF. Tests cover spans, escaping, inline formatting, captions, raw preservation, PDF dimensions, rotated coordinates and actual overlay pixels. |
| Provenance/performance | Backend badge, revision, effective settings, warnings, stages, render/load/inference/postprocess/total timings, request count and throughput; Python memory where available. Raw/corrected content stay separate. |
| Samples | Synthetic PDFs/scans, blank page, image/TIFF, Office documents, merged/empty cells, chart percentages/captions, multilingual prose, code and mathematics. These cover implementation behavior; not every sample has a real-model accuracy assertion. |

## Validation actually run

The repository includes the numerical reports, synthetic input pages, and test fixtures. Full traces, build logs, and UI exports are local records from the original validation workspace.

| Run | Result | Evidence |
|---|---|---|
| Python NVIDIA helper + worker tests | 82 passed | `Conversion/evidence/verification/python-tests.log` |
| Swift tests, with `NEMOTRON_TEST_NATIVE=1` | 21 passed, no failures/skips; includes real native import/parse/private formatting | `Conversion/evidence/verification/swift-tests.log` |
| Positional-geometry comparison/export test | 1 passed covering seven geometries | `Conversion/tests/test_vision_geometry.py`, `Conversion/evidence/verification/geometry-tests.log` |
| Actual stdio model integration | Single/double/triple, formatting raw invariance, partial token-limit error, cancellation/retry, same PID, zero sockets passed | `Conversion/evidence/stdio/all-report.json` and saved stage responses |
| Actual Swift/Core AI versus PyTorch CPU | Both profiles passed numerical gates and exact raw serialization | `Conversion/evidence/native-validation.json` |
| Xcode Release build after final editor fix | BUILD SUCCEEDED | `Conversion/evidence/verification/xcode-release.log` |
| Interactive app | Native parse, aligned boxes and tables, populated cell editor, correction surviving reformat, undo/redo, native Save dialog | `Conversion/evidence/native-ui-correction.json`, `Conversion/evidence/verification/ui-export.json` |

The UI export intentionally contains a synthetic Apples count correction from 3 to 4. Its effective table shows 4, while the immutable model raw and original hierarchy retain 3. The complete UI raw matches the separately validated native mixed-page prediction. After export, the sample was restored to its original value and formatting defaults. The app remains open with the native result.

Recorded command-line timings are individual functional runs with varying cache/load states. No production accuracy, speed ranking, or broad-language coverage is claimed. [Core AI details](CORE_AI.md) give numerical thresholds and reproduction commands. Older environment, HTTP bridge and unsuccessful conversion reports remain historical evidence, not current service/configuration descriptions.

## Limits and unsupported cases

- **General native geometry:** only the two validated profiles work in Core AI. Arbitrary PDF/image dimensions use the explicitly selected Transformers backend. There is no silent resize or fallback masquerading as Core AI.
- **Native modes/compute:** native sampling, multipass, GPU/ANE, dynamic shapes and unbounded KV growth are not implemented/validated. The fixed 1024-token cache includes the visual prefix; native requests exceeding remaining capacity are rejected. The validated preset uses 256 generated tokens.
- **vLLM/CUDA:** configurable outbound loopback/private-LAN vLLM is implemented with connection checks, disabled proxy inheritance and no redirects. A Linux vLLM server and CUDA execution were not exercised here. Studio starts no server. Server-reported model names do not cryptographically prove remote checkpoint identity.
- **Additional formats:** HEIC and other ImageIO types, plus ODT/ODP/ODS, have implementation paths but no explicit fixture coverage. Complex Office fidelity depends on fonts/LibreOffice. Active or resource-referencing HTML is rejected; encrypted PDFs require an unlocked copy.
- **Searchable PDF:** an accurately aligned new text layer is unsupported because model block boxes do not supply line/glyph geometry. Annotated PDF preserves selectable source text when present.
- **Compound tables:** only the first table in a predicted block is editable/exported as a native grid. All official conversions and raw source remain in structured metadata, with a warning. CSV cannot express spans itself; its companion manifest does.
- **Table conversion versus correction:** the conversion-format tabs show original NVIDIA conversions. Corrected exports use effective cells and preserve the original hierarchy separately; they do not claim the original hierarchy was recomputed from manual edits.
- **Packaging:** Python environments and `.aimodel` assets are workspace dependencies rather than bundled redistributable resources. No App Store sandbox/notarization workflow was attempted. Native API availability is guarded; older macOS versions have not been runtime-tested.
- **Metrics:** Core AI currently reports timings and generated tokens, but not device peak memory. Rendering is per page; document-wide benchmarking is not provided.

No cloud service, replacement OCR model, or Docker workflow was used. NVIDIA licenses and source hashes are retained in `Backend/vendor`; generated weights and build outputs are excluded from source-control rules. See [README](../README.md) and [import/export behavior](IMPORT_EXPORT.md).
