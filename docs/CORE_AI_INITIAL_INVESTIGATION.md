# Core AI investigation and reference validation

This report records the initial macOS 26 investigation. See the [current Core AI results](CORE_AI.md) for the macOS 27 implementation and private worker.

Native Core AI is **not validated and is unavailable on this machine**. The installed host is macOS 26.6.2 (25G83), with Xcode 27.0 (27A266a). Apple specifies macOS 27.0 or newer for Core AI. An SDK does not supply the host runtime. `xcrun --find coreai-build` also fails on this installation. The application must report this limitation and use the explicitly selected local Python Transformers or LAN vLLM backend.

## Sources and exact scope

- NVIDIA model and all model Python sources: revision [`a9a1febb1b9bf497fc67bd1d23fa33e170b249db`](https://huggingface.co/nvidia/Nemotron-3.5-Parse/tree/a9a1febb1b9bf497fc67bd1d23fa33e170b249db).
- [Core AI framework availability and preparation](https://developer.apple.com/documentation/coreai.md).
- [Apple model catalog](https://github.com/apple/coreai-models/blob/main/models/README.md), inspected November 10, 2026; it contains no Nemotron Parse recipe.
- [Apple EDSR conversion recipe](https://github.com/apple/coreai-models/blob/main/models/edsr/export.py): real `TorchConverter`, `get_decomp_table`, `to_coreai`, and `save_asset` API example. A source copy and its BSD license are retained in `Conversion/evidence/`.

Apple repository `main` is research evidence, not a replacement for the pinned NVIDIA implementation. Package versions used by this experiment are recorded with the evidence.

## Architecture inspected

`hf_nemotron_parse_modeling.py`, `hf_nemotron_parse_processor.py`, `hf_prefix_geometry.py`, and `config.json` establish the following boundaries:

1. **Image preparation.** RGB pages are reduced only when exceeding the configured 2048 × 1664 bounds, then padded to 16-pixel rows and 64-pixel columns. Normalized 16 × 16 RGB patches flatten to 768 values. Content dimensions remain distinct from padded dimensions.
2. **Vision encoder.** C-RADIOv4-SO400M has 27 transformer blocks, width 1152, and 16 attention heads. A 768→1152 projection and ten class tokens precede them. The learned 128 × 128 positional field uses content-normalized bilinear interpolation and grid sampling. This geometry must not be replaced with generic resize/position handling.
3. **Convolutional adapter.** A 1152→1024 1D projection and layer normalization are followed by a (1,4) stride-(1,4) 2D convolution and normalization. Summary tokens 0 and 1 are concatenated, projected 2304→1024 and appended to the visual prefix.
4. **Decoder.** Ten causal mBART-style layers, hidden width 1024, FFN width 4096, 16 query heads, eight KV heads, vocabulary 72512, and tied token embedding/output weights. The visual prefix and task control tokens enter the same causal self-attention stream. The inspected decoder has no additional learned positional-embedding stage; the vision geometry is encoded upstream.
5. **Cache.** Upstream `DynamicCache` retains per-layer keys and values and `_nemotron_visual_mask`. First generation step adds vision; later steps append token embeddings. Maximum visual-plus-text context is 16384. A native port needs equivalent causal masking, visual masks, cache growth, EOS handling, repetition penalty, token decoding, and cache reset.

## Implemented experiments

- `Conversion/probe_environment.py` records OS, Xcode, Core AI tool availability, PyTorch and MPS capability.
- `Conversion/make_validation_pages.py` creates nonconfidential RGB samples: plain text, a mixed table/chart page and an empty page.
- `Conversion/run_reference.py` loads the exact local model, preserves full control-token output, and records timing, input and output hashes, package versions and token-limit status. It does not call a cloud service.
- `Conversion/staged_export.py` separates the actual vision encoder, exact convolutional adapter and functional decoder with explicit KV tensors. It compares adapter output, prefill logits and one cached decoding step with the original model. `--torch-export` additionally compares exported PyTorch graphs. `--aimodel` attempts the documented Core AI conversion API and captures errors.

The staged conversion is intentionally **experimental**: fixed one-page geometry and explicit cache tensors are useful for numerical debugging, but are not a complete native dynamic-cache autoregressive implementation. Any generated `.aimodel` is labelled unvalidated until executed through Core AI and compared against reference outputs. No converted model is silently selected by the application.

## Reproduction

From the repository root, after local model installation:

```sh
.venv-model/bin/python Conversion/probe_environment.py
.venv-model/bin/python Conversion/make_validation_pages.py
.venv-model/bin/python Conversion/run_reference.py \
  --image Conversion/evidence/pages/simple.png \
  --output Conversion/evidence/reference-simple --device mps --max-new-tokens 256
.venv-model/bin/python Conversion/staged_export.py \
  --image Conversion/evidence/pages/simple.png --torch-export
```

Use `--device cpu` if MPS is unavailable. MPS float32 is an explicitly recorded execution adaptation; NVIDIA's reference script uses CPU float32 or CUDA bfloat16. Do not compare speed or bitwise equality across those execution configurations.

For Core AI conversion, install `coreai-core==1.0.0b3` and `coreai-torch==0.4.3` in an isolated environment with their declared dependencies. Apple currently pins a different Torch version from NVIDIA's runtime; this is another independently validated stage, not an in-place upgrade of the inference environment. Run the same script with `--aimodel`, then on macOS 27+ implement and validate Core AI execution, the tokenizer and complete decode loop before enabling a native backend.

## Acceptance gates still required for native inference

- Numerical Core AI comparison of vision features and adapter outputs over different padded/content dimensions.
- Decoder prefill and multi-step logits/token comparison with correct cache masking and EOS stopping.
- Table, chart, multilingual and mixed-page end-to-end output comparisons.
- Actual Swift model loading, image preprocessing, tokenizer decoding and generation loop.
- Memory and performance measurements at representative page resolution.

Reference parsing success is evidence for the Python backend only. It is not evidence of `.aimodel` compatibility.

## Measured results on this Mac

All pages below are synthetic test documents generated by the included script. These are functional checks, not a benchmark corpus.

| Test | Observed result |
|---|---|
| Local checkpoint integrity | 2,523,266,640 bytes; SHA256 `81dd7609c589f3694ce75520c25af5010953231358de6aec4d03e8c6aac787e9`, exactly matching NVIDIA `runtime_compatibility.json` |
| Simple page, 512 × 256, MPS float32 | Real text, class and bounding box; 4.68 s load, 5.34 s inference; completed at EOS, no token exhaustion |
| Mixed page, 640 × 800, MPS float32 | Correct table items/counts/prices, chart 60%/40%, caption and text; 3.75 s load, 9.65 s inference; 190 total output tokens |
| Local companion HTTP | `/parse` produced six blocks and two table payloads (table plus chart rendering); 10.46 s inference, 6.39 s load |
| Reformat existing prediction | `/postprocess` enabled chart value/percent splitting and caption relocation; immutable raw prediction remained identical |
| PyTorch partition parity, simple page | Adapter max absolute error 0; decoder prefill 4.1962e-5; cached one-token step 0 |
| `torch.export` graph execution | Vision, adapter, prefill and decoder-step graph outputs matched their wrapper outputs exactly on the fixed sample |
| Core AI vision conversion | **Failed** with unsupported `aten._upsample_bilinear2d_aa.default` and `aten.grid_sampler_2d.default` in the content-normalized positional geometry |

Exact results, hashes, raw predictions, package locks and traces are retained in `Conversion/evidence/`. Full Core AI numerical validation has not been performed. A geometry-preserving custom lowering or separately validated positional preprocessing would be needed to address the actual conversion error; disabling antialiasing or dropping grid sampling is not an acceptable silent workaround.

Additional real companion checks passed: double pass used three inference requests (15.58 s inference); triple pass used four (13.94 s), each returning six structured blocks. A deliberately exhausted 16-token request returned HTTP 422 with partial raw prediction and stage evidence. Cancelling an active triple-pass request stopped after the active decoder step, returned retained partial output within 2.01 s of the cancellation request, and a subsequent single-pass retry succeeded. Timing differences between modes on one tiny document are not a speed ranking.

Core AI conversion was also attempted independently for the adapter and decoder so that the vision failure did not conceal later-stage issues. `coreai-torch==0.4.3` and `coreai-core==1.0.0b3`, with Torch explicitly pinned back to 2.11.0, successfully wrote experimental `adapter.aimodel` (30 MB), `decoder_prefill.aimodel` (736 MB), and `decoder_step.aimodel` (740 MB) under `Conversion/evidence/coreai-adapter-decoder/`. They are ignored as generated artifacts. **These assets have never executed in Core AI.** They do not include a converted vision encoder, dynamic geometry, a complete tokenizer/decode loop, or a working Swift pipeline, and must not be presented as a native parsing backend. Their prefill/step shapes are fixed to the synthetic sample. Inspect `report.json` for the exact scope.

The initial bridge evidence uses the original camelCase metric keys. The final bridge schema uses snake_case metric keys; this schema-only adjustment does not rerun or relabel the recorded measurements. The integration scripts accept the updated request-count key.

The standalone reference runs used the checkpoint generation config's repetition penalty **1.1**, greedy decoding, and seed 0. The companion runs used NVIDIA `GenerationSettings` with repetition penalty **1.05** and an explicit temperature override of 0 for reproducibility. These are two documented entry points from the same pinned revision. They are not advertised as byte-identical settings: the mixed page differed in one predicted x-coordinate by 0.001, while the extracted text and table values matched. Effective generation settings are retained in the reports and companion inference metadata.
