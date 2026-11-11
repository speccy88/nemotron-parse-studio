# Nemotron 3.5 Parse: Transformers and vLLM

## Installation

Use Python 3.12 and a fresh environment.

The reference serving contract is vLLM 0.26.0, Transformers 5.14.1,
huggingface-hub 1.24.0, and BF16 on one NVIDIA H100. The package declares the exact
core dependencies; its `[vllm,client]` extras add serving and the HTTP evaluation
client. See `pyproject.toml` for pinned direct dependencies.

Download the repository and install its serving extras:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "huggingface-hub==1.24.0"
hf download nvidia/Nemotron-3.5-Parse --local-dir nemotron-parse
cd nemotron-parse
export MODEL_REPO="$PWD"
python -m pip install ".[vllm,client]"
python -m pip check
```

One-token MTP and the compatible FA3 scheduler are enabled automatically for this model.

For Transformers-only inference, no Nemotron package is required. Follow the
[Quick Start](README.md#quick-start) to load the model directly from the Hub.

## Generation contract

The recommended prompt is identical in Transformers, vLLM, and evaluation:

`</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`

Use one RGB page per request, temperature 0, top-k 1, repetition penalty 1.1,
`max_tokens=8744`, and `skip_special_tokens=False`. Preserve the coordinate and
class tokens in the raw prediction used by postprocessing and scoring.

Boxes are normalized to the evaluated image content. Scale x by that image's
width and y by its height; no legacy center-pad inverse is needed. The image
processor accepts dynamically sized pages and applies the exported NaViT
preprocessing configuration.

## One-token MTP

`model.safetensors` contains the main model and all four trained MTP tensors.
No auxiliary weight sidecar is required. The model has **630,800,496 unique
trainable parameters**, including **2,099,200 MTP parameters**. The decoder
embedding and LM head share weights and are counted once. Eight stored scalar
buffers are excluded from the parameter count.

Transformers generation uses the normal decoder. vLLM enables the trained
one-token proposer automatically; it shares the decoder embedding and output
head.

```bash
vllm serve "$MODEL_REPO" \
  --served-model-name nemotron-parse \
  --trust-remote-code \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --data-parallel-size 1 \
  --max-model-len 16384 \
  --max-num-seqs 192 \
  --max-num-batched-tokens 16384 \
  --no-enable-prefix-caching \
  --limit-mm-per-prompt '{"image":1}' \
  --attention-config '{"flash_attn_version":3}' \
  --chat-template "$MODEL_REPO/chat_template.jinja" \
  --host 127.0.0.1 \
  --port 8000
```

This is the reference H100 throughput profile. Reduce `--max-num-seqs` when
using a GPU with less memory. Set `NP_VLLM_MTP=0` before startup to run the matched normal-decoding path. Clients use `model="nemotron-parse"`, matching
the served model name. Startup must show MTP with one speculative token; vLLM's
`SpecDecoding metrics` report its measured draft acceptance during generation.

## Verification and references

The [reference records](reference/README.md) document the previous release at
revision `cf22d043200e0173ee8939b3b211ed5ac815a12e`. Its fresh-install,
quick20/full500, and throughput measurements apply to the model and wheel
digests recorded there. They do not validate the replacement weights or rebuilt
wheel. See `runtime_compatibility.json` for artifact identities and validation status.

Set `NP_VLLM_MTP=0` before starting vLLM to use normal decoding.


### Docker alternative

From the downloaded folder, run the included Compose recipe:

```bash
docker compose up --build
```

Use the OpenAI-compatible client in the [Quick Start](README.md#quick-start). Container verification details are in [reference/README.md](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/reference/README.md).

### Prompt and logits-processor options

The recommended prompt extracts bounding boxes, semantic classes, markdown text, and text inside pictures:

`</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`

Use this same prompt with Transformers, vLLM, and the reference evaluation client.

This repository includes two optional logits processors:
* `NemotronParseRepetitionStopProcessor`: detects repeating n-grams during generation and forces the model to close the coordinate block when repeated structured output suggests a potential hallucination.
* `NemotronParseTableInsertionLogitsProcessor`: forces every block to follow a table structure, which can be useful when running the model on table image crops.

See `example_with_processor.py` for optional Transformers logits-processor examples.

An example of inference with the vLLM OpenAI-compatible server is available in `vllm_example.py`.

