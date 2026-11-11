---
license: openmdw-1.1
pipeline_tag: image-text-to-text
tags:
- VLM
- OCR
- Parse
library_name: transformers
---
# Nemotron 3.5 Parse

### Description:
Nemotron 3.5 Parse is a document image parser that produces structured, machine-readable text with semantic layout classes, bounding boxes, and reading order. It combines a C-RADIOv4-SO400M vision encoder with a 10-layer mBART decoder that consumes the visual sequence as a causal prefix. The release uses dynamic rectangular NaViT preprocessing, content-normalized coordinates, grouped-query self-attention (16 query heads and 8 key/value heads), a 72,505-token multilingual tokenizer, and chart-aware output through the `<class_Chart>` token. It is intended for document understanding, information retrieval, data extraction, and multimodal data-curation workflows.<br>

Nemotron 3.5 Parse was developed by NVIDIA as a part of Nemotron.

This model is ready for commercial or non-commercial use. <br>

## Benchmark highlights

| Model | Mean ↑ | Tables ↑ | Charts ↑ | Content Faithfulness ↑ | Semantic Formatting ↑ | Visual Grounding ↑ |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Nemotron 3.5 Parse** | **75.20** | **84.67** | **64.40** | **87.96** | **68.28** | **70.67** |

Results are for the Nemotron 3.5 Parse triple-pass pipeline with region union on the 2,078 unique ParseBench pages. See the [reproduction instructions](reference/parsebench/pipeline/README.md) and run with `--mode triple --triple-regions union`. [Evaluation results](reference/parsebench/pipeline/validation_metrics.json).

### License/Terms of Use:
Governing Terms: Your use of this model is governed by the [OpenMDW License Agreement, version 1.1](https://openmdw.ai/license/1-1/). Use of the tokenizer included in this model is governed by the [MIT license and tokenizer attribution](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/TOKENIZER_ATTRIBUTION.md). See [third-party notices](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/THIRD_PARTY_NOTICES.md) for separately installed dependencies.<br>

Tokenizer source and modifications are documented in [TOKENIZER_ATTRIBUTION.md](TOKENIZER_ATTRIBUTION.md).

### Deployment Geography:
Global<br>

### Use Case: <br>
Nemotron 3.5 Parse is designed for developers and teams building document intelligence, retrieval-augmented generation (RAG), curator, extractor, and agentic AI applications. It can convert scanned or rendered PDFs, presentation slides, forms, reports, tables, and mixed-content document pages into structured outputs for downstream indexing, retrieval, analytics, model training-data creation, and human-in-the-loop review.<br>

### Release Date:  <br>
Hugging Face 10/17/2026 via https://huggingface.co/nvidia/Nemotron-3.5-Parse <br>

## References(s):
* [Hugging Face Transformers mBART documentation](https://huggingface.co/docs/transformers/en/model_doc/mbart) <br>
* [NVIDIA C-RADIOv4-SO400M](https://huggingface.co/nvidia/C-RADIOv4-SO400M) <br>

## Model Architecture:
**Architecture Type:** Transformer<br>

**Network Architecture:** <br>
* Vision Encoder: NVIDIA C-RADIOv4-SO400M (`vit_so400m_patch16_224`) with dynamic rectangular NaViT preprocessing<br>
* Adapter Layer: 1D convolutions and normalization layers that compress the vision latent sequence before decoding<br>
* Decoder: 10-layer mBART causal visual-prefix decoder with 16 query heads and 8 key/value heads<br>
* Auxiliary Prediction Head: One trained one-step multi-token-prediction (MTP) head is included in `model.safetensors`. Transformers uses the normal decoder for generation; the vLLM plugin enables the bundled head for one-token speculative decoding.<br>
* Tokenizer: 72,505 entries with multilingual and task/control tokens including `<predict_bbox>`, `<predict_classes>`, `<predict_text_in_pic>`, and `<class_Chart>`; the decoder embedding is padded to 72,512 rows. Use of the tokenizer included in this model is governed by the [MIT license and tokenizer attribution](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/TOKENIZER_ATTRIBUTION.md).<br>
* Number of Parameters: 630,800,496 unique trainable parameters (approximately 0.631B), including the MTP head; shared decoder embedding/output weights are counted once<br>

## Input(s): <br>
**Input Type(s):** Image, Text<br>

**Input Format(s):** <br>
* Image: Red, Green, Blue (RGB)<br>
* Text: Prompt string<br>

**Input Parameters:** <br>
* Image: Two-Dimensional (2D)<br>
* Text: One-Dimensional (1D)<br>

**Other Properties Related to Input:** <br>
* Recommended maximum input resolution (Width, Height): 1664, 2048<br>
* Channel Count: 3<br>
* Prompt format: one task prompt composed from the supported control tokens. The default prompt extracts bounding boxes, semantic classes, and text in markdown format: `</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>`.<br>

## Output(s): <br>
**Output Type(s):** Text<br>

**Output Format(s):** String<br>

**Output Parameters:** One-Dimensional (1D)<br>

**Other Properties Related to Output:** Nemotron 3.5 Parse returns a string encoding document text, semantic element classes, and content-normalized bounding boxes. Scale x coordinates by the evaluated image width and y coordinates by its height. Postprocessing utilities can convert table or chart-associated text into LaTeX, HTML, markdown, JSON, or CSV where supported.<br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA hardware and software frameworks, the model achieves faster training and inference times compared to CPU-only solutions. <br>

## Quick Start

### Transformers installation

Use Python 3.12 and a fresh environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install "transformers==5.14.1" "torchvision==0.26.0" "timm==1.0.22"
```

Run `hf auth login` if repository access requires authentication. Transformers downloads the model weights, processor, and custom model code automatically. No Nemotron package installation is needed for Transformers inference.

### Usage example

```python
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig

model_path = "nvidia/Nemotron-3.5-Parse"
device = "cuda:0"

model = AutoModel.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16
).to(device).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

image = Image.open("path/to/your/image.jpg").convert("RGB")
task_prompt = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"

inputs = processor(images=[image], text=task_prompt, return_tensors="pt", add_special_tokens=False).to(device)

generation_config = GenerationConfig.from_pretrained(model_path)
generation_config.max_new_tokens = 8744

with torch.inference_mode():
    outputs = model.generate(**inputs, generation_config=generation_config)

generated_text = processor.batch_decode(outputs, skip_special_tokens=False)[0]
print(generated_text)
```

### Postprocessing

The shared helpers render a complete document as Markdown with HTML tables, and
return structured blocks with normalized boxes and optional pixel boxes. They
preserve model reading order, table spans, blank cells, and in-cell line breaks.
They also decode rich table formatting, remove literal duplicate elements, repair
empty serialization prefixes, and normalize existing list bullets and discretionary
soft hyphens in text. Code and math are protected from typography cleanup.

Download the helpers and their shared implementation into your working directory
(no Nemotron package installation is needed for this path):

```bash
python -m pip install "beautifulsoup4==4.13.3"
hf download nvidia/Nemotron-3.5-Parse \
  --include 'postprocessing.py' 'latex2html.py' 'nemotron_parse/**' --local-dir .
```

Continue from the Transformers example above:

```python
from pathlib import Path
from PIL import ImageDraw
from postprocessing import postprocess_document

document = postprocess_document(generated_text, image_size=image.size)
Path("prediction.md").write_text(document["markdown"], encoding="utf-8")
draw = ImageDraw.Draw(image)
for block in document["blocks"]:
    draw.rectangle(block["bbox_pixels"], outline="red", width=2)
image.save("prediction.png")
```

The existing `extract_classes_bboxes`, `transform_bbox_to_original`, and
`postprocess_text` functions remain available. Supported `table_format` values:
`latex` | `HTML` | `markdown` | `json` | `json_hierarchical` | `csv`.
HTML and hierarchical JSON retain merged-cell structure; flat Markdown, CSV, and
record JSON flatten spans. Headerless tables use neutral column names in flat
exports, retaining every data row.

Explicit chart titles, subtitles, and panel labels are preserved with their tables.
Header inference uses predicted spans and existing bold markup; it can be disabled.
Numeric value/percentage column splitting and geometric chart-caption relocation
are optional presentation heuristics, disabled by default. See
[postprocessing and multipass usage](POSTPROCESSING.md) for controls and examples.

## Inference with vLLM

vLLM requires the Nemotron plugin for this architecture and its MTP head. In a fresh Python 3.12 environment, download the repository and install the serving extras:

```bash
python -m pip install "huggingface-hub==1.24.0"
hf download nvidia/Nemotron-3.5-Parse --local-dir nemotron-parse
cd nemotron-parse
python -m pip install ".[vllm,client]"
```

One-token MTP and the compatible FA3 scheduler are enabled automatically.

### vLLM Inference example

#### Option 1: Python inference with MTP

```python
from PIL import Image
from vllm import LLM, SamplingParams

def main():
    llm = LLM(
        model=".",
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=16384,
        max_num_seqs=64,
        max_num_batched_tokens=16384,
        limit_mm_per_prompt={"image": 1},
        enable_prefix_caching=False,
        attention_config={"flash_attn_version": 3},
    )
    sampling_params = SamplingParams(
        temperature=0,
        top_k=1,
        repetition_penalty=1.1,
        max_tokens=8744,
        skip_special_tokens=False,
    )
    image = Image.open("<YOUR-IMAGE-PATH>").convert("RGB")
    outputs = llm.generate([{
        "prompt": "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>",
        "multi_modal_data": {"image": image},
    }], sampling_params)
    print(outputs[0].outputs[0].text)

if __name__ == "__main__":
    main()
```

#### Option 2: vLLM serve with MTP

```bash
vllm serve . \
    --served-model-name nemotron-parse \
    --dtype bfloat16 \
    --max-model-len 16384 \
    --max-num-seqs 192 \
    --max-num-batched-tokens 16384 \
    --no-enable-prefix-caching \
    --limit-mm-per-prompt '{"image":1}' \
    --attention-config '{"flash_attn_version":3}' \
    --trust-remote-code \
    --port 8000 \
    --chat-template chat_template.jinja
```

Then run inference through the OpenAI-compatible API:

```python
import base64
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

with open("<YOUR-IMAGE-PATH>", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode("utf-8")

prompt_text = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"

resp = client.chat.completions.create(
    model="nemotron-parse",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt_text,
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}",
                    },
                },
            ],
        }
    ],
    max_tokens=8744,
    temperature=0.0,
    extra_body={
        "repetition_penalty": 1.1,
        "top_k": 1,
        "skip_special_tokens": False,
    },
)
print(resp.choices[0].message.content)
```

For Docker deployment and optional generation controls, see [the serving guide](VLLM_COMPATIBILITY.md).

## Software Integration:
**Runtime Engine(s):**
* Transformers <br>
* vLLM <br>

**Reference Hardware:** NVIDIA Hopper H100, BF16. The FA3/MTP speed profile above is measured on H100.<br>

**Supported Operating System(s):**
* Linux <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels is essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>

## Model Version(s):
Nemotron 3.5 Parse · October 17, 2026 release <br>

This AI model can be embedded as an Application Programming Interface (API) call through the documented Transformers or vLLM serving interfaces.

## Training, Testing, and Evaluation Datasets:

### Training Dataset

**Data Modality:** <br>
* Image <br>
* Text <br>

**Image Training Data Size:** <br>
* 1 Million to 1 Billion Images <br>

**Text Training Data Size:** <br>
* 1 Billion to 10 Trillion Tokens <br>

**Data Collection Method by dataset:** <br>
* Automated, Manually-Collected, Synthetic <br>

**Labeling Method by dataset:** Automated, Manually-Labeled, Synthetic <br>
**Labeling Details:** Labels are produced through a combination of automated OCR and layout annotation, manual annotation, and synthetic generation. The combination varies by constituent dataset. <br>

**Properties:** The training set contains millions of image-text items aggregated across large document, table, and layout datasets. The data consists of document-page and table images paired with OCR text, bounding boxes, and layout labels. Sources include rendered digital documents, scientific papers, PDFs, Wikipedia-style pages, and synthetic document, table, word, and character renderings. Annotations come from OCR and layout models, third-party OCR services, synthetic-generation pipelines, and human labeling. <br>

### Testing Dataset

**Data Collection Method by dataset:** Automated, Manually-Collected, Synthetic  
**Collection Details:** Sources combine automated collection, manually collected material, and synthetic generation; methods vary by constituent dataset.

**Labeling Method by dataset:** Automated, Manually-Labeled, Synthetic  
**Labeling Details:** Labels are produced through a combination of automated OCR and layout annotation, manual annotation, and synthetic generation. The combination varies by constituent dataset.

**Properties:** Tested on internal and public document-understanding benchmarks using document-page images and reference text, layout classes, bounding boxes, and structure annotations. Coverage includes OCR quality, layout structure, table parsing, reading order, and visual grounding across document types and languages.

### Evaluation Dataset

**Data Collection Method by dataset:** Automated, Manually-Collected, Synthetic  
**Collection Details:** Sources combine automated collection, manually collected material, and synthetic generation; methods vary by constituent dataset.

**Labeling Method by dataset:** Automated, Manually-Labeled, Synthetic  
**Labeling Details:** Labels are produced through a combination of automated OCR and layout annotation, manual annotation, and synthetic generation. The combination varies by constituent dataset.

**Properties:** Evaluated on internal and public document-understanding benchmarks covering OCR quality, layout structure, table parsing, reading order, and visual grounding. The ParseBench reference evaluation covers 2,078 unique document pages. Additional benchmarks cover Indic-language OCR and multilingual synthetic OCR across the scripts listed below. <br>

Benchmark coverage: <br>
* ParseBench evaluates document parsing across text fidelity, semantic formatting, tables, charts, and visual grounding with layout boxes/classes. <br>
* IndicVisionBench evaluates OCR quality for Indic-language pages, including Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, and Telugu. <br>
* MOSCAR evaluates multilingual synthetic OCR across broad script coverage, including Latin, Arabic, Cyrillic, Han, Hangul, Japanese, Indic scripts, Hebrew, Thai, Greek, and others. <br>

### Evaluation Results

#### IndicVisionBench OCR

Single-pass OCR results on 876 test pages across 10 languages. ANLS-char and character error rate (CER) are shown as percentages. CER includes insertion errors and can exceed 100%.

| Language | Pages | Parse 2.0 ANLS-char ↑ | Parse 3.5 ANLS-char ↑ | Parse 2.0 CER ↓ | Parse 3.5 CER ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Overall | 876 | 72.12 | 83.73 | 52.67 | 16.00 |
| Bengali | 89 | 55.47 | 90.25 | 68.92 | 7.69 |
| Gujarati | 90 | 40.70 | 66.96 | 127.53 | 34.08 |
| Hindi | 85 | 85.03 | 88.34 | 16.32 | 9.36 |
| Kannada | 89 | 87.54 | 91.37 | 19.25 | 7.33 |
| Malayalam | 88 | 48.89 | 63.15 | 75.55 | 42.47 |
| Marathi | 83 | 92.95 | 93.19 | 6.50 | 6.01 |
| Odia | 86 | 67.77 | 73.61 | 71.46 | 24.61 |
| Punjabi | 89 | 83.25 | 91.95 | 66.57 | 7.30 |
| Tamil | 88 | 81.77 | 91.24 | 42.05 | 7.41 |
| Telugu | 89 | 79.86 | 87.75 | 43.57 | 10.53 |

#### MOSCAR multilingual OCR

Single-pass OCR results on 8,111 held-out synthetic pages covering all 163 configured language/script entries across 26 script groups. Character bag F1 (BoC F1, higher is better) and CER (lower is better) are percentages. Overall CER is computed across all reference characters, so it is not an unweighted average of the script rows.

| Script | Pages | Parse 2.0 BoC F1 ↑ | Parse 3.5 BoC F1 ↑ | Parse 2.0 CER ↓ | Parse 3.5 CER ↓ |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Overall | 8111 | 97.44 | 99.51 | 11.55 | 1.35 |
| Arabic | 824 | 91.27 | 98.87 | 20.17 | 5.80 |
| Armenian | 50 | 99.38 | 99.92 | 1.15 | 0.09 |
| Bengali | 100 | 98.82 | 99.80 | 1.78 | 0.34 |
| Cyrillic | 597 | 98.75 | 99.69 | 2.36 | 0.89 |
| Devanagari | 350 | 97.99 | 99.47 | 5.41 | 0.79 |
| Ethiopic | 100 | 98.31 | 99.89 | 5.52 | 0.14 |
| Georgian | 50 | 99.91 | 99.99 | 0.13 | 0.02 |
| Greek | 50 | 99.63 | 99.93 | 1.07 | 0.10 |
| Gujarati | 50 | 99.52 | 99.92 | 0.74 | 0.15 |
| Gurmukhi | 50 | 94.85 | 99.91 | 14.87 | 0.16 |
| Han (Simplified) | 50 | 98.33 | 99.50 | 1.99 | 1.05 |
| Han (Traditional) | 100 | 97.83 | 99.68 | 3.10 | 0.54 |
| Hangul | 50 | 98.82 | 99.62 | 0.99 | 0.43 |
| Hebrew | 100 | 97.62 | 99.79 | 5.79 | 4.16 |
| Japanese | 50 | 98.66 | 99.77 | 2.61 | 0.47 |
| Kannada | 50 | 95.73 | 99.84 | 5.60 | 0.24 |
| Khmer | 50 | 96.56 | 99.63 | 5.79 | 0.71 |
| Lao | 50 | 98.74 | 99.74 | 2.26 | 0.46 |
| Latin | 5040 | 98.20 | 99.53 | 12.82 | 0.79 |
| Malayalam | 50 | 97.65 | 99.27 | 6.46 | 1.10 |
| Myanmar | 50 | 87.83 | 99.56 | 45.41 | 1.47 |
| Odia | 50 | 97.79 | 99.74 | 7.58 | 0.41 |
| Sinhala | 50 | 96.52 | 99.91 | 15.28 | 2.48 |
| Tamil | 50 | 97.89 | 99.92 | 7.92 | 0.14 |
| Telugu | 50 | 96.69 | 99.69 | 9.33 | 0.47 |
| Thai | 50 | 99.66 | 99.95 | 0.47 | 0.06 |

Both comparisons use the [released Parse 2.0 weights](https://huggingface.co/nvidia/NVIDIA-Nemotron-Parse-2.0) and current Parse 3.5 weights, identical page sets, greedy decoding, and the same scorer for each benchmark. Each model uses its documented default task prompt. These single-pass OCR evaluations are separate from the triple-pass ParseBench pipeline above. All outputs, including those reaching the token limit, are included. See the [full MOSCAR results for all 163 language/script entries](reference/multilingual/moscar_by_language.md), [downloadable results](reference/multilingual/README.md), and [evaluation settings](reference/multilingual/README.md#evaluation-settings).

Inference reference outputs and verification instructions are available in [reference/](https://huggingface.co/nvidia/Nemotron-3.5-Parse/blob/main/reference/README.md).

## Inference:
**Acceleration Engine:** Transformers, vLLM <br>
**Test Hardware:** <br>
* NVIDIA Hopper H100 <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

Please make sure you have proper rights and permissions for all input image and video content. If an input image contains people, personal health information, confidential business data, or intellectual property, the model may extract or reproduce visible text from that content. <br>

For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](bias.md), [Explainability](explainability.md), [Safety & Security](safety.md), and [Privacy](privacy.md) subcards. <br>
Please report model quality, risk, security vulnerabilities, or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/). <br>

## Single-, double-, and triple-pass image parsing

After installing the package and starting the vLLM server above, parse an ordinary
page image with the general wrapper:

```python
from nemotron_parse.inference import DocumentParser

with DocumentParser(base_url="http://localhost:8000/v1", model="nemotron-parse") as parser:
    document = parser.parse_double("page.png")
    # Or: document = parser.parse_triple("page.png", triple_regions="union")
print(document["markdown"])
```

Double-pass parsing refines Table/Chart regions predicted by the full-page parse.
Triple-pass parsing adds a separate detection pass before crop refinement. A crop
stage can issue multiple requests; these modes cost more than one full-page call.
The result includes structured blocks, the original stage outputs, and the actual
sampling and crop settings. Empty or truncated responses raise an error rather
than silently returning a complete-looking document.

```bash
nemotron-parse --image page.png --mode triple --output parsed-page
# A directory of images:
nemotron-parse --image-dir pages --mode double --output parsed-pages
```

The wrapper accepts images without benchmark annotations or manifests. It shares
the reference pipeline's structural merging and general rendering rules, while
keeping presentation heuristics configurable. Its defaults do not imply the
reported ParseBench score; use the [pinned reproduction pipeline](reference/parsebench/pipeline/README.md)
for that contract. See [POSTPROCESSING.md](POSTPROCESSING.md) for the download-only
route, custom inference backends, stage settings, and limitations.
