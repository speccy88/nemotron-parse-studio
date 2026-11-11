# Document postprocessing and multipass parsing

The public helpers operate on model predictions and ordinary page images. They do
not load benchmark labels, scorer code, category IDs, reference answers, or
page-specific rules. They are adapted from the general conversion and crop-merge
components of the bundled ParseBench pipeline. The pinned reproduction bundle
remains unchanged so its recorded hashes and results stay reproducible.

## Installation

For vLLM, use the model card's repository-source installation:

```bash
hf download nvidia/Nemotron-3.5-Parse --local-dir nemotron-parse
cd nemotron-parse
python -m pip install '.[vllm,client]'
```

This installs the plugin, public `nemotron_parse` Python package, compatibility
modules, and `nemotron-parse` command. The repository also includes a matching
`nemotron_parse-3.5.0.post3-py3-none-any.whl`; installing that wheel with `[vllm,client]`
is an alternative. The older wheel is retained for the pinned reproduction path.

For Transformers, keep the model card's inference dependencies. To download the
helpers without installing the package:

```bash
python -m pip install 'beautifulsoup4==4.13.3' 'Pillow==12.2.0' 'numpy==2.2.6' 'httpx==0.28.1'
hf download nvidia/Nemotron-3.5-Parse \
  --include 'postprocessing.py' 'latex2html.py' 'nemotron_parse/**' --local-dir .
```

Run Python in that directory, or put it on your Python import path. Basic text/table
postprocessing only needs Beautiful Soup; the image wrappers also need Pillow and
NumPy. The HTTP backend needs httpx. Downloaded source and installed packages
provide the same public API.

## Render an existing prediction

```python
from nemotron_parse import postprocess_document, PostprocessOptions

options = PostprocessOptions(
    infer_headers=True,
    render_charts=True,
    split_chart_value_percent=False,
    relocate_chart_captions=False,
)
document = postprocess_document(raw_model_output, image_size=(width, height), options=options)
print(document['markdown'])
print(document['blocks'])
```

`raw_out` contains the cleaned model serialization. Each block has its original
class, normalized `bbox`, literal `text`, and rendered `content`. Supplying
`image_size` adds clipped, ordered `bbox_pixels`. `markdown` renders the complete
page in model order, with section headings and HTML tables for span preservation.
Raw per-block text remains available independently of presentation changes.
Malformed table markup that cannot be converted is preserved rather than passed
through a lossy row splitter; document `warnings` identify these blocks.

Default cleanup preserves empty cells, spans, line breaks, nested inline styles,
and escaped literals. It removes exact repeated blocks at identical coordinates,
repairs empty incomplete serialization prefixes, removes internal continuation
markers, and normalizes existing bullets and soft hyphens in textual roles.
Code/math roles and marked code/math text are protected from soft-hyphen cleanup.
Table conversion is applied to Table/Chart blocks; literal LaTeX in prose, code,
formulas, or Picture blocks retains its source representation.
Header inference uses predicted span topology and existing bold styling, never an
English vocabulary list. It can still make mistakes; disable `infer_headers` to
use only explicit predicted headers. Unknown classes and page numbers are retained.

Chart titles, subtitles, and explicit panel labels can be rendered with their
associated tables. Two additional presentation choices are disabled by default:

- `split_chart_value_percent=True` splits consistently predicted numeric pairs
  such as `32 (16.6%)` into Value and Percent subcolumns. It requires at least two
  matching nonempty cells, adds those column labels, and does not calculate values.
- `relocate_chart_captions=True` associates a plain caption immediately above a
  chart when geometry and text-preservation checks pass. Notes below remain
  standalone. Default behavior retains source caption order.

`normalize_text=False`, `deduplicate=False`, and `render_charts=False` disable the
corresponding cleanup/rendering choices. The existing root imports continue to
work:

```python
from postprocessing import extract_classes_bboxes, postprocess_text
from latex2html import convert_html_table_to_json_hierarchical
```

`postprocess_text(..., cls='Table', table_format=...)` supports LaTeX, HTML,
Markdown, record JSON, hierarchical JSON, and CSV. HTML retains merged-cell
structure. Flat exports expand spans and use neutral column names for headerless
tables without consuming the first data row as a header. Duplicate column names
are disambiguated without overwriting values. Hierarchical JSON retains the
existing `headers`, `columns`, and flat `data` fields and additionally exposes
`body_cells` with explicit text, span, and header-role metadata.

## Parse a page with a running vLLM server

Start the server using the model card. Then:

```python
from nemotron_parse.inference import DocumentParser

with DocumentParser(base_url='http://localhost:8000/v1', model='nemotron-parse') as parser:
    single = parser.parse('page.png')
    double = parser.parse_double('page.png')
    triple = parser.parse_triple('page.png', triple_regions='union')
    documents = parser.parse_many(['one.png', 'two.png'], mode='double', concurrency=4)
```

| Mode | Work performed |
| --- | --- |
| Single | Full-page parse and document rendering |
| Double | Full-page parse, then Table/Chart crops from its predicted regions |
| Triple | Full-page parse, separate detection, then Table/Chart crop parsing |

A pass denotes a stage, not one request: each retained crop needs its own model
call. `triple_regions='union'` retains detector regions first and adds parser
regions after overlap filtering; `'detector'` uses only the separate detector.
The region budget defaults to 80 per page and can be changed with `max_regions`.
Crops use 4% plus 12-pixel margins and JPEG quality 95, configurable margins, and
content-normalized coordinates mapped back to the original image. Tiny crops are
skipped. Crops are made in memory and no temporary image files remain.

Crop merging preserves full-page block order and uses guarded structural
replacement. Tables require compatible predicted column counts when both grids
are valid, overlap and text similarity; malformed native grids are rejected.
Charts require overlapping, similar predictions with a complete chart/table
serialization. Rejected overlapping crops are not inserted as duplicate objects.
Only nonoverlapping Table/Chart candidates are added. These conservative guards
can retain a weaker full-page prediction instead of accepting a substantially
different crop. Raw crop responses remain available for inspection.

Returned results include `stages` with original raw responses and usage, and
`inference` with request counts, regions, sampling, and crop/rendering settings.
Empty, failed, filtered, or token-limited generations raise `GenerationError`;
its `stage` and `response` expose the failed response. Successful stage responses
from the same page are only returned when parsing completes. No automatic retry
policy changes the request settings.

Sampling is configurable per stage:

```python
from nemotron_parse.inference import DocumentParser, GenerationSettings

with DocumentParser(
    fullpage=GenerationSettings(temperature=0, top_k=1, repetition_penalty=1.1),
    detection=GenerationSettings(temperature=0, top_k=1, max_tokens=4096),
    crop=GenerationSettings(temperature=0, top_k=1, repetition_penalty=1.0),
) as parser:
    document = parser.parse_double('page.png')
```

Defaults are temperature 0.2 / repetition penalty 1.05 for full-page and detection,
and 0.1 / 1.0 for crops, top-k 50, top-p 1.0, seed 0. Token caps are 8,744 for
parsing and 4,096 for detection. These are practical starting settings rather than
a claim of optimal sampling for every document type.

## Use an already loaded Transformers model

Continue from the model card's Transformers setup:

```python
from nemotron_parse.inference import DocumentParser, TransformersBackend

backend = TransformersBackend(model, processor, generation_config=generation_config)
with DocumentParser(backend=backend) as parser:
    document = parser.parse_triple(image)
```

This uses the same image/crop orchestration without a server or plugin. The adapter
serializes calls to the shared model, uses inference mode, applies stage settings,
and restores the CPU/current-GPU RNG after generation. Model weights and processor
remain caller-owned. Reaching the token cap is conservatively reported as
truncation, including when generation forces an EOS at the cap.

Other inference implementations can supply a callable
`backend(image, *, prompt, stage, settings)` returning
`{'raw_out': text, 'finish_reason': 'stop', 'usage': {...}}`. A string return is also
accepted, but then the custom backend must detect truncation itself. Custom
backends used with `parse_many` should support concurrent calls or use concurrency 1.

## Command line

```bash
nemotron-parse --image page.png --mode double --output parsed-page
nemotron-parse --image-dir pages --mode triple --triple-regions union --output parsed-pages
```

The download-only equivalent is `python -m nemotron_parse.inference` with the same
arguments. Each input gets a Markdown file and a JSON result containing blocks,
raw stages, and settings. Use a new/empty output directory. Files are written only
after all pages complete. The API accepts PIL images and image paths; the CLI
accepts image files. Render PDF pages to images before calling it.

## Validation and scope

Run the focused CPU regression suite from the downloaded repository:

```bash
python -m pip install pytest
python -m pytest -q tests/test_postprocessing.py tests/test_inference.py
```

The suite covers table structure, inline commands, headerless exports, column-name
collisions, text preservation, chart controls, crop geometry, merge guards, HTTP
request/response handling, failure/truncation reporting, and batch ordering.
The Transformers adapter is checked with a small CPU model stub; endpoint tests
use deterministic responses. These checks do not establish model accuracy or GPU
throughput. No new full ParseBench scoring or real-model inference is claimed for
this helper release. For the headline score, use the separately pinned
[reproduction instructions](reference/parsebench/pipeline/README.md).
