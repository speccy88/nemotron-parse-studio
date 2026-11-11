# Synthetic verification documents

Regenerate these local fixtures from the project root:

```sh
swift Scripts/generate_samples.swift Samples
```

The PDF/image page contains a title, multilingual prose, a merged-cell table with intentional empty cells, a bar chart with percentages and original counts, a caption, code, and mathematical notation. All values are synthetic. These are input fixtures, not expected Nemotron predictions.

| File | Coverage |
| --- | --- |
| `single-page.pdf` | Vector text and graphics; 612 × 792 points |
| `multipage.pdf` | Three pages, tested with incremental page rendering |
| `scanned.pdf` | Two raster-only PDF pages |
| `empty.pdf` | One blank page |
| `mixed-layout.png`, `mixed-layout.jpg` | Native RGB image input |
| `multipage.tiff` | Two independently selectable TIFF frames |
| `simple.docx` | WordprocessingML text and table |
| `simple.pptx` | Single-slide PresentationML text |
| `simple.xlsx` | SpreadsheetML values and headers |
| `simple.rtf` | RTF text |
| `simple.html` | Restricted passive local HTML with a merged table heading |

The native integration tests generate a rotated PDF from the single-page fixture and test displayed dimensions after rotation. Office conversion uses a dedicated LibreOffice profile, disables macros, and denies network access on macOS through the system sandbox. HTML with scripts, styles, links, images, or embedded resources is deliberately rejected; use a trusted PDF export for richer HTML.

DOCX, PPTX, XLSX, RTF and passive HTML were successfully converted to readable PDF pages by the installed local LibreOffice runtime. Fonts and pagination can differ from Microsoft Office, particularly for complex presentations and spreadsheets. Keep the unchanged copied original when reviewing conversion fidelity.
