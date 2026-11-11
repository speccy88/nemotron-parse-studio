# Import, coordinates, and export

## Imports

The loader copies the unchanged original into `~/Library/Application Support/Nemotron Parse Studio/Imports/<UUID>/`. Security-scoped file access is held only while making this copy. Working files and rendered pages remain within that import directory, so a resumed workspace does not depend on the source file still being selected or present. The application does not delete the user's source.

PDFs are read one page at a time using Core Graphics. Rendered pages are white-backed sRGB PNGs, with a separate small thumbnail. Scanned PDFs work through the same rendering path. PDF crop boxes and page rotations are applied by `CGPDFPage.getDrawingTransform`; dimensions describe the displayed rotated page. Password-protected PDFs need an unlocked copy. The PDF DPI control accepts 36–600 DPI; a per-page 64-million-pixel guard prevents unusually large allocations. Large PDFs are never held as a complete array of uncompressed page images.

ImageIO determines supported image types, including PNG, JPEG, TIFF and HEIC when the OS decoder supports them. TIFF frames become individual pages. EXIF orientation is applied before inference, and transparency is composited against white. Input images retain their native pixel dimensions; DPI adjusts PDF and converted-document rendering, not source-image resolution. HEIC decoding is implemented through ImageIO but has not been exercised with a HEIC fixture.

DOCX, PPTX, XLSX, RTF, HTML, ODT, ODP, and ODS use local LibreOffice. The executable is found through `NEMOTRON_SOFFICE`, the standard Applications installation, common Homebrew paths, or the installed development runtime. A separate disposable profile disables macros. On macOS the converter runs through `sandbox-exec` with network access denied. Conversion is cancelled with the request and stopped after 120 seconds. The app accepts only a passive HTML subset: scripts, styles, embedded objects, links, images, resource references, and event handlers are rejected. This avoids executing active content or fetching referenced resources. Unsupported or unsuccessful conversions fail visibly.

Basic DOCX, PPTX, XLSX, RTF and passive HTML conversions have been tested with actual local LibreOffice output, including verifying expected PDF text. ODF types are enabled by the same converter but have not been tested. Complex Office layout fidelity depends on LibreOffice and available fonts. XLSX print areas, page settings and sheet layout can change pagination; review the preview before parsing. No claim of pixel-identical Microsoft Office rendering is made.

## Coordinates

The parser receives the RGB image's `(width, height)`. Bounding boxes use `[x1, y1, x2, y2]` normalized to the displayed page, with top-left origin. Multiplying x values by rendered width and y values by rendered height gives pixels. The exported PDF uses the same displayed page dimensions in points; its bottom-left Core Graphics origin requires `y = (1 − y2) × pageHeight`. The source PDF drawing transform already accounts for rotation, so rotation is not applied a second time to predicted boxes.

## Exports

- Markdown retains HTML tables so row and column spans remain expressible.
- HTML escapes block text and allows only attribute-free inline formatting inside table cells. A restrictive Content Security Policy disables script and network content. No imported HTML is executed by the exporter.
- CSV quotes every string and preserves original numerical representations and line breaks. Empty cells and cells covered by a span get explicit empty fields. A companion `manifest.json` records cell positions, spans, header roles, and original hierarchy. CSV itself cannot represent merged cells.
- Structured JSON includes immutable raw model output, all effective blocks including excluded blocks, normalized and pixel coordinates, rendered dimensions, source paths, and caller-supplied complete extraction/correction metadata. JSON uses UTF-8. It references the preserved local originals; it does not embed their binary bytes. Copy the original documents separately when moving an export to another computer.
- Hierarchical-table JSON includes the original NVIDIA hierarchy and the effective cell array, preserving manual corrections without pretending the original hierarchy was recomputed. Original hierarchy and effective cells are deliberately separate keys.
- Table and chart captions are retained in HTML/Markdown captions, plain text, and table JSON. A compound predicted block containing multiple tables currently exposes only its first table in the native editable grid and per-table export. The complete raw block and postprocessed source remain in the structured extraction metadata; review its warning and raw output when this occurs.
- Annotated PDF redraws the original vector PDF page where available, retaining source text/graphics; image documents use the rendered page image. Boxes and reading-order/class labels are overlaid in the displayed coordinate system. Existing source PDF text remains selectable. The exporter does not add a synthetic search layer.
- Newly searchable PDFs are explicitly unsupported because block-level boxes cannot establish the per-line or per-glyph positions required for accurate text alignment. Annotated PDFs and structured JSON remain available.

Plain text, Markdown, HTML, annotated PDF and structured JSON accept a list of pages covering one page, a document, or a batch. Table CSV exports a directory with unique per-page table filenames. Raw stage responses and source table markup remain in extraction metadata. Manual table edits clear the previous inline markup for that cell, ensuring the edited plain text wins while the original prediction remains recoverable.

## Verification

`swift test --filter DocumentIOTests` covers unchanged source copies; PDF page counts and DPI dimensions; scanned PDFs; PNG/JPEG; TIFF frames; page rotation; cancelled rendering and retry; all five Office/HTML conversion fixtures; rejected active HTML; merged/empty cells; raw-prediction and metadata preservation; HTML sanitization and captions; CSV quoting; annotated PDF page counts/dimensions; and raster pixel verification that normalized boxes are not vertically flipped. The sample generator is `Scripts/generate_samples.swift`.
