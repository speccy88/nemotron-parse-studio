import XCTest
import Foundation
import CoreGraphics
import ImageIO
import PDFKit
@testable import ParseCore

final class DocumentIOTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func temporary() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("NemotronDocumentTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: url) }
        return url
    }

    func testPDFImportIsCopiedAndRendersOneRGBPageAtRequestedDPI() async throws {
        let loader = DocumentLoader(storageURL: try temporary())
        let source = root.appendingPathComponent("Samples/multipage.pdf")
        let doc = try await loader.load(url: source)
        XCTAssertEqual(doc.pageCount, 3)
        XCTAssertNotEqual(doc.originalURL, source)
        XCTAssertEqual(try Data(contentsOf: source), try Data(contentsOf: doc.originalURL))
        let page = try await loader.render(document: doc, page: 1, dpi: 144)
        XCTAssertEqual(page.width, 1224); XCTAssertEqual(page.height, 1584)
        XCTAssertEqual(page.documentWidth, 612); XCTAssertEqual(page.documentHeight, 792)
        let imageSource = try XCTUnwrap(CGImageSourceCreateWithURL(page.imageURL as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(imageSource, 0, nil))
        XCTAssertEqual(image.colorSpace?.model, .rgb)
        XCTAssertTrue(FileManager.default.fileExists(atPath: page.thumbnailURL.path))
    }

    func testScannedPDFImageFormatsAndMultiFrameTIFF() async throws {
        let loader = DocumentLoader(storageURL: try temporary())
        for (name, count) in [("scanned.pdf", 2), ("mixed-layout.png", 1), ("mixed-layout.jpg", 1), ("multipage.tiff", 2), ("empty.pdf", 1)] {
            let doc = try await loader.load(url: root.appendingPathComponent("Samples/\(name)"))
            XCTAssertEqual(doc.pageCount, count, name)
            let page = try await loader.render(document: doc, page: count - 1, dpi: 72)
            XCTAssertEqual(page.width, 612, name); XCTAssertEqual(page.height, 792, name)
        }
    }

    func testRotatedPDFUsesDisplayedCoordinates() async throws {
        let temp = try temporary()
        let source = try XCTUnwrap(PDFDocument(url: root.appendingPathComponent("Samples/single-page.pdf")))
        source.page(at: 0)?.rotation = 90
        let url = temp.appendingPathComponent("rotated.pdf")
        XCTAssertTrue(source.write(to: url))
        let loader = DocumentLoader(storageURL: temp.appendingPathComponent("imports"))
        let doc = try await loader.load(url: url)
        let rendered = try await loader.render(document: doc, page: 0, dpi: 72)
        XCTAssertEqual(rendered.rotation, 90); XCTAssertEqual(rendered.width, 792); XCTAssertEqual(rendered.height, 612)
        XCTAssertEqual(rendered.documentWidth, 792); XCTAssertEqual(rendered.documentHeight, 612)
    }

    func testInvalidPageDPIAndCancellationAreRecoverable() async throws {
        let loader = DocumentLoader(storageURL: try temporary())
        let doc = try await loader.load(url: root.appendingPathComponent("Samples/single-page.pdf"))
        do { _ = try await loader.render(document: doc, page: 1, dpi: 72); XCTFail("Out-of-range page accepted") } catch DocumentIOError.invalidPage { }
        do { _ = try await loader.render(document: doc, page: 0, dpi: .nan); XCTFail("Nonfinite DPI accepted") } catch { }
        let cancelled = Task {
            withUnsafeCurrentTask { $0?.cancel() }
            return try await loader.render(document: doc, page: 0, dpi: 72)
        }
        do { _ = try await cancelled.value; XCTFail("Cancelled render completed") } catch is CancellationError { }
        let retried = try await loader.render(document: doc, page: 0, dpi: 72)
        XCTAssertEqual(retried.width, 612)
    }

    func testUnsafeHTMLRejectedBeforeConversion() async throws {
        let temp = try temporary(), file = try temporary().appendingPathComponent("unsafe.html")
        try "<html><script>alert('never execute')</script><img src=\"https://example.invalid/private\"></html>".write(to: file, atomically: true, encoding: .utf8)
        let loader = DocumentLoader(storageURL: temp)
        do { _ = try await loader.load(url: file); XCTFail("Active HTML accepted") } catch DocumentIOError.unsafeHTML { }
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(at: temp, includingPropertiesForKeys: nil).count, 0)
    }

    func testLocalOfficeRTFAndPassiveHTMLConversion() async throws {
        guard DocumentLoader.officeConverterExecutable() != nil else {
            throw XCTSkip("LibreOffice is not installed")
        }
        let loader = DocumentLoader(storageURL: try temporary())
        for name in ["simple.docx", "simple.pptx", "simple.xlsx", "simple.rtf", "simple.html"] {
            let document = try await loader.load(url: root.appendingPathComponent("Samples/\(name)"))
            XCTAssertGreaterThan(document.pageCount, 0)
            XCTAssertEqual(document.workingURL.pathExtension, "pdf")
            let convertedText = PDFDocument(url: document.workingURL)?.string ?? ""
            XCTAssertTrue(convertedText.contains(name == "simple.xlsx" ? "Product" : "Local"), "Converted PDF lost expected text: \(name)")
            let result = try await loader.render(document: document, page: 0, dpi: 72)
            XCTAssertGreaterThan(result.width, 0)
        }
    }

    func testExportsPreserveRawMergedCellsAndEscapes() async throws {
        let temp = try temporary(), loader = DocumentLoader(storageURL: try temporary())
        let doc = try await loader.load(url: root.appendingPathComponent("Samples/empty.pdf"))
        let rendered = try await loader.render(document: doc, page: 0, dpi: 72)
        let table = ExportTable(cells: [ExportCell(row: 0, column: 0, columnSpan: 2, text: "Header", isHeader: true), ExportCell(row: 1, column: 0, text: "1,250.00"), ExportCell(row: 1, column: 1, text: "")], rowCount: 2, columnCount: 2, hierarchicalJSON: "{\"source\":\"unchanged\"}")
        let raw = "IMMUTABLE <script>unsafe()</script> raw prediction"
        let blocks = [ExportBlock(id: "t", text: "raw table", semanticClass: "Table", bbox: [0.25, 0.25, 0.75, 0.75], readingOrder: 0, table: table), ExportBlock(id: "x", text: "<script>alert(1)</script>", semanticClass: "UnknownClass", bbox: [0, 0, 1, 1], readingOrder: 1), ExportBlock(id: "excluded", text: "Excluded", semanticClass: "Text", bbox: [0, 0, 1, 1], readingOrder: 2, included: false)]
        let page = ExportPage(documentName: doc.name, pageNumber: 1, rendered: rendered, blocks: blocks, raw: raw, metadata: Data("{\"corrections\":{\"t\":\"manual\"}}".utf8), originalURL: doc.originalURL, workingURL: doc.workingURL)
        let exporter = DocumentExporter()
        let markdown = temp.appendingPathComponent("page.md")
        try await exporter.write(pages: [page], format: .markdown, to: markdown)
        let md = try String(contentsOf: markdown, encoding: .utf8)
        XCTAssertTrue(md.contains("colspan=\"2\"")); XCTAssertFalse(md.contains("Excluded"))
        let html = temp.appendingPathComponent("page.html")
        try await exporter.write(pages: [page], format: .html, to: html)
        let htmlText = try String(contentsOf: html, encoding: .utf8)
        XCTAssertFalse(htmlText.contains("<script>")); XCTAssertTrue(htmlText.contains("&lt;script&gt;"))
        let json = temp.appendingPathComponent("page.json")
        try await exporter.write(pages: [page], format: .structuredJSON, to: json)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: json)) as? [String: Any])
        let first = try XCTUnwrap((object["pages"] as? [[String: Any]])?.first)
        XCTAssertEqual(first["raw"] as? String, raw)
        XCTAssertEqual((first["blocks"] as? [[String: Any]])?.count, 3)
        XCTAssertNotNil(first["extractionAndCorrections"])
        let csvDir = temp.appendingPathComponent("csv")
        try await exporter.write(pages: [page], format: .tableCSV, to: csvDir)
        let csvURL = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: csvDir, includingPropertiesForKeys: nil).first { $0.pathExtension == "csv" })
        let csv = try String(contentsOf: csvURL, encoding: .utf8)
        XCTAssertEqual(csv, "\"Header\",\"\"\r\n\"1,250.00\",\"\"\r\n")
        XCTAssertTrue(FileManager.default.fileExists(atPath: csvDir.appendingPathComponent("manifest.json").path))
        let pdf = temp.appendingPathComponent("annotated.pdf")
        try await exporter.write(pages: [page, page], format: .annotatedPDF, to: pdf)
        XCTAssertEqual(CGPDFDocument(pdf as CFURL)?.numberOfPages, 2)
        let pdfImported = try await loader.load(url: pdf)
        let pdfRendered = try await loader.render(document: pdfImported, page: 0, dpi: 72)
        XCTAssertEqual(pdfRendered.width, 612); XCTAssertEqual(pdfRendered.height, 792)
        do { try await exporter.write(pages: [page], format: .searchablePDF, to: temp.appendingPathComponent("searchable.pdf")); XCTFail("Unaligned search layer generated") } catch ExportError.searchableUnavailable { }
    }

    func testInlineFormattingPreservedAndActiveMarkupEscaped() {
        let html = DocumentExporter.sanitizeInlineHTML("<b>Bold &amp; safe</b><br>H<sub>2</sub>O <sup>2</sup><img src=x onerror=alert(1)><script>alert(1)</script><em onclick='evil()'>text</em>")
        XCTAssertTrue(html.contains("<b>Bold &amp; safe</b><br>H<sub>2</sub>O <sup>2</sup>"))
        XCTAssertFalse(html.contains("<img")); XCTAssertFalse(html.contains("<script")); XCTAssertFalse(html.contains("<em onclick"))
        let styled = ExportTable(cells: [ExportCell(row: 0, column: 0, text: "Bold", html: "<b>Bold</b>")], rowCount: 1, columnCount: 1, caption: "Share <60%>")
        XCTAssertTrue(DocumentExporter.html(styled).contains("<b>Bold</b>"))
        XCTAssertTrue(DocumentExporter.html(styled).contains("<caption>Share &lt;60%&gt;</caption>"))
        let corrected = ExportTable(cells: [ExportCell(row: 0, column: 0, text: "<b>Manual</b>", html: nil)], rowCount: 1, columnCount: 1)
        XCTAssertTrue(DocumentExporter.html(corrected).contains("&lt;b&gt;Manual&lt;/b&gt;"))
    }

    func testAnnotatedPDFUsesTopLeftNormalizedBoundingBoxes() async throws {
        let temp = try temporary(), loader = DocumentLoader(storageURL: try temporary())
        let document = try await loader.load(url: root.appendingPathComponent("Samples/empty.pdf"))
        let rendered = try await loader.render(document: document, page: 0, dpi: 72)
        let block = ExportBlock(id: "region", text: "", semanticClass: "Table", bbox: [0.1, 0.15, 0.35, 0.45], readingOrder: 0)
        let page = ExportPage(documentName: document.name, pageNumber: 1, rendered: rendered, blocks: [block], raw: "", workingURL: document.workingURL)
        let url = temp.appendingPathComponent("coordinates.pdf")
        try await DocumentExporter().write(pages: [page], format: .annotatedPDF, to: url)
        let imported = try await loader.load(url: url)
        let imagePage = try await loader.render(document: imported, page: 0, dpi: 72)
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(imagePage.imageURL as CFURL, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        let context = try XCTUnwrap(CGContext(data: nil, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        let bytes = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        func greenNear(x: Int, y: Int) -> Bool {
            for row in (y - 2)...(y + 2) {
                for column in (x - 2)...(x + 2) {
                    let index = row * context.bytesPerRow + column * 4
                    if Int(bytes[index + 1]) > Int(bytes[index]) + 40 && Int(bytes[index + 1]) > Int(bytes[index + 2]) + 25 { return true }
                }
            }
            return false
        }
        XCTAssertTrue(greenNear(x: 61, y: 237), "Expected left border at normalized x=0.1, y=0.3")
        XCTAssertFalse(greenNear(x: 61, y: 554), "Bounding box was incorrectly flipped vertically")
    }
}
