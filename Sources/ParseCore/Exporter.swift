import Foundation
import CoreGraphics
import CoreText
import ImageIO

public enum ExportFormat: String, CaseIterable, Identifiable, Sendable {
    case markdown = "Markdown", html = "HTML", plainText = "Plain text", structuredJSON = "Structured JSON"
    case tableCSV = "Table CSV files", hierarchicalTablesJSON = "Hierarchical table JSON", annotatedPDF = "Annotated PDF", searchablePDF = "Searchable PDF"
    public var id: String { rawValue }
    public var fileExtension: String {
        switch self { case .markdown: "md"; case .html: "html"; case .plainText: "txt"; case .structuredJSON, .hierarchicalTablesJSON: "json"; case .tableCSV: ""; case .annotatedPDF, .searchablePDF: "pdf" }
    }
}

public struct ExportCell: Sendable {
    public var row: Int, column: Int, rowSpan: Int, columnSpan: Int
    public var text: String
    public var isHeader: Bool
    /// Untrusted inline source markup. Set to nil when the cell's plain text is manually edited.
    public var html: String?
    public init(row: Int, column: Int, rowSpan: Int = 1, columnSpan: Int = 1, text: String, isHeader: Bool = false, html: String? = nil) {
        self.row = row; self.column = column; self.rowSpan = rowSpan; self.columnSpan = columnSpan; self.text = text; self.isHeader = isHeader; self.html = html
    }
}

public struct ExportTable: Sendable {
    public var cells: [ExportCell]
    public var rowCount: Int, columnCount: Int
    /// Official hierarchical JSON retained alongside effective manually edited cells.
    public var hierarchicalJSON: String
    public var caption: String?
    public init(cells: [ExportCell], rowCount: Int, columnCount: Int, hierarchicalJSON: String = "{}", caption: String? = nil) {
        self.cells = cells; self.rowCount = rowCount; self.columnCount = columnCount; self.hierarchicalJSON = hierarchicalJSON; self.caption = caption
    }
}

public struct ExportBlock: Sendable {
    public var id: String, text: String, semanticClass: String
    public var bbox: [Double]
    public var readingOrder: Int
    public var included: Bool
    public var table: ExportTable?
    public init(id: String, text: String, semanticClass: String, bbox: [Double], readingOrder: Int, included: Bool = true, table: ExportTable? = nil) {
        self.id = id; self.text = text; self.semanticClass = semanticClass; self.bbox = bbox; self.readingOrder = readingOrder; self.included = included; self.table = table
    }
}

public struct ExportPage: Sendable {
    public var documentName: String
    /// One-based original page number.
    public var pageNumber: Int
    public var rendered: RenderedPage
    public var blocks: [ExportBlock]
    public var raw: String
    /// Complete extraction response and manual-correction history, encoded by the caller.
    public var metadata: Data?
    public var originalURL: URL?
    public var workingURL: URL?
    public init(documentName: String, pageNumber: Int, rendered: RenderedPage, blocks: [ExportBlock], raw: String, metadata: Data? = nil, originalURL: URL? = nil, workingURL: URL? = nil) {
        self.documentName = documentName; self.pageNumber = pageNumber; self.rendered = rendered; self.blocks = blocks; self.raw = raw; self.metadata = metadata; self.originalURL = originalURL; self.workingURL = workingURL
    }
}

public enum ExportError: LocalizedError {
    case noPages, invalidDestination, searchableUnavailable, unreadableImage
    public var errorDescription: String? {
        switch self {
        case .noPages: "Select at least one parsed page to export."
        case .invalidDestination: "The export destination could not be created."
        case .searchableUnavailable: "Accurate searchable PDF export requires line or glyph positions. Nemotron's block bounding boxes alone cannot guarantee alignment. Use annotated PDF and structured JSON."
        case .unreadableImage: "An original page image is no longer available. Render the page again before exporting."
        }
    }
}

public actor DocumentExporter {
    public init() {}
    /// Writes one page, a whole document, or multiple documents according to the supplied page list.
    public func write(pages: [ExportPage], format: ExportFormat, to destination: URL) async throws {
        guard !pages.isEmpty else { throw ExportError.noPages }
        try Task.checkCancellation()
        let scoped = destination.startAccessingSecurityScopedResource()
        defer { if scoped { destination.stopAccessingSecurityScopedResource() } }
        switch format {
        case .markdown:
            try pages.map { page in "# \(page.documentName) — Page \(page.pageNumber)\n\n" + Self.ordered(page).map(Self.markdown).joined(separator: "\n\n") }.joined(separator: "\n\n---\n\n").write(to: destination, atomically: true, encoding: .utf8)
        case .plainText:
            try pages.map { page in "\(page.documentName) — Page \(page.pageNumber)\n\n" + Self.ordered(page).map { block in block.table.map { table in (table.caption.map { $0 + "\n" } ?? "") + Self.csv(table) } ?? block.text }.joined(separator: "\n\n") }.joined(separator: "\n\n\u{000C}\n\n").write(to: destination, atomically: true, encoding: .utf8)
        case .html:
            let content = pages.map { page in "<section><h1>\(Self.escape(page.documentName)) — Page \(page.pageNumber)</h1>" + Self.ordered(page).map { block in block.table.map(Self.html) ?? "<div class=\"block\" data-class=\"\(Self.escape(block.semanticClass))\">\(Self.escape(block.text).replacingOccurrences(of: "\n", with: "<br>"))</div>" }.joined(separator: "\n") + "</section>" }.joined(separator: "\n")
            let document = "<!doctype html><html><head><meta charset=\"utf-8\"><meta http-equiv=\"Content-Security-Policy\" content=\"default-src 'none'; style-src 'unsafe-inline'\"><title>Nemotron Parse export</title><style>body{font-family:system-ui;margin:2rem;line-height:1.5}table{border-collapse:collapse}td,th{border:1px solid #999;padding:.4rem;white-space:pre-wrap}section{break-after:page}.block{margin:1em 0}</style></head><body>\(content)</body></html>"
            try document.write(to: destination, atomically: true, encoding: .utf8)
        case .structuredJSON:
            try Self.jsonData(["schemaVersion": 1, "coordinateSystem": "top-left normalized xyxy; rendered pixels and displayed PDF points", "pages": try pages.map(Self.pageJSON)]).write(to: destination, options: .atomic)
        case .tableCSV:
            try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
            var manifest: [[String: Any]] = []
            for (pageIndex, page) in pages.enumerated() {
                for (index, block) in Self.ordered(page).enumerated() {
                    guard let table = block.table else { continue }
                    try Task.checkCancellation()
                    let name = "\(pageIndex + 1)-\(Self.safeName(page.documentName))-page-\(page.pageNumber)-table-\(index + 1).csv"
                    try Self.csv(table).write(to: destination.appendingPathComponent(name), atomically: true, encoding: .utf8)
                    manifest.append(["file": name, "document": page.documentName, "page": page.pageNumber, "blockID": block.id, "note": "CSV has empty placeholders for covered merged cells. Span and header metadata are preserved in manifest.", "table": Self.tableJSON(table)])
                }
            }
            try Self.jsonData(["tables": manifest]).write(to: destination.appendingPathComponent("manifest.json"), options: .atomic)
        case .hierarchicalTablesJSON:
            var tables: [[String: Any]] = []
            for page in pages {
                for block in Self.ordered(page) {
                    if let table = block.table { tables.append(["document": page.documentName, "page": page.pageNumber, "blockID": block.id, "table": Self.tableJSON(table)]) }
                }
            }
            try Self.jsonData(["tables": tables]).write(to: destination, options: .atomic)
        case .annotatedPDF:
            try Self.pdf(pages: pages, to: destination)
        case .searchablePDF:
            throw ExportError.searchableUnavailable
        }
    }

    public static func csv(_ table: ExportTable) -> String {
        let rows = max(0, min(table.rowCount, 100_000)), columns = max(0, min(table.columnCount, 10_000))
        guard rows > 0, columns > 0, rows * columns <= 2_000_000 else { return "" }
        var grid = Array(repeating: Array(repeating: "", count: columns), count: rows)
        for cell in table.cells where cell.row >= 0 && cell.row < rows && cell.column >= 0 && cell.column < columns { grid[cell.row][cell.column] = cell.text }
        return grid.map { $0.map { "\"" + $0.replacingOccurrences(of: "\"", with: "\"\"") + "\"" }.joined(separator: ",") }.joined(separator: "\r\n") + "\r\n"
    }

    public static func html(_ table: ExportTable) -> String {
        var body = "<table>"
        if let caption = table.caption, !caption.isEmpty { body += "<caption>\(escape(caption))</caption>" }
        let rows = max(0, min(table.rowCount, 100_000)), columns = max(0, min(table.columnCount, 10_000))
        guard rows * columns <= 2_000_000 else { return "<p>Table exceeds export grid limits; original data retained in structured JSON.</p>" }
        var occupied = Set<String>()
        var starts: [String: ExportCell] = [:]
        for cell in table.cells where cell.row >= 0 && cell.column >= 0 && cell.row < rows && cell.column < columns {
            starts["\(cell.row),\(cell.column)"] = cell
            for row in cell.row..<min(rows, cell.row + max(1, cell.rowSpan)) {
                for column in cell.column..<min(columns, cell.column + max(1, cell.columnSpan)) where row != cell.row || column != cell.column { occupied.insert("\(row),\(column)") }
            }
        }
        for row in 0..<rows {
            body += "<tr>"
            for column in 0..<columns {
                let key = "\(row),\(column)"
                if let cell = starts[key] {
                    let tag = cell.isHeader ? "th" : "td"
                    let content = cell.html.map(sanitizeInlineHTML) ?? escape(cell.text).replacingOccurrences(of: "\n", with: "<br>")
                    body += "<\(tag) rowspan=\"\(max(1, cell.rowSpan))\" colspan=\"\(max(1, cell.columnSpan))\">\(content)</\(tag)>"
                } else if !occupied.contains(key) { body += "<td></td>" }
            }
            body += "</tr>"
        }
        return body + "</table>"
    }

    private static func markdown(_ block: ExportBlock) -> String {
        if let table = block.table { return html(table) }
        // Parsed Markdown is preserved as plain file content, never executed by the exporter.
        return block.text
    }
    private static func ordered(_ page: ExportPage) -> [ExportBlock] { page.blocks.filter(\.included).sorted { $0.readingOrder < $1.readingOrder } }
    private static func escape(_ value: String) -> String {
        value.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;").replacingOccurrences(of: "\"", with: "&quot;").replacingOccurrences(of: "'", with: "&#39;")
    }
    /// Preserves only attribute-free inline formatting and entity references. Every other tag is shown as text.
    /// This is intentionally conservative: extracted markup is never treated as a general HTML document.
    public static func sanitizeInlineHTML(_ source: String) -> String {
        let pattern = #"<[^>]*>|&(?:amp|lt|gt|quot|apos|nbsp|#\d+|#x[0-9a-fA-F]+);"#
        guard let tokenizer = try? NSRegularExpression(pattern: pattern) else { return escape(source) }
        let string = source as NSString
        let matches = tokenizer.matches(in: source, range: NSRange(location: 0, length: string.length))
        let tags: Set<String> = ["b", "strong", "i", "em", "u", "s", "del", "sup", "sub", "code", "kbd", "small", "mark"]
        var result = "", offset = 0
        for match in matches {
            result += escape(string.substring(with: NSRange(location: offset, length: match.range.location - offset)))
            let token = string.substring(with: match.range)
            if token.hasPrefix("&") { result += token }
            else {
                let normalized = token.lowercased()
                if normalized == "<br>" || normalized == "<br/>" || normalized == "<br />" { result += "<br>" }
                else {
                    let name = normalized.replacingOccurrences(of: #"^</?|>$"#, with: "", options: .regularExpression)
                    result += tags.contains(name) ? normalized : escape(token)
                }
            }
            offset = match.range.location + match.range.length
        }
        result += escape(string.substring(from: offset))
        return result.replacingOccurrences(of: "\n", with: "<br>")
    }
    private static func safeName(_ value: String) -> String {
        let cleaned = value.replacingOccurrences(of: #"[^\p{L}\p{N}._-]"#, with: "_", options: .regularExpression)
        return String(cleaned.prefix(100))
    }
    private static func jsonData(_ object: Any) throws -> Data { try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed]) }
    private static func tableJSON(_ table: ExportTable) -> [String: Any] {
        ["rowCount": table.rowCount, "columnCount": table.columnCount, "caption": table.caption as Any? ?? NSNull(),
         "cells": table.cells.map { cell -> [String: Any] in
             var result: [String: Any] = ["row": cell.row, "column": cell.column, "rowSpan": cell.rowSpan, "columnSpan": cell.columnSpan, "text": cell.text, "isHeader": cell.isHeader]
             if let html = cell.html { result["html"] = sanitizeInlineHTML(html) }
             return result
         },
         "originalHierarchicalJSON": (try? JSONSerialization.jsonObject(with: Data(table.hierarchicalJSON.utf8), options: .fragmentsAllowed)) ?? table.hierarchicalJSON,
         "html": html(table), "csv": csv(table)]
    }
    private static func pageJSON(_ page: ExportPage) throws -> [String: Any] {
        var object: [String: Any] = ["document": page.documentName, "page": page.pageNumber, "raw": page.raw,
          "width": page.rendered.width, "height": page.rendered.height, "documentWidth": page.rendered.documentWidth,
          "documentHeight": page.rendered.documentHeight, "rotation": page.rendered.rotation,
          "imageURL": page.rendered.imageURL.path,
          "blocks": page.blocks.map { block -> [String: Any] in
              var value: [String: Any] = ["id": block.id, "text": block.text, "semanticClass": block.semanticClass, "bbox": block.bbox, "readingOrder": block.readingOrder, "included": block.included]
              if block.bbox.count == 4 { value["pixelBbox"] = [block.bbox[0] * Double(page.rendered.width), block.bbox[1] * Double(page.rendered.height), block.bbox[2] * Double(page.rendered.width), block.bbox[3] * Double(page.rendered.height)] }
              if let table = block.table { value["table"] = tableJSON(table) }
              return value
          }]
        if let metadata = page.metadata { object["extractionAndCorrections"] = try JSONSerialization.jsonObject(with: metadata, options: .fragmentsAllowed) }
        if let original = page.originalURL { object["originalURL"] = original.path }
        return object
    }

    private static func pdf(pages: [ExportPage], to destination: URL) throws {
        let temporary = destination.deletingLastPathComponent().appendingPathComponent(".\(UUID().uuidString).pdf")
        defer { try? FileManager.default.removeItem(at: temporary) }
        guard let consumer = CGDataConsumer(url: temporary as CFURL), let context = CGContext(consumer: consumer, mediaBox: nil, nil) else { throw ExportError.invalidDestination }
        for page in pages {
            try Task.checkCancellation()
            let width = page.rendered.documentWidth, height = page.rendered.documentHeight
            var box = CGRect(x: 0, y: 0, width: width, height: height)
            let data = Data(bytes: &box, count: MemoryLayout<CGRect>.size)
            context.beginPDFPage([kCGPDFContextMediaBox as String: data] as CFDictionary)
            if let working = page.workingURL, working.pathExtension.lowercased() == "pdf", let pdf = CGPDFDocument(working as CFURL), let sourcePage = pdf.page(at: page.pageNumber) {
                context.saveGState()
                context.concatenate(sourcePage.getDrawingTransform(.cropBox, rect: box, rotate: 0, preserveAspectRatio: true))
                context.drawPDFPage(sourcePage)
                context.restoreGState()
            } else {
                guard let source = CGImageSourceCreateWithURL(page.rendered.imageURL as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { context.endPDFPage(); context.closePDF(); throw ExportError.unreadableImage }
                context.draw(image, in: box)
            }
            for block in ordered(page) where block.bbox.count == 4 && block.bbox.allSatisfy(\.isFinite) {
                let b = block.bbox.map { min(1, max(0, $0)) }
                let rect = CGRect(x: b[0] * width, y: (1 - b[3]) * height, width: max(0, b[2] - b[0]) * width, height: max(0, b[3] - b[1]) * height)
                let color = classColor(block.semanticClass)
                context.setStrokeColor(color); context.setLineWidth(1.2); context.stroke(rect)
                let font = CTFontCreateWithName("Helvetica" as CFString, 7, nil)
                let line = CTLineCreateWithAttributedString(NSAttributedString(string: "\(block.readingOrder + 1) · \(block.semanticClass)", attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font, NSAttributedString.Key(kCTForegroundColorAttributeName as String): color]))
                context.textPosition = CGPoint(x: rect.minX + 2, y: min(height - 9, rect.maxY + 2))
                CTLineDraw(line, context)
            }
            context.endPDFPage()
        }
        context.closePDF()
        if FileManager.default.fileExists(atPath: destination.path) { _ = try FileManager.default.replaceItemAt(destination, withItemAt: temporary) }
        else { try FileManager.default.moveItem(at: temporary, to: destination) }
    }
    private static func classColor(_ name: String) -> CGColor {
        switch name.lowercased() {
        case "table": CGColor(srgbRed: 0.12, green: 0.6, blue: 0.38, alpha: 1)
        case "chart": CGColor(srgbRed: 0.9, green: 0.45, blue: 0.12, alpha: 1)
        case "title", "section-header": CGColor(srgbRed: 0.55, green: 0.3, blue: 0.8, alpha: 1)
        default: CGColor(srgbRed: 0.1, green: 0.4, blue: 0.85, alpha: 1)
        }
    }
}
