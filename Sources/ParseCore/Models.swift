import Foundation

public let pinnedRevision = "a9a1febb1b9bf497fc67bd1d23fa33e170b249db"

public struct PostprocessOptions: Codable, Equatable, Sendable {
    public var infer_headers = true
    public var render_charts = true
    public var split_chart_value_percent = false
    public var relocate_chart_captions = false
    public var normalize_text = true
    public var deduplicate = true
    public init() {}
}

public struct InferenceSettings: Codable, Equatable, Sendable {
    public var backend = "transformers"
    public var server_url = "http://127.0.0.1:8000/v1"
    public var model = "nvidia/Nemotron-3.5-Parse"
    public var revision = pinnedRevision
    public var model_directory = ""
    public var native_model_directory: String?
    #if arch(arm64)
    public var device = "mps"
    #else
    public var device = "auto"
    #endif
    public var mode = "single"
    public var max_new_tokens = 8744
    public var temperature = 0.2
    public var top_p = 1.0
    public var top_k = 50
    public var repetition_penalty = 1.05
    public var seed = 0
    public var max_crops = 80
    public var crop_margin = 0.04
    public var crop_margin_pixels = 12
    public var triple_regions = "union"
    public var predict_text_in_pic = true
    public init() {}
}

public struct TableCell: Codable, Identifiable, Equatable, Sendable {
    public var row: Int
    public var column: Int
    public var rowSpan: Int
    public var columnSpan: Int
    public var text: String
    public var isHeader: Bool
    public var html: String?
    public var id: String { "\(row):\(column)" }
    public init(row: Int, column: Int, rowSpan: Int = 1, columnSpan: Int = 1, text: String, isHeader: Bool = false) {
        self.row = row; self.column = column; self.rowSpan = rowSpan; self.columnSpan = columnSpan; self.text = text; self.isHeader = isHeader
    }
}
public struct ParsedTable: Codable, Equatable, Sendable {
    public var cells: [TableCell]
    public var rowCount: Int
    public var columnCount: Int
    public var html: String
    public var markdown: String
    public var latex: String
    public var csv: String
    public var recordsJSON: String
    public var hierarchicalJSON: String
    public var caption: String?
    public var contextText: String?
}
public struct ParsedBlock: Codable, Identifiable, Equatable, Sendable {
    public var id: String
    public var rawText: String
    public var text: String
    public var semanticClass: String
    public var bbox: [Double]
    public var readingOrder: Int
    public var table: ParsedTable?
    public var provenance: String
    public var warnings: [String]
    public var bounds: NormalizedBox? { NormalizedBox(bbox) }
}
public struct StageResponse: Codable, Identifiable, Sendable {
    public var name: String
    public var raw: String
    public var finishReason: String?
    public var cropBBoxPixels: [Double]?
    public var id: String { name + (cropBBoxPixels?.description ?? "") + String(raw.prefix(32)) }
}
public struct ParseResult: Codable, Sendable {
    public var raw: String
    public var mergedRaw: String?
    public var markdown: String
    public var blocks: [ParsedBlock]
    public var warnings: [String]
    public var stages: [StageResponse]
    public var metrics: [String: Double]
    public var sourceResponse: JSONValue?
    public var postprocessingResponse: JSONValue?
    public var backend: String
    public var revision: String
    public init(raw: String, mergedRaw: String? = nil, markdown: String, blocks: [ParsedBlock], warnings: [String], stages: [StageResponse], metrics: [String: Double], sourceResponse: JSONValue? = nil, postprocessingResponse: JSONValue? = nil, backend: String, revision: String) {
        self.raw = raw; self.mergedRaw = mergedRaw; self.markdown = markdown; self.blocks = blocks; self.warnings = warnings; self.stages = stages; self.metrics = metrics; self.sourceResponse = sourceResponse; self.postprocessingResponse = postprocessingResponse; self.backend = backend; self.revision = revision
    }
}
public struct NormalizedBox: Codable, Equatable, Sendable {
    public var x: Double; public var y: Double; public var width: Double; public var height: Double
    public init?(_ values: [Double]) {
        guard values.count == 4, values.allSatisfy(\.isFinite) else { return nil }
        let x1 = min(1, max(0, min(values[0], values[2])))
        let y1 = min(1, max(0, min(values[1], values[3])))
        let x2 = min(1, max(0, max(values[0], values[2])))
        let y2 = min(1, max(0, max(values[1], values[3])))
        x = x1; y = y1; width = x2 - x1; height = y2 - y1
    }
    public func rect(width w: Double, height h: Double) -> CGRect { CGRect(x: x*w, y: y*h, width: width*w, height: height*h) }
}

/// Corrections are an overlay. Neither raw responses nor postprocessed blocks are mutated.
public struct BlockCorrection: Codable, Equatable, Sendable {
    public var text: String?
    public var semanticClass: String?
    public var readingOrder: Int?
    public var excluded = false
    public var tableCells: [String: String] = [:]
    public var selectedStage: String?
    public var tableOverride: ParsedTable?
    public init() {}
}
public struct CorrectionStore: Codable, Equatable, Sendable {
    public var values: [String: BlockCorrection] = [:]
    public init() {}
    public func applying(to blocks: [ParsedBlock], includeExcluded: Bool = false) -> [ParsedBlock] {
        blocks.compactMap { original -> ParsedBlock? in
            guard let c = values[original.id] else { return original }
            if c.excluded && !includeExcluded { return nil }
            var b = original
            if let text = c.text { b.text = text }
            if let cls = c.semanticClass { b.semanticClass = cls }
            if let order = c.readingOrder { b.readingOrder = order }
            if let replacement = c.tableOverride { b.table = replacement }
            if var table = b.table {
                for index in table.cells.indices { if let value = c.tableCells[table.cells[index].id] { table.cells[index].text = value; table.cells[index].html = nil } }
                b.table = table
            }
            return b
        }.sorted { $0.readingOrder == $1.readingOrder ? $0.id < $1.id : $0.readingOrder < $1.readingOrder }
    }
}

public enum StudioError: LocalizedError {
    case message(String)
    case backendFailure(String, String)
    public var errorDescription: String? { switch self { case .message(let text): return text; case .backendFailure(let text, _): return text } }
}

/// Lossless JSON envelope for vendor metadata not used directly by the UI.
public enum JSONValue: Codable, Sendable {
    case object([String: JSONValue]), array([JSONValue]), string(String), number(Double), bool(Bool), null
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let v = try? c.decode(Bool.self) { self = .bool(v) }
        else if let v = try? c.decode(Double.self) { self = .number(v) }
        else if let v = try? c.decode(String.self) { self = .string(v) }
        else if let v = try? c.decode([JSONValue].self) { self = .array(v) }
        else { self = .object(try c.decode([String: JSONValue].self)) }
    }
    public func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self { case .object(let v): try c.encode(v); case .array(let v): try c.encode(v); case .string(let v): try c.encode(v); case .number(let v): try c.encode(v); case .bool(let v): try c.encode(v); case .null: try c.encodeNil() }
    }
}
