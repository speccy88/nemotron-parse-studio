import Foundation

public protocol InferenceBackend: Sendable {
    func parse(imageURL: URL, width: Int, height: Int, settings: InferenceSettings, options: PostprocessOptions, requestID: String) async throws -> ParseResult
    func postprocess(result: ParseResult, width: Int, height: Int, options: PostprocessOptions) async throws -> ParseResult
}

public actor ReferenceBackend: InferenceBackend {
    public let worker: PythonWorker
    public init(worker: PythonWorker) { self.worker = worker }
    public func health() async throws -> String {
        let data = try await request(path: "health", body: nil)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        guard object?["service"] as? String == "Nemotron Parse Studio", object?["transport"] as? String == "stdio", object?["revision"] as? String == pinnedRevision else { throw StudioError.message("Unexpected Python worker identity or revision.") }
        return String(decoding: data, as: UTF8.self)
    }
    public func parse(imageURL: URL, width: Int, height: Int, settings: InferenceSettings, options: PostprocessOptions, requestID: String) async throws -> ParseResult {
        struct Body: Encodable { let image_base64: String; let width: Int; let height: Int; let settings: InferenceSettings; let options: PostprocessOptions; let request_id: String }
        try Task.checkCancellation()
        let encoded = try JSONEncoder().encode(Body(image_base64: try Data(contentsOf: imageURL).base64EncodedString(), width: width, height: height, settings: settings, options: options, request_id: requestID))
        let data = try await request(path: "parse", body: encoded)
        var result = try JSONDecoder().decode(ParseResult.self, from: data)
        result.sourceResponse = try JSONDecoder().decode(JSONValue.self, from: data)
        return result
    }
    public func postprocess(result: ParseResult, width: Int, height: Int, options: PostprocessOptions) async throws -> ParseResult {
        struct Body: Encodable { let raw: String; let originalRaw: String; let width: Int; let height: Int; let options: PostprocessOptions }
        let encoded = try JSONEncoder().encode(Body(raw: result.mergedRaw ?? result.raw, originalRaw: result.raw, width: width, height: height, options: options))
        let data = try await request(path: "postprocess", body: encoded)
        var updated = try JSONDecoder().decode(ParseResult.self, from: data)
        updated.sourceResponse = result.sourceResponse
        updated.postprocessingResponse = try JSONDecoder().decode(JSONValue.self, from: data)
        for index in updated.blocks.indices { if let original = result.blocks.first(where: { $0.id == updated.blocks[index].id }) { updated.blocks[index].provenance = original.provenance } }
        updated.raw = result.raw
        updated.mergedRaw = result.mergedRaw
        updated.stages = result.stages
        updated.backend = result.backend
        updated.revision = result.revision
        updated.metrics.merge(result.metrics.filter { $0.key != "postprocess_seconds" }) { old, _ in old }
        return updated
    }
    public func testConnection(settings: InferenceSettings) async throws -> String {
        struct Body: Encodable { let settings: InferenceSettings }
        let data = try await request(path: "test-connection", body: JSONEncoder().encode(Body(settings: settings)))
        return String(decoding: data, as: UTF8.self)
    }
    public func cancel(requestID: String) async {
        _ = try? await request(path: "cancel", body: JSONSerialization.data(withJSONObject: ["request_id": requestID]))
    }
    private func request(path: String, body: Data?) async throws -> Data {
        try await worker.request(method: path, body: body)
    }

}

public struct MockBackend: InferenceBackend {
    public init() {}
    public func parse(imageURL: URL, width: Int, height: Int, settings: InferenceSettings, options: PostprocessOptions, requestID: String) async throws -> ParseResult {
        try await Task.sleep(for: .milliseconds(250))
        let block = ParsedBlock(id: "mock-title", rawText: "Demonstration — no inference performed", text: "Demonstration — no inference performed", semanticClass: "Title", bbox: [0.08,0.08,0.9,0.17], readingOrder: 0, provenance: "deterministic mock", warnings: [])
        return ParseResult(raw: "MOCK ONLY", mergedRaw: "MOCK ONLY", markdown: "# Demonstration — no inference performed", blocks: [block], warnings: ["MOCK OUTPUT: this is a UI fixture, not a parsed document."], stages: [], metrics: ["inference_seconds": 0, "request_count": 0], backend: "mock", revision: "none")
    }
    public func postprocess(result: ParseResult, width: Int, height: Int, options: PostprocessOptions) async throws -> ParseResult { result }
}
