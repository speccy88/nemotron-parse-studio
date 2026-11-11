import XCTest
import Foundation
@testable import ParseCore

final class NativeCoreAITests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func manifestDirectory() throws -> URL {
        let path = root.appendingPathComponent("NativeModels")
        guard FileManager.default.fileExists(atPath: path.appendingPathComponent("manifest.json").path) else { throw XCTSkip("Native assets not installed") }; return path
    }
    func testNativeGeometryAndCacheLimitsAreEnforcedBeforeModelLoad() async throws {
        guard #available(macOS 27.0, *) else { throw XCTSkip("Core AI requires macOS 27") }
        let directory = try manifestDirectory(), engine = NativeCoreAIEngine()
        do { _ = try await engine.generate(imageURL: root.appendingPathComponent("Samples/mixed-layout.png"), directory: directory, maxTokens: 256); XCTFail("Unmatched geometry accepted") }
        catch { XCTAssertTrue(error.localizedDescription.contains("geometry")) }
        do { _ = try await engine.generate(imageURL: root.appendingPathComponent("Conversion/evidence/pages/simple.png"), directory: directory, maxTokens: 9000); XCTFail("Unbounded KV cache accepted") }
        catch { XCTAssertTrue(error.localizedDescription.contains("at most")) }
    }
    func testUnvalidatedNativeProfileCannotBeSelected() async throws {
        guard #available(macOS 27.0, *) else { throw XCTSkip("Core AI requires macOS 27") }
        let directory = try manifestDirectory()
        var manifest = try JSONDecoder().decode(NativeManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        for index in manifest.profiles.indices { manifest.profiles[index].validated = false }
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        try JSONEncoder().encode(manifest).write(to: temporary.appendingPathComponent("manifest.json"))
        do { _ = try await NativeCoreAIEngine().generate(imageURL: root.appendingPathComponent("Conversion/evidence/pages/simple.png"), directory: temporary, maxTokens: 256); XCTFail("Unvalidated native profile accepted") }
        catch { XCTAssertTrue(error.localizedDescription.contains("not passed")) }
    }
    func testRealNativeImportParseAndPrivatePostprocessing() async throws {
        guard ProcessInfo.processInfo.environment["NEMOTRON_TEST_NATIVE"] == "1" else { throw XCTSkip("Set NEMOTRON_TEST_NATIVE=1 for actual Core AI inference") }
        guard #available(macOS 27.0, *) else { throw XCTSkip("Core AI requires macOS 27") }
        let directory = try manifestDirectory()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let loader = DocumentLoader(storageURL: temporary)
        let doc = try await loader.load(url: root.appendingPathComponent("Conversion/evidence/pages/mixed.png"))
        let page = try await loader.render(document: doc, page: 0, dpi: 144)
        let worker = PythonWorker(projectRoot: root)
        let backend = NativeCoreAIBackend(directory: directory, postprocessor: ReferenceBackend(worker: worker), engine: NativeCoreAIEngine())
        var settings = InferenceSettings(); settings.backend = "coreai"; settings.temperature = 0; settings.max_new_tokens = 256
        let parsed = try await backend.parse(imageURL: page.imageURL, width: page.width, height: page.height, settings: settings, options: PostprocessOptions(), requestID: "native-integration")
        let reference = try String(
            contentsOf: root.appendingPathComponent("Backend/fixtures/native_mixed_reference.txt"),
            encoding: .utf8
        )
        XCTAssertEqual(parsed.raw, reference)
        XCTAssertEqual(parsed.backend, "coreai")
        XCTAssertEqual(parsed.blocks.count, 6)
        XCTAssertEqual(parsed.blocks.filter { $0.table != nil }.count, 2)
        var options = PostprocessOptions(); options.split_chart_value_percent = true
        let formatted = try await backend.postprocess(result: parsed, width: page.width, height: page.height, options: options)
        XCTAssertEqual(formatted.raw, parsed.raw); XCTAssertEqual(formatted.backend, "coreai")
        await worker.stop()
    }
}
