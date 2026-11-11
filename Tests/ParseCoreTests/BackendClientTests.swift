import XCTest
import Foundation
@testable import ParseCore

final class BackendClientTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testPinnedDefaultsMatchNVIDIA() {
        let options = PostprocessOptions()
        XCTAssertTrue(options.infer_headers); XCTAssertTrue(options.render_charts)
        XCTAssertTrue(options.normalize_text); XCTAssertTrue(options.deduplicate)
        XCTAssertFalse(options.split_chart_value_percent); XCTAssertFalse(options.relocate_chart_captions)
        XCTAssertEqual(InferenceSettings().revision, pinnedRevision)
        XCTAssertEqual(InferenceSettings().max_new_tokens, 8744)
        XCTAssertEqual(InferenceSettings().max_crops, 80)
        XCTAssertEqual(InferenceSettings().crop_margin, 0.04)
    }

    func testCoordinateMappingClipsReordersAndRejectsNonFinite() throws {
        let box = try XCTUnwrap(NormalizedBox([0.75, 0.8, 0.25, 0.2]))
        let rect = box.rect(width: 1200, height: 800)
        XCTAssertEqual(rect.minX, 300); XCTAssertEqual(rect.minY, 160)
        XCTAssertEqual(rect.width, 600); XCTAssertEqual(rect.height, 480, accuracy: 0.0001)
        let clipped = try XCTUnwrap(NormalizedBox([-1, -2, 3, 4]))
        XCTAssertEqual(clipped.rect(width: 100, height: 200), CGRect(x: 0, y: 0, width: 100, height: 200))
        XCTAssertNil(NormalizedBox([0, 0, .nan, 1]))
        XCTAssertNil(NormalizedBox([0, 0, .infinity, 1]))
        XCTAssertNil(NormalizedBox([0, 1]))
    }

    func testCorrectionsAreSeparateAndCanRestoreOriginal() throws {
        let block = ParsedBlock(id: "stable", rawText: "Original model text", text: "Formatted", semanticClass: "Text", bbox: [0.1,0.2,0.9,0.3], readingOrder: 1, provenance: "fullpage", warnings: [])
        var correction = BlockCorrection()
        correction.text = "Manual text"; correction.semanticClass = "Title"; correction.readingOrder = 0
        var store = CorrectionStore()
        store.values[block.id] = correction
        let edited = try XCTUnwrap(store.applying(to: [block]).first)
        XCTAssertEqual(edited.text, "Manual text"); XCTAssertEqual(edited.semanticClass, "Title"); XCTAssertEqual(edited.readingOrder, 0)
        XCTAssertEqual(edited.rawText, block.rawText); XCTAssertEqual(block.text, "Formatted")
        var reformatted = block; reformatted.text = "New postprocessed format"
        XCTAssertEqual(store.applying(to: [reformatted]).first?.text, "Manual text")
        correction.excluded = true; store.values[block.id] = correction
        XCTAssertTrue(store.applying(to: [block]).isEmpty)
        XCTAssertEqual(store.applying(to: [block], includeExcluded: true).count, 1)
        store.values.removeValue(forKey: block.id)
        XCTAssertEqual(store.applying(to: [block]).first, block)
    }

    func testTableCellCorrectionsRetainOriginalValuesAndSpans() throws {
        var cell = TableCell(row: 0, column: 0, rowSpan: 2, columnSpan: 3, text: "0032", isHeader: true)
        cell.html = "<b>0032</b>"
        let table = ParsedTable(cells: [cell], rowCount: 2, columnCount: 3, html: "<table></table>", markdown: "", latex: "raw latex", csv: "0032", recordsJSON: "[]", hierarchicalJSON: "{}")
        let block = ParsedBlock(id: "table", rawText: "immutable table", text: "table", semanticClass: "Table", bbox: [0,0,1,1], readingOrder: 0, table: table, provenance: "fullpage", warnings: [])
        var correction = BlockCorrection(); correction.tableCells[cell.id] = "0048"
        var store = CorrectionStore(); store.values[block.id] = correction
        let corrected = try XCTUnwrap(store.applying(to: [block]).first?.table?.cells.first)
        XCTAssertEqual(corrected.text, "0048"); XCTAssertEqual(corrected.rowSpan, 2); XCTAssertEqual(corrected.columnSpan, 3)
        XCTAssertTrue(corrected.isHeader)
        XCTAssertNil(corrected.html, "Manual text cannot retain stale model inline markup")
        XCTAssertEqual(block.table?.cells.first?.html, "<b>0032</b>")
        XCTAssertEqual(block.table?.cells.first?.text, "0032")
        XCTAssertEqual(block.table?.latex, "raw latex")
    }

    func testMockIsExplicitlyLabelledAndCancellable() async throws {
        let backend = MockBackend(), path = URL(fileURLWithPath: "/unused-mock-input")
        let result = try await backend.parse(imageURL: path, width: 800, height: 600, settings: InferenceSettings(), options: PostprocessOptions(), requestID: "mock")
        XCTAssertEqual(result.backend, "mock")
        XCTAssertTrue(result.warnings.contains { $0.contains("MOCK") })
        let task = Task { try await backend.parse(imageURL: path, width: 800, height: 600, settings: InferenceSettings(), options: PostprocessOptions(), requestID: "cancel") }
        task.cancel()
        do { _ = try await task.value; XCTFail("Cancelled mock completed") } catch is CancellationError { }
    }

    func testWorkerPersistsAndCorrelatesConcurrentResponses() async throws {
        let worker = PythonWorker(projectRoot: root)
        let backend = ReferenceBackend(worker: worker)
        let first = try await backend.health()
        let firstJSON = try JSONSerialization.jsonObject(with: Data(first.utf8)) as! [String: Any]
        let pid = firstJSON["pid"] as! Int
        XCTAssertEqual(firstJSON["transport"] as? String, "stdio")
        try await withThrowingTaskGroup(of: Void.self) { group in
            for index in 0..<6 { group.addTask {
                let raw = "<x_0><y_0>request-\(index)<x_1><y_1><class_Text>"
                let input = ParseResult(raw: raw, markdown: "", blocks: [], warnings: [], stages: [], metrics: [:], backend: "transformers", revision: pinnedRevision)
                let result = try await backend.postprocess(result: input, width: 100, height: 100, options: PostprocessOptions())
                XCTAssertEqual(result.blocks.first?.text, "request-\(index)")
            } }
            try await group.waitForAll()
        }
        let secondJSON = try JSONSerialization.jsonObject(with: Data(try await backend.health().utf8)) as! [String: Any]
        XCTAssertEqual(secondJSON["pid"] as? Int, pid)
        let probe = Process(); probe.executableURL = URL(fileURLWithPath: "/usr/sbin/lsof"); probe.arguments = ["-a", "-p", String(pid), "-i", "-Fn"]
        let output = Pipe(); probe.standardOutput = output; probe.standardError = FileHandle.nullDevice
        try probe.run(); probe.waitUntilExit()
        XCTAssertEqual(output.fileHandleForReading.readDataToEndOfFile().count, 0, "Postprocessing worker must own no sockets")
        await worker.stop()
        let restartedJSON = try JSONSerialization.jsonObject(with: Data(try await backend.health().utf8)) as! [String: Any]
        XCTAssertNotEqual(restartedJSON["pid"] as? Int, pid)
        await worker.stop()
    }

    func testWorkerErrorsDoNotBreakProtocol() async throws {
        let worker = PythonWorker(projectRoot: root)
        do { _ = try await worker.request(method: "unknown-method"); XCTFail("Unknown method accepted") }
        catch { XCTAssertTrue(error.localizedDescription.contains("Unknown worker method")) }
        let backend = ReferenceBackend(worker: worker)
        let health = try await backend.health(); XCTAssertTrue(health.contains(pinnedRevision))
        await worker.stop()
    }

    func testDecodeActualWorkerResponseFixture() throws {
        let path = root.appendingPathComponent("Backend/fixtures/stdio_model_response.json")
        let result = try JSONDecoder().decode(ParseResult.self, from: Data(contentsOf: path))
        XCTAssertEqual(result.revision, pinnedRevision)
        XCTAssertEqual(result.backend, "transformers")
        XCTAssertFalse(result.raw.isEmpty)
        XCTAssertTrue(result.blocks.contains { $0.semanticClass == "Table" && $0.table != nil })
        XCTAssertTrue(result.blocks.allSatisfy { $0.bounds != nil })
    }

    /// Launches an isolated persistent worker; no model inference is rerun here.
    func testLivePostprocessingKeepsRawAndManualCorrections() async throws {
        let worker = PythonWorker(projectRoot: root)
        let backend = ReferenceBackend(worker: worker)
        let health = try await backend.health()
        XCTAssertTrue(health.contains(pinnedRevision))
        let raw = "<x_0.1><y_0.1>soft\u{00AD}hyphen<x_0.9><y_0.2><class_Text>"
        let originalEnvelope = JSONValue.object(["raw": .string(raw), "revision": .string(pinnedRevision), "usage": .object(["completion_tokens": .number(42)])])
        let input = ParseResult(raw: raw, mergedRaw: raw, markdown: "", blocks: [], warnings: [], stages: [], metrics: ["inference_seconds": 7], sourceResponse: originalEnvelope, backend: "transformers", revision: pinnedRevision)
        let first = try await backend.postprocess(result: input, width: 800, height: 600, options: PostprocessOptions())
        var options = PostprocessOptions(); options.normalize_text = false
        let second = try await backend.postprocess(result: first, width: 800, height: 600, options: options)
        XCTAssertEqual(first.raw, raw); XCTAssertEqual(second.raw, raw)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
        XCTAssertEqual(try encoder.encode(second.sourceResponse), try encoder.encode(originalEnvelope))
        XCTAssertNotNil(first.postprocessingResponse); XCTAssertNotNil(second.postprocessingResponse)
        let postprocessJSON = try JSONSerialization.jsonObject(with: encoder.encode(second.postprocessingResponse)) as? [String: Any]
        let appliedOptions = postprocessJSON?["options"] as? [String: Bool]
        XCTAssertEqual(appliedOptions?["normalize_text"], false)
        XCTAssertEqual(first.blocks.first?.id, second.blocks.first?.id)
        XCTAssertEqual(second.backend, "transformers"); XCTAssertEqual(second.metrics["inference_seconds"], 7)
        XCTAssertNotEqual(first.blocks.first?.text, second.blocks.first?.text)
        let block = try XCTUnwrap(first.blocks.first)
        var store = CorrectionStore(), correction = BlockCorrection(); correction.text = "Manual override"
        store.values[block.id] = correction
        XCTAssertEqual(store.applying(to: second.blocks).first?.text, "Manual override")
        await worker.stop()
    }
}
