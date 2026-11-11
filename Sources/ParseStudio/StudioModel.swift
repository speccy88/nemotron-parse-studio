import SwiftUI
import AppKit
#if SWIFT_PACKAGE
import ParseCore
#endif

struct PageState: Codable, Identifiable {
    var number: Int
    var id: Int { number }
    var render: RenderedPage?
    var result: ParseResult?
    var corrections = CorrectionStore()
    var status = "Ready"
    var error: String?
    var failureResponseJSON: String?
    var options = PostprocessOptions()
    var appliedSettings: InferenceSettings?
}
struct DocumentState: Codable, Identifiable {
    var document: ImportedDocument
    var pages: [PageState]
    var id: UUID { document.id }
}
struct Preferences: Codable {
    var settings = InferenceSettings()
    var dpi = 144.0
    var concurrency = 1
    var options = PostprocessOptions()
    var outputDirectory = ""
    var coreAICompute = "Automatic"
}

struct CropReview: Identifiable { let id = UUID(); let documentID: UUID; let page: Int; let blockID: String; let stageName: String; let candidates: [ParsedBlock] }

@MainActor final class StudioModel: ObservableObject {
    @Published var documents: [DocumentState] = []
    @Published var selectedDocumentID: UUID?
    @Published var selectedPageNumber = 0
    @Published var selectedBlockID: String?
    @Published var preferences = Preferences()
    @Published var search = ""
    @Published var isBusy = false
    @Published var isImporting = false
    @Published var notice: String?
    @Published var serviceStatus = "Starting private Python worker…"
    @Published var showSettings = false
    @Published var showExport = false
    @Published var postOptions = PostprocessOptions()
    @Published var isPostprocessing = false
    @Published var showBoxes = true
    @Published var showOrder = true
    @Published var showOnlyFailures = false
    @Published var selectedInspector = "Content"
    @Published var zoom = 1.0
    @Published var fit = "Page"
    @Published var cropReview: CropReview?
    @Published var canUndo = false
    @Published var canRedo = false
    private let loader = DocumentLoader()
    private var queue: Task<Void, Never>?
    private var preview: Task<Void, Never>?
    private var importPanel: NSOpenPanel?
    private let worker: PythonWorker
    private var nativeEngine: Any?
    private var activeRequests: [String: ReferenceBackend] = [:]
    private struct Edit { var doc: UUID; var page: Int; var before: CorrectionStore; var after: CorrectionStore }
    private var undoEdits: [Edit] = []; private var redoEdits: [Edit] = []
    private let sessionURL: URL
    let projectRoot: URL

    init() {
        projectRoot = URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "StudioSourceRoot") as? String ?? FileManager.default.currentDirectoryPath)
        worker = PythonWorker(projectRoot: projectRoot)
        sessionURL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("NemotronParseStudio/session.json")
        if let data = UserDefaults.standard.data(forKey: "studio.preferences"), let saved = try? JSONDecoder().decode(Preferences.self, from: data) { preferences = saved }
        if preferences.settings.model_directory.isEmpty {
            let model = projectRoot.appendingPathComponent("Models")
            if FileManager.default.fileExists(atPath: model.appendingPathComponent("config.json").path) { preferences.settings.model_directory = model.path }
        }
        if let data = try? Data(contentsOf: sessionURL), let saved = try? JSONDecoder().decode([DocumentState].self, from: data) {
            documents = saved
            for d in documents.indices { for p in documents[d].pages.indices { if ["Queued", "Parsing", "Rendering"].contains(documents[d].pages[p].status) { documents[d].pages[p].status = "Interrupted — retry" } } }
        }
        selectedDocumentID = documents.first?.id
        postOptions = preferences.options
        Task { await startServiceIfNeeded(); await ensurePreview() }
    }
    var currentDocument: DocumentState? { documents.first { $0.id == selectedDocumentID } }
    var currentPage: PageState? { currentDocument?.pages.first { $0.number == selectedPageNumber } }
    var correctedBlocks: [ParsedBlock] { guard let p = currentPage else { return [] }; return p.corrections.applying(to: p.result?.blocks ?? [], includeExcluded: true) }
    var selectedBlock: ParsedBlock? { correctedBlocks.first { $0.id == selectedBlockID } }
    var completedPages: Int { documents.flatMap(\.pages).filter { $0.result != nil }.count }
    var totalPages: Int { documents.reduce(0) { $0 + $1.pages.count } }
    var visibleDocuments: [DocumentState] { documents.filter { (search.isEmpty || $0.document.name.localizedCaseInsensitiveContains(search)) && (!showOnlyFailures || $0.pages.contains { $0.error != nil }) } }
    func save() {
        do {
            try FileManager.default.createDirectory(at: sessionURL.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(documents).write(to: sessionURL, options: .atomic)
            UserDefaults.standard.set(try JSONEncoder().encode(preferences), forKey: "studio.preferences")
        } catch { notice = "Could not save workspace: \(error.localizedDescription)" }
    }
    func client() -> ReferenceBackend { ReferenceBackend(worker: worker) }
    func startServiceIfNeeded() async {
        do { _ = try await client().health(); serviceStatus = "Private worker ready" }
        catch { serviceStatus = error.localizedDescription }
    }
    func stopWorker() { Task { await worker.stop() } }
    func testConnection() {
        Task {
            serviceStatus = "Testing…"
            do { serviceStatus = try await client().testConnection(settings: preferences.settings) } catch { serviceStatus = error.localizedDescription }
        }
    }
    func importDialog() {
        if let existing = importPanel { existing.makeKeyAndOrderFront(nil); return }
        let panel = NSOpenPanel(); importPanel = panel; panel.allowsMultipleSelection = true; panel.canChooseDirectories = false
        panel.message = "Import PDF, images, or office documents (office conversion requires LibreOffice)."
        panel.begin { [weak self] response in Task { @MainActor in self?.importPanel = nil; if response == .OK { await self?.importFiles(panel.urls) } } }
    }
    func importFiles(_ urls: [URL]) async {
        isImporting = true; defer { isImporting = false }
        for url in urls {
            do {
                let doc = try await loader.load(url: url)
                documents.append(DocumentState(document: doc, pages: (0..<doc.pageCount).map { PageState(number: $0) }))
                selectedDocumentID = doc.id; selectedPageNumber = 0; selectedBlockID = nil
                save(); await ensurePreview()
            } catch { notice = "\(url.lastPathComponent): \(error.localizedDescription)" }
        }
    }
    func select(doc: UUID, page: Int) {
        selectedDocumentID = doc; selectedPageNumber = page; selectedBlockID = nil
        postOptions = currentPage?.options ?? preferences.options
        preview?.cancel(); preview = Task { await ensurePreview() }
    }
    private func index(_ id: UUID) -> Int? { documents.firstIndex { $0.id == id } }
    func ensurePreview() async {
        guard let doc = currentDocument, currentPage?.render == nil else { return }
        let number = selectedPageNumber
        do {
            let render = try await loader.render(document: doc.document, page: number, dpi: preferences.dpi)
            guard let d = index(doc.id), documents[d].pages.indices.contains(number) else { return }
            if documents[d].pages[number].render == nil { documents[d].pages[number].render = render }
            save()
        } catch { notice = error.localizedDescription }
    }
    func parse(scope: String = "page", retryOnly: Bool = false) {
        guard !isBusy, !isPostprocessing else { return }
        let items: [(UUID, Int)] = documents.flatMap { doc in
            doc.pages.compactMap { page in
                if scope != "batch" && doc.id != selectedDocumentID { return nil }
                if scope == "page" && page.number != selectedPageNumber { return nil }
                if retryOnly && page.error == nil && page.status != "Cancelled" && !page.status.hasPrefix("Interrupted") { return nil }
                return (doc.id, page.number)
            }
        }
        guard !items.isEmpty else { return }
        let settings = preferences.settings, options = preferences.options, dpi = preferences.dpi
        let concurrency = max(1, min(4, preferences.concurrency))
        for (id, p) in items { if let d = index(id) { documents[d].pages[p].status = "Queued"; documents[d].pages[p].error = nil } }
        isBusy = true
        queue = Task {
            await withTaskGroup(of: Void.self) { group in
                var next = 0
                for _ in 0..<min(concurrency, items.count) {
                    let item = items[next]; next += 1
                    group.addTask { await self.runPage(id: item.0, page: item.1, dpi: dpi, settings: settings, options: options) }
                }
                while await group.next() != nil {
                    if Task.isCancelled { group.cancelAll(); break }
                    if next < items.count {
                        let item = items[next]; next += 1
                        group.addTask { await self.runPage(id: item.0, page: item.1, dpi: dpi, settings: settings, options: options) }
                    }
                }
            }
            for (id, p) in items { if let d = index(id), ["Queued", "Parsing", "Rendering"].contains(documents[d].pages[p].status) { documents[d].pages[p].status = "Cancelled" } }
            isBusy = false; save()
        }
    }
    private func runPage(id: UUID, page: Int, dpi: Double, settings: InferenceSettings, options: PostprocessOptions) async {
        guard let d = index(id), !Task.isCancelled else { return }
        let requestID = UUID().uuidString
        let service = client()
        let start = Date()
        do {
            documents[d].pages[page].status = "Rendering"
            let render = try await loader.render(document: documents[d].document, page: page, dpi: dpi)
            try Task.checkCancellation()
            documents[d].pages[page].render = render; documents[d].pages[page].status = "Parsing"
            activeRequests[requestID] = service
            let backend: any InferenceBackend
            if settings.backend == "coreai" {
                if #available(macOS 27.0, *) {
                    let engine = (nativeEngine as? NativeCoreAIEngine) ?? NativeCoreAIEngine(); nativeEngine = engine
                    let directory = settings.native_model_directory.map { URL(fileURLWithPath: $0) } ?? projectRoot.appendingPathComponent("NativeModels")
                    backend = NativeCoreAIBackend(directory: directory, postprocessor: service, engine: engine)
                } else { throw StudioError.message("Native Core AI requires macOS 27 or later.") }
            } else { backend = settings.backend == "mock" ? MockBackend() : service }
            var result = try await backend.parse(imageURL: render.imageURL, width: render.width, height: render.height, settings: settings, options: options, requestID: requestID)
            try Task.checkCancellation()
            result.metrics["render_seconds"] = render.renderSeconds; result.metrics["total_seconds"] = Date().timeIntervalSince(start)
            documents[d].pages[page].result = result
            documents[d].pages[page].appliedSettings = settings
            documents[d].pages[page].options = options
            documents[d].pages[page].status = result.backend == "mock" ? "Demo only" : "Complete"
            documents[d].pages[page].error = nil
            documents[d].pages[page].failureResponseJSON = nil
            if id == selectedDocumentID && page == selectedPageNumber { postOptions = options }
        } catch {
            documents[d].pages[page].status = Task.isCancelled ? "Cancelled" : "Failed"
            documents[d].pages[page].error = Task.isCancelled ? nil : error.localizedDescription
            if case let StudioError.backendFailure(_, payload) = error { documents[d].pages[page].failureResponseJSON = payload }
        }
        activeRequests.removeValue(forKey: requestID); save()
    }
    func cancel() {
        queue?.cancel()
        for (id, service) in activeRequests { Task { await service.cancel(requestID: id) } }
    }
    func reapply() {
        guard !isBusy, !isPostprocessing else { return }
        guard let doc = currentDocument, let page = currentPage, let result = page.result, let render = page.render else { return }
        let options = postOptions
        isPostprocessing = true
        Task {
            defer { isPostprocessing = false }
            do {
                let backend: any InferenceBackend = result.backend == "mock" ? MockBackend() : client()
                let updated = try await backend.postprocess(result: result, width: render.width, height: render.height, options: options)
                guard let d = index(doc.id) else { return }
                documents[d].pages[page.number].result = updated
                documents[d].pages[page.number].options = options
                save()
            } catch { notice = error.localizedDescription }
        }
    }
    func reviewCrop(_ stage: StageResponse) {
        guard let doc = currentDocument, let page = currentPage, let render = page.render, let selected = selectedBlock else { return }
        Task {
            do {
                let rawResult = ParseResult(raw: stage.raw, mergedRaw: stage.raw, markdown: "", blocks: [], warnings: [], stages: [], metrics: [:], backend: "crop-review", revision: pinnedRevision)
                let parsed = try await client().postprocess(result: rawResult, width: render.width, height: render.height, options: page.options)
                let candidates = parsed.blocks.filter { $0.semanticClass == "Table" || $0.semanticClass == "Chart" }
                guard !candidates.isEmpty else { throw StudioError.message("This crop contains no complete Table or Chart candidate. Its raw prediction remains in the stage inspector.") }
                cropReview = CropReview(documentID: doc.id, page: page.number, blockID: selected.id, stageName: stage.name, candidates: candidates)
            } catch { notice = error.localizedDescription }
        }
    }
    func acceptCrop(_ candidate: ParsedBlock, review: CropReview) {
        select(doc: review.documentID, page: review.page)
        selectedBlockID = review.blockID
        edit(blockID: review.blockID) { $0.text = candidate.text; $0.tableOverride = candidate.table; $0.tableCells = [:]; $0.selectedStage = review.stageName }
        cropReview = nil
    }
    func edit(blockID: String, change: (inout BlockCorrection) -> Void) {
        guard let d = documents.firstIndex(where: { $0.id == selectedDocumentID }) else { return }
        let before = documents[d].pages[selectedPageNumber].corrections
        var after = before; var correction = after.values[blockID] ?? BlockCorrection(); change(&correction); after.values[blockID] = correction
        applyEdit(Edit(doc: documents[d].id, page: selectedPageNumber, before: before, after: after))
    }
    func restore(blockID: String) {
        guard let doc = currentDocument, let page = currentPage else { return }
        var after = page.corrections; after.values.removeValue(forKey: blockID)
        applyEdit(Edit(doc: doc.id, page: page.number, before: page.corrections, after: after))
    }
    private func applyEdit(_ edit: Edit) {
        guard let d = index(edit.doc) else { return }
        documents[d].pages[edit.page].corrections = edit.after
        undoEdits.append(edit); redoEdits = []; updateUndo(); save()
    }
    func undo() { guard let edit = undoEdits.popLast(), let d = index(edit.doc) else { return }; documents[d].pages[edit.page].corrections = edit.before; redoEdits.append(edit); updateUndo(); save() }
    func redo() { guard let edit = redoEdits.popLast(), let d = index(edit.doc) else { return }; documents[d].pages[edit.page].corrections = edit.after; undoEdits.append(edit); updateUndo(); save() }
    private func updateUndo() { canUndo = !undoEdits.isEmpty; canRedo = !redoEdits.isEmpty }
}
