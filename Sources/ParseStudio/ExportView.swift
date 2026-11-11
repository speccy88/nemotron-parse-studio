import SwiftUI
import AppKit
#if SWIFT_PACKAGE
import ParseCore
#endif

struct ExportView: View {
    @EnvironmentObject var model: StudioModel
    @Environment(\.dismiss) var dismiss
    @State private var scope = "document"
    @State private var format = "json"
    @State private var working = false
    @State private var error: String?
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Export extraction").font(.title2.weight(.semibold))
            Picker("Scope", selection: $scope) { Text("Selected page").tag("page"); Text("Entire document").tag("document"); Text("All documents").tag("batch") }
            Picker("Format", selection: $format) {
                Text("Complete structured JSON").tag("json")
                Text("Markdown (merged cells as HTML)").tag("md")
                Text("Safe HTML").tag("html")
                Text("Plain text").tag("txt")
                Text("Per-table CSV + span manifest").tag("csv")
                Text("Hierarchical table JSON").tag("tables")
                Text("Annotated PDF").tag("pdf")
            }
            Text("Exports apply corrections and reading order. Structured JSON also preserves raw predictions, settings, warnings, stage responses, and the correction overlay.").font(.caption).foregroundStyle(.secondary)
            Text("Searchable PDF is unavailable: block boxes alone cannot provide a correctly aligned line-by-line text layer.").font(.caption).foregroundStyle(.secondary)
            if let error { Text(error).font(.caption).foregroundStyle(.red).textSelection(.enabled) }
            Spacer()
            HStack { Button("Cancel") { dismiss() }; Spacer(); if working { ProgressView().controlSize(.small) }; Button("Choose Destination…", action: chooseDestination).buttonStyle(.borderedProminent).disabled(working) }
        }.padding(24)
    }
    private var exportFormat: ExportFormat {
        switch format { case "md": return .markdown; case "html": return .html; case "txt": return .plainText; case "csv": return .tableCSV; case "tables": return .hierarchicalTablesJSON; case "pdf": return .annotatedPDF; default: return .structuredJSON }
    }
    private func chooseDestination() {
        let chosen: [(DocumentState, PageState)] = model.documents.flatMap { doc in
            doc.pages.compactMap { page in
                guard page.result != nil, page.render != nil else { return nil }
                guard scope == "batch" || doc.id == model.selectedDocumentID else { return nil }
                guard scope != "page" || page.number == model.selectedPageNumber else { return nil }
                return (doc, page)
            }
        }
        guard !chosen.isEmpty else { error = "No parsed pages in this scope."; return }
        let expected = model.documents.filter { scope == "batch" || $0.id == model.selectedDocumentID }.reduce(0) { $0 + (scope == "page" ? 1 : $1.pages.count) }
        if chosen.count < expected { error = "Only \(chosen.count) of \(expected) pages are parsed. Parse all pages in this scope before exporting."; return }
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "Nemotron-Extraction.\(format == "tables" ? "json" : format == "csv" ? "tables" : format)"
        if !model.preferences.outputDirectory.isEmpty { panel.directoryURL = URL(fileURLWithPath: model.preferences.outputDirectory) }
        panel.canCreateDirectories = true
        panel.begin { response in
            guard response == .OK, let destination = panel.url else { return }
            working = true
            Task {
                do {
                    let pages = try chosen.map { doc, page -> ExportPage in
                        let result = page.result!, render = page.render!
                        let blocks = page.corrections.applying(to: result.blocks).map { b in
                            ExportBlock(id: b.id, text: b.text, semanticClass: b.semanticClass, bbox: b.bbox, readingOrder: b.readingOrder, table: b.table.map { t in
                                ExportTable(cells: t.cells.map { ExportCell(row: $0.row, column: $0.column, rowSpan: $0.rowSpan, columnSpan: $0.columnSpan, text: $0.text, isHeader: $0.isHeader, html: $0.html) }, rowCount: t.rowCount, columnCount: t.columnCount, hierarchicalJSON: t.hierarchicalJSON, caption: t.caption)
                            })
                        }
                        return ExportPage(documentName: doc.document.name, pageNumber: page.number + 1, rendered: render, blocks: blocks, raw: result.raw, metadata: try JSONEncoder().encode(page), originalURL: doc.document.originalURL, workingURL: doc.document.workingURL)
                    }
                    try await DocumentExporter().write(pages: pages, format: exportFormat, to: destination)
                    working = false; dismiss(); NSWorkspace.shared.activateFileViewerSelecting([destination])
                } catch { working = false; self.error = error.localizedDescription }
            }
        }
    }
}
