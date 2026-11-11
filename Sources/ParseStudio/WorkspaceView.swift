import SwiftUI
import AppKit
import UniformTypeIdentifiers
#if SWIFT_PACKAGE
import ParseCore
#endif

func classColor(_ name: String) -> Color {
    switch name.lowercased() {
    case "title", "section-header", "sectionheader", "heading": return .purple
    case "table": return .orange
    case "chart": return .pink
    case "picture", "figure": return .green
    case "text", "paragraph": return .blue
    default: return .teal
    }
}
struct WorkspaceView: View {
    @EnvironmentObject var model: StudioModel
    var body: some View {
        NavigationSplitView {
            sidebar.navigationSplitViewColumnWidth(min: 210, ideal: 245, max: 380)
        } content: {
            PageViewer().navigationSplitViewColumnWidth(min: 340, ideal: 660)
        } detail: {
            InspectorView().navigationSplitViewColumnWidth(min: 340, ideal: 440, max: 780)
        }
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                Button(action: model.importDialog) { Label("Import", systemImage: "square.and.arrow.down") }.help("Import documents (⌘O)")
                Menu {
                    Button("Selected Page") { model.parse() }
                    Button("Entire Document") { model.parse(scope: "document") }
                    Button("All Documents") { model.parse(scope: "batch") }
                    Button("Retry Failed / Cancelled") { model.parse(scope: "batch", retryOnly: true) }
                } label: { Label("Parse", systemImage: "play.fill") }.disabled(model.isBusy || model.currentPage == nil)
                Button(action: model.cancel) { Label("Cancel", systemImage: "stop.fill") }.disabled(!model.isBusy)
                Button { model.reapply(); model.selectedInspector = "Postprocess" } label: { Label("Reprocess", systemImage: "slider.horizontal.3") }.disabled(model.currentPage?.result == nil || model.isPostprocessing || model.isBusy)
                Button { model.showExport = true } label: { Label("Export", systemImage: "square.and.arrow.up") }.disabled(model.completedPages == 0)
                Button { model.showSettings = true } label: { Label("Settings", systemImage: "gearshape") }
            }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            HStack(spacing: 12) {
                Circle().fill(model.serviceStatus == "Private worker ready" ? .green : .orange).frame(width: 6, height: 6)
                Text(model.serviceStatus == "Private worker ready" ? "Private Python pipes connected" : model.serviceStatus).lineLimit(1)
                Divider().frame(height: 12)
                Text(backendLabel)
                Spacer()
                if model.isBusy || model.isImporting { ProgressView().controlSize(.mini) }
                Text("\(model.completedPages) / \(model.totalPages) pages parsed").monospacedDigit()
            }.font(.caption).foregroundStyle(.secondary).padding(.horizontal, 16).padding(.vertical, 8).background(.bar)
        }
        .onDrop(of: [.fileURL], isTargeted: nil) { providers in
            for provider in providers {
                _ = provider.loadObject(ofClass: URL.self) { url, _ in
                    if let url { Task { @MainActor in await model.importFiles([url]) } }
                }
            }
            return !providers.isEmpty
        }
        .sheet(isPresented: $model.showSettings) { SettingsView().environmentObject(model).frame(width: 680, height: 720) }
        .sheet(isPresented: $model.showExport) { ExportView().environmentObject(model).frame(width: 540, height: 380) }
        .alert("Nemotron Parse Studio", isPresented: Binding(get: { model.notice != nil }, set: { if !$0 { model.notice = nil } })) {
            Button("OK") { model.notice = nil }
        } message: { Text(model.notice ?? "") }
    }
    private var backendLabel: String {
        switch model.preferences.settings.backend {
        case "mock": return "DEMO · no model inference"
        case "vllm": return "LAN vLLM · \(model.preferences.settings.server_url)"
        case "coreai": return "Native Core AI · CPU · validated geometries"
        default: return "Local Transformers · NVIDIA Nemotron 3.5"
        }
    }
    private var sidebar: some View {
        VStack(spacing: 0) {
            HStack { Text("LIBRARY").font(.caption.weight(.semibold)).foregroundStyle(.secondary); Spacer(); Text("\(model.documents.count)").font(.caption).foregroundStyle(.secondary) }.padding()
            TextField("Find a document", text: $model.search).textFieldStyle(.roundedBorder).padding(.horizontal).padding(.bottom, 8)
            Toggle("Failures only", isOn: $model.showOnlyFailures).toggleStyle(.checkbox).font(.caption).padding(.horizontal).padding(.bottom, 8)
            if model.documents.isEmpty {
                ContentUnavailableView { Label("Your documents", systemImage: "doc.on.doc") } description: { Text("Drop PDFs and images here, or import files to begin.") } actions: { Button("Import Documents…", action: model.importDialog) }.frame(maxHeight: .infinity)
            } else {
                List {
                    ForEach(model.visibleDocuments) { doc in
                        Section {
                            ForEach(doc.pages) { page in
                                Button { model.select(doc: doc.id, page: page.number) } label: {
                                    HStack(spacing: 10) {
                                        if let render = page.render, let image = NSImage(contentsOf: render.thumbnailURL) {
                                            Image(nsImage: image).resizable().scaledToFit().frame(width: 38, height: 50).background(.white).clipShape(RoundedRectangle(cornerRadius: 3))
                                        } else { Image(systemName: "doc").font(.title2).foregroundStyle(.secondary).frame(width: 38, height: 50) }
                                        VStack(alignment: .leading, spacing: 4) {
                                            Text("Page \(page.number + 1)").font(.callout.weight(.medium))
                                            HStack(spacing: 4) {
                                                if page.status == "Parsing" || page.status == "Rendering" { ProgressView().controlSize(.mini) }
                                                Text(page.status).font(.caption).foregroundStyle(page.error == nil ? Color.secondary : Color.red)
                                            }
                                        }
                                        Spacer(minLength: 0)
                                        if page.result != nil { Image(systemName: "checkmark.circle.fill").foregroundStyle(page.result?.backend == "mock" ? .orange : .green).font(.caption) }
                                    }.padding(6).background(model.selectedDocumentID == doc.id && model.selectedPageNumber == page.number ? Color.accentColor.opacity(0.14) : .clear, in: RoundedRectangle(cornerRadius: 7))
                                }.buttonStyle(.plain).accessibilityLabel("\(doc.document.name), page \(page.number + 1), \(page.status)")
                            }
                        } header: {
                            VStack(alignment: .leading, spacing: 3) {
                                Text(doc.document.name).font(.callout.weight(.semibold)).foregroundStyle(.primary).lineLimit(2)
                                Text("\(doc.document.kind.uppercased()) · \(doc.pages.count) pages · \(ByteCountFormatter.string(fromByteCount: doc.document.fileSize, countStyle: .file))").font(.caption2)
                            }.padding(.vertical, 4)
                        }
                    }
                }.listStyle(.sidebar)
            }
        }
    }
}

struct PageViewer: View {
    @EnvironmentObject var model: StudioModel
    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 10) {
                Text(model.currentDocument?.document.name ?? "Document viewer").font(.headline).lineLimit(1)
                Spacer()
                Toggle(isOn: $model.showBoxes) { Image(systemName: "viewfinder") }.toggleStyle(.button).help("Show bounding boxes")
                Toggle(isOn: $model.showOrder) { Image(systemName: "list.number") }.toggleStyle(.button).help("Show reading order")
            }.padding(12).background(.bar)
            if let page = model.currentPage {
                if let error = page.error {
                    HStack { Image(systemName: "exclamationmark.triangle"); Text(error).font(.caption); Spacer(); Button("Retry") { model.parse() }.disabled(model.isBusy) }.foregroundStyle(.red).padding(12).background(.red.opacity(0.06))
                    if let failure = page.failureResponseJSON { DisclosureGroup("Preserved failed-stage response") { ScrollView { Text(failure).font(.caption.monospaced()).textSelection(.enabled) }.frame(maxHeight: 140) }.padding(.horizontal, 12) }
                }
                if let render = page.render, let image = NSImage(contentsOf: render.imageURL) {
                    GeometryReader { proxy in
                        let widthFit = max(100, proxy.size.width - 48) / Double(render.width)
                        let pageFit = min(widthFit, max(100, proxy.size.height - 48) / Double(render.height))
                        let scale = (model.fit == "Width" ? widthFit : pageFit) * model.zoom
                        let size = CGSize(width: Double(render.width) * scale, height: Double(render.height) * scale)
                        ScrollView([.horizontal, .vertical]) {
                            ZStack(alignment: .topLeading) {
                                Image(nsImage: image).resizable().frame(width: size.width, height: size.height).background(.white)
                                if model.showBoxes {
                                    ForEach(model.correctedBlocks) { block in
                                        if let box = block.bounds {
                                            let rect = box.rect(width: size.width, height: size.height)
                                            Button { model.selectedBlockID = block.id; model.selectedInspector = "Blocks" } label: {
                                                ZStack(alignment: .topLeading) {
                                                    Rectangle().fill(classColor(block.semanticClass).opacity(model.selectedBlockID == block.id ? 0.22 : 0.03))
                                                    Rectangle().strokeBorder(classColor(block.semanticClass), lineWidth: model.selectedBlockID == block.id ? 3 : 1.2)
                                                    if model.showOrder { Text("\(block.readingOrder + 1)").font(.system(size: 10, weight: .bold)).padding(3).background(classColor(block.semanticClass)).foregroundStyle(.white) }
                                                }
                                            }.buttonStyle(.plain).frame(width: max(1, rect.width), height: max(1, rect.height)).offset(x: rect.minX, y: rect.minY).help("\(block.semanticClass): \(block.text.prefix(120))").accessibilityLabel("Block \(block.readingOrder + 1), \(block.semanticClass)")
                                        }
                                    }
                                }
                            }.frame(width: size.width, height: size.height).shadow(color: .black.opacity(0.15), radius: 12, y: 4).padding(24)
                                .frame(minWidth: proxy.size.width, minHeight: proxy.size.height)
                        }.background(Color(nsColor: .underPageBackgroundColor))
                    }
                    HStack {
                        Picker("Fit", selection: $model.fit) { Text("Fit page").tag("Page"); Text("Fit width").tag("Width") }.labelsHidden().frame(width: 120)
                        Button { model.zoom = max(0.25, model.zoom / 1.25) } label: { Image(systemName: "minus.magnifyingglass") }
                        Text("\(Int(model.zoom * 100))%").font(.caption.monospacedDigit()).frame(width: 42)
                        Button { model.zoom = min(8, model.zoom * 1.25) } label: { Image(systemName: "plus.magnifyingglass") }
                        Button("Reset") { model.zoom = 1 }
                        Spacer()
                        Text("\(render.width) × \(render.height) RGB · page \(page.number + 1)").font(.caption).foregroundStyle(.secondary)
                    }.controlSize(.small).padding(10).background(.bar)
                } else { ProgressView("Rendering page…").frame(maxWidth: .infinity, maxHeight: .infinity) }
            } else {
                ContentUnavailableView { Label("A clearer view of every page", systemImage: "doc.viewfinder") } description: { Text("Import a document to inspect its layout, extract structured content, and refine the result.") } actions: { Button("Import Documents…", action: model.importDialog).buttonStyle(.borderedProminent) }.frame(maxHeight: .infinity)
            }
        }
    }
}
