import SwiftUI
#if SWIFT_PACKAGE
import ParseCore
#endif

struct InspectorView: View {
    @EnvironmentObject var model: StudioModel
    @State private var editBlock: ParsedBlock?
    var body: some View {
        VStack(spacing: 0) {
            HStack { Text("EXTRACTION").font(.caption.weight(.semibold)).foregroundStyle(.secondary); Spacer(); if let result = model.currentPage?.result { Text(result.backend == "mock" ? "DEMO" : result.backend.uppercased()).font(.caption2.weight(.bold)).padding(5).background(result.backend == "mock" ? Color.orange.opacity(0.2) : Color.green.opacity(0.12), in: Capsule()) } }.padding(14)
            Picker("Inspector", selection: $model.selectedInspector) { ForEach(["Content", "Blocks", "Tables", "Raw", "Postprocess"], id: \.self) { Text($0).tag($0) } }.pickerStyle(.segmented).padding(.horizontal, 10).padding(.bottom, 10)
            Divider()
            if let page = model.currentPage, let result = page.result {
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        switch model.selectedInspector {
                        case "Blocks": blocksView
                        case "Tables": tablesView
                        case "Raw": rawView(result)
                        case "Postprocess": postprocessView(result)
                        default: contentView(result)
                        }
                    }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                ContentUnavailableView { Label("Ready to extract", systemImage: "text.viewfinder") } description: { Text("Choose Parse to run NVIDIA Nemotron. Raw predictions, tables, and postprocessing will appear here.") } actions: { Button("Parse Selected Page") { model.parse() }.disabled(model.currentPage == nil || model.isBusy) }.frame(maxHeight: .infinity)
            }
        }
        .sheet(item: $editBlock) { block in BlockEditor(block: block).environmentObject(model).frame(width: 600, height: 600) }
        .sheet(item: $model.cropReview) { review in
            VStack(alignment: .leading, spacing: 16) {
                Text("Choose a crop prediction").font(.title2.weight(.semibold))
                Text("Replaces the selected block’s content and table. Its page coordinates and immutable model predictions are preserved. Undo restores the previous choice.").font(.caption).foregroundStyle(.secondary)
                ScrollView { ForEach(review.candidates) { candidate in
                    VStack(alignment: .leading) { Text(candidate.semanticClass).font(.headline); Text(candidate.text).font(.caption.monospaced()).textSelection(.enabled); Button("Use This Prediction") { model.acceptCrop(candidate, review: review) } }.padding().background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                } }
                Button("Cancel") { model.cropReview = nil }
            }.padding(24).frame(width: 620, height: 620)
        }
    }
    private func displayText(_ block: ParsedBlock) -> String {
        if block.semanticClass.lowercased().contains("title") || block.semanticClass.lowercased().contains("header") { return block.text.replacingOccurrences(of: #"^#{1,6}\s+"#, with: "", options: .regularExpression) }
        return block.text
    }
    @ViewBuilder private func contentView(_ result: ParseResult) -> some View {
        if result.backend == "mock" { Label("Demonstration fixture — no parsing performed", systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.callout) }
        ForEach(model.currentPage?.corrections.applying(to: result.blocks) ?? []) { block in
            VStack(alignment: .leading, spacing: 7) {
                if let table = block.table {
                    Text(block.semanticClass).font(.caption.weight(.semibold)).foregroundStyle(classColor(block.semanticClass))
                    NativeTableView(table: table, blockID: block.id).environmentObject(model)
                } else {
                    // SwiftUI Text never executes HTML/scripts or loads remote image resources.
                    Text((try? AttributedString(markdown: displayText(block), options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(block.text))
                        .font((block.semanticClass.lowercased().contains("title") || block.semanticClass.lowercased().contains("header")) ? .title2.weight(.semibold) : .body)
                        .textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding(10).background(model.selectedBlockID == block.id ? Color.accentColor.opacity(0.08) : .clear, in: RoundedRectangle(cornerRadius: 6)).onTapGesture { model.selectedBlockID = block.id }
        }
        if result.blocks.isEmpty { Text(result.markdown.isEmpty ? "No blocks were returned. The raw response is preserved in Raw." : result.markdown).textSelection(.enabled) }
        metadata(result)
    }
    private var blocksView: some View {
        VStack(spacing: 10) {
            ForEach(model.correctedBlocks) { block in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("\(block.readingOrder + 1)").font(.caption.monospaced()).foregroundStyle(.secondary)
                        Text(block.semanticClass).font(.caption.weight(.semibold)).foregroundStyle(classColor(block.semanticClass))
                        Spacer()
                        if model.currentPage?.corrections.values[block.id] != nil { Image(systemName: "pencil.circle.fill").foregroundStyle(.orange).help("Manual correction preserved") }
                        Button("Edit") { editBlock = block }.controlSize(.small)
                    }
                    Text(block.text).font(.callout).lineLimit(model.selectedBlockID == block.id ? nil : 4).textSelection(.enabled)
                    if model.currentPage?.corrections.values[block.id]?.excluded == true { Label("Excluded from export", systemImage: "eye.slash").font(.caption).foregroundStyle(.secondary) }
                    if model.selectedBlockID == block.id {
                        Text("Normalized: \(block.bbox.map { String(format: "%.4f", $0) }.joined(separator: ", "))").font(.caption.monospaced()).textSelection(.enabled)
                        if let render = model.currentPage?.render, let bounds = block.bounds {
                            let rect = bounds.rect(width: Double(render.width), height: Double(render.height))
                            Text("Pixels: x \(Int(rect.minX)), y \(Int(rect.minY)), w \(Int(rect.width)), h \(Int(rect.height))").font(.caption.monospaced())
                        }
                        Text("Source: \(block.provenance)").font(.caption).foregroundStyle(.secondary)
                        ForEach(block.warnings, id: \.self) { Text($0).foregroundStyle(.orange).font(.caption) }
                    }
                }.padding(12).frame(maxWidth: .infinity, alignment: .leading).background(model.selectedBlockID == block.id ? Color.accentColor.opacity(0.1) : Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8)).overlay(RoundedRectangle(cornerRadius: 8).stroke(model.selectedBlockID == block.id ? Color.accentColor : .clear)).onTapGesture { model.selectedBlockID = block.id }
            }
        }
    }
    @ViewBuilder private var tablesView: some View {
        let tables = model.correctedBlocks.filter { $0.table != nil }
        if tables.isEmpty { Text("No convertible tables or chart grids on this page. Malformed source remains available in Raw and Blocks.").foregroundStyle(.secondary) }
        ForEach(tables) { block in
            VStack(alignment: .leading, spacing: 10) {
                Text("\(block.semanticClass) · block \(block.readingOrder + 1)").font(.headline)
                if let table = block.table { NativeTableView(table: table, blockID: block.id); TableFormatsView(table: table) }
            }
        }
    }
    private func rawView(_ result: ParseResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Immutable full-page prediction").font(.headline)
            Text(result.raw).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
            if let merged = result.mergedRaw, merged != result.raw {
                DisclosureGroup("Merged prediction used for postprocessing") { Text(merged).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }
            }
            ForEach(Array(result.stages.enumerated()), id: \.offset) { _, stage in
                DisclosureGroup("\(stage.name) · \(stage.finishReason ?? "complete")") {
                    Text(stage.raw).font(.system(.caption, design: .monospaced)).textSelection(.enabled)
                    if let bbox = stage.cropBBoxPixels { Text("Crop pixels: \(bbox.description)").font(.caption.monospaced()) }
                    if stage.cropBBoxPixels != nil && model.selectedBlock != nil {
                        Button("Review crop candidates for selected block…") { model.reviewCrop(stage) }
                            .help("Choose a postprocessed Table or Chart crop as a manual content correction. The page bounding box is preserved.")
                    }
                }
            }
            metadata(result)
        }
    }
    private func postprocessView(_ result: ParseResult) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Refine without another model call").font(.headline)
            Text("NVIDIA’s pinned postprocessing. Manual corrections remain a separate overlay.").font(.caption).foregroundStyle(.secondary)
            OptionToggles(options: $model.postOptions)
            HStack {
                Button("Apply Formatting", action: model.reapply).buttonStyle(.borderedProminent).disabled(model.isPostprocessing || model.isBusy)
                Button("Reset to Defaults") { model.postOptions = PostprocessOptions() }
                if model.isPostprocessing { ProgressView().controlSize(.small) }
            }
            if let page = model.currentPage {
                let ids = Set(result.blocks.map(\.id)); let orphans = page.corrections.values.keys.filter { !ids.contains($0) }.count
                if orphans > 0 { Text("\(orphans) correction(s) belong to blocks hidden by these formatting options. They remain saved and will reappear when matching blocks return.").font(.caption).foregroundStyle(.orange) }
            }
            Divider()
            HStack(alignment: .top, spacing: 12) {
                VStack(alignment: .leading) { Text("Original").font(.caption.weight(.bold)); Text(result.raw).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }.frame(maxWidth: .infinity, alignment: .leading)
                Divider()
                VStack(alignment: .leading) { Text("Postprocessed").font(.caption.weight(.bold)); Text(result.markdown).font(.system(.caption, design: .monospaced)).textSelection(.enabled) }.frame(maxWidth: .infinity, alignment: .leading)
            }
        }
    }
    private func metadata(_ result: ParseResult) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Divider()
            ForEach(result.warnings, id: \.self) { Label($0, systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption) }
            DisclosureGroup("Timing & provenance") {
                VStack(alignment: .leading, spacing: 5) {
                    Text("Backend: \(result.backend)")
                    Text("Revision: \(result.revision)").textSelection(.enabled)
                    ForEach(result.metrics.keys.sorted(), id: \.self) { key in HStack { Text(key.replacingOccurrences(of: "_", with: " ")); Spacer(); Text(String(format: "%.3f", result.metrics[key] ?? 0)).monospacedDigit() } }
                    if let settings = model.currentPage?.appliedSettings { Text("Mode: \(settings.mode) · token cap: \(settings.max_new_tokens)") }
                }.font(.caption).foregroundStyle(.secondary)
            }
        }
    }
}

struct OptionToggles: View {
    @Binding var options: PostprocessOptions
    var body: some View {
        Group {
            Toggle("Infer table headers", isOn: $options.infer_headers)
            Toggle("Render chart tables", isOn: $options.render_charts)
            Toggle("Split chart values and percentages", isOn: $options.split_chart_value_percent)
            Toggle("Relocate chart captions", isOn: $options.relocate_chart_captions)
            Toggle("Normalize text", isOn: $options.normalize_text)
            Toggle("Remove exact duplicate blocks", isOn: $options.deduplicate)
        }.toggleStyle(.checkbox)
    }
}

struct NativeTableView: View {
    @EnvironmentObject var model: StudioModel
    var table: ParsedTable
    var blockID: String
    @State private var editing: TableCell?
    private let columnWidth = 135.0
    private let rowHeight = 56.0
    var body: some View {
        if let caption = table.caption, !caption.isEmpty { Text(caption).font(.subheadline.weight(.semibold)).textSelection(.enabled) }
        if let context = table.contextText, !context.isEmpty { Text(context).font(.caption).textSelection(.enabled) }
        ScrollView([.horizontal, .vertical]) {
            ZStack(alignment: .topLeading) {
                Color(nsColor: .textBackgroundColor)
                ForEach(table.cells) { cell in
                    Button { editing = cell } label: {
                        Text(cell.text.isEmpty ? " " : cell.text).font(.system(size: 12, weight: cell.isHeader ? .semibold : .regular)).multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading).padding(8)
                            .frame(width: columnWidth * Double(cell.columnSpan), height: rowHeight * Double(cell.rowSpan))
                            .background(cell.isHeader ? Color.accentColor.opacity(0.09) : Color.clear)
                            .border(Color.secondary.opacity(0.25), width: 0.5)
                    }.buttonStyle(.plain).offset(x: Double(cell.column) * columnWidth, y: Double(cell.row) * rowHeight)
                        .accessibilityLabel("Row \(cell.row + 1), column \(cell.column + 1), \(cell.text)")
                        .help("\(cell.text)\nSpan \(cell.rowSpan) rows × \(cell.columnSpan) columns. Click to edit.")
                }
            }.frame(width: Double(table.columnCount) * columnWidth, height: Double(table.rowCount) * rowHeight)
        }.frame(height: min(420, Double(table.rowCount) * rowHeight + 16))
        .sheet(item: $editing) { cell in
            TableCellEditor(cell: cell) { text in
                model.edit(blockID: blockID) { $0.tableCells[cell.id] = text }
            }
        }
    }
}

struct TableCellEditor: View {
    @Environment(\.dismiss) private var dismiss
    let cell: TableCell
    let save: (String) -> Void
    @State private var text: String

    init(cell: TableCell, save: @escaping (String) -> Void) {
        self.cell = cell
        self.save = save
        _text = State(initialValue: cell.text)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Edit cell · row \(cell.row + 1), column \(cell.column + 1)").font(.headline)
            Text("Spans and header roles are preserved. This correction will not change the model prediction.").font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $text).font(.body).border(.separator).frame(minHeight: 160)
            HStack {
                Button("Cancel") { dismiss() }
                Spacer()
                Button("Save Correction") { save(text); dismiss() }.buttonStyle(.borderedProminent)
            }
        }.padding(24).frame(width: 480, height: 300)
    }
}

struct TableFormatsView: View {
    let table: ParsedTable
    @State private var format = "HTML"
    var body: some View {
        VStack(alignment: .leading) {
            Picker("Conversion", selection: $format) { ForEach(["HTML", "Markdown", "LaTeX", "CSV", "Record JSON", "Hierarchical JSON"], id: \.self) { Text($0) } }
            Text("Official postprocessed conversion; manual cell changes are applied in exports.").font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.system(.caption, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(10).background(.quaternary, in: RoundedRectangle(cornerRadius: 6))
        }
    }
    private var value: String {
        switch format { case "Markdown": return table.markdown; case "LaTeX": return table.latex; case "CSV": return table.csv; case "Record JSON": return table.recordsJSON; case "Hierarchical JSON": return table.hierarchicalJSON; default: return table.html }
    }
}

struct BlockEditor: View {
    @EnvironmentObject var model: StudioModel
    @Environment(\.dismiss) var dismiss
    let block: ParsedBlock
    @State private var text = ""
    @State private var semanticClass = ""
    @State private var order = 0
    @State private var excluded = false
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Correct extracted block").font(.title2.weight(.semibold))
            HStack {
                TextField("Semantic class", text: $semanticClass)
                Stepper("Order: \(order + 1)", value: $order, in: 0...10000)
            }
            TextEditor(text: $text).font(.system(.body, design: .monospaced)).border(.separator)
            Toggle("Exclude this block from export", isOn: $excluded)
            Text("Original model text").font(.caption.weight(.bold))
            ScrollView { Text(block.rawText).font(.caption.monospaced()).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 80)
            HStack {
                Button("Restore Original") { model.restore(blockID: block.id); dismiss() }
                Spacer(); Button("Cancel") { dismiss() }
                Button("Save Correction") { model.edit(blockID: block.id) { $0.text = text; $0.semanticClass = semanticClass; $0.readingOrder = order; $0.excluded = excluded }; dismiss() }.buttonStyle(.borderedProminent)
            }
        }.padding(24).onAppear { text = block.text; semanticClass = block.semanticClass; order = block.readingOrder; excluded = model.currentPage?.corrections.values[block.id]?.excluded ?? false }
    }
}
