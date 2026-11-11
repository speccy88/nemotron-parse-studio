import SwiftUI

@main struct ParseStudioApp: App {
    @StateObject private var model = StudioModel()
    var body: some Scene {
        WindowGroup("Nemotron Parse Studio") {
            WorkspaceView().environmentObject(model).frame(minWidth: 1040, minHeight: 700)
                .onOpenURL { url in Task { await model.importFiles([url]) } }
        }
        .defaultSize(width: 1480, height: 940)
        .commands {
            CommandGroup(replacing: .newItem) { Button("Import Documents…", action: model.importDialog).keyboardShortcut("o") }
            CommandGroup(replacing: .undoRedo) {
                Button("Undo Correction", action: model.undo).keyboardShortcut("z").disabled(!model.canUndo)
                Button("Redo Correction", action: model.redo).keyboardShortcut("z", modifiers: [.command, .shift]).disabled(!model.canRedo)
            }
            CommandMenu("Document") {
                Button("Parse Page") { model.parse() }.keyboardShortcut("r").disabled(model.isBusy || model.currentPage == nil)
                Button("Parse Document") { model.parse(scope: "document") }.keyboardShortcut("r", modifiers: [.command, .shift]).disabled(model.isBusy)
                Button("Cancel", action: model.cancel).keyboardShortcut(".").disabled(!model.isBusy)
                Button("Export…") { model.showExport = true }.keyboardShortcut("e").disabled(model.completedPages == 0)
            }
        }
        Settings { SettingsView().environmentObject(model).frame(width: 680, height: 700) }
    }
}
