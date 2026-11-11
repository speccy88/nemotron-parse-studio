import SwiftUI
#if SWIFT_PACKAGE
import ParseCore
#endif

struct SettingsView: View {
    @EnvironmentObject var model: StudioModel
    @Environment(\.dismiss) var dismiss
    var body: some View {
        VStack(spacing: 0) {
            HStack { Text("Studio Settings").font(.title2.weight(.semibold)); Spacer(); Button("Done") { model.save(); dismiss() }.keyboardShortcut(.defaultAction) }.padding(20)
            Form {
                Section("Inference backend") {
                    Picker("Backend", selection: $model.preferences.settings.backend) {
                        Text("Local NVIDIA Transformers (Python / MPS)").tag("transformers")
                        Text("Local / LAN vLLM server").tag("vllm")
                        Text("Native Core AI · validated fixed geometries").tag("coreai")
                        Text("Deterministic demo — no parsing").tag("mock")
                    }
                    Text("Python companion: persistent --stdio worker over private pipes. No listening port.").font(.caption).foregroundStyle(.secondary)
                    if model.preferences.settings.backend == "vllm" {
                        TextField("vLLM endpoint", text: $model.preferences.settings.server_url)
                        Text("Page images go to this local-network server. Public endpoints and redirects are rejected.").font(.caption).foregroundStyle(.secondary)
                    }
                    TextField("Model name / served name", text: $model.preferences.settings.model)
                    TextField("Local model directory", text: $model.preferences.settings.model_directory)
                    TextField("Model revision", text: $model.preferences.settings.revision).font(.caption.monospaced())
                    Picker("Transformers device", selection: $model.preferences.settings.device) { ForEach(["auto", "mps", "cpu", "cuda"], id: \.self) { Text($0) } }
                    if model.preferences.settings.backend == "coreai" {
                        Text("Compute: CPU float32. Native profiles are validated for 512×256 and 640×800 RGB inputs. Other sizes require local Transformers; no implicit resize.").font(.caption).foregroundStyle(.secondary)
                        TextField("Native model bundle", text: Binding(get: { model.preferences.settings.native_model_directory ?? model.projectRoot.appendingPathComponent("NativeModels").path }, set: { model.preferences.settings.native_model_directory = $0 }))
                        Button("Use Validated Native Settings") {
                            model.preferences.settings.mode = "single"; model.preferences.settings.temperature = 0
                            model.preferences.settings.max_new_tokens = 256; model.preferences.settings.predict_text_in_pic = true
                            model.preferences.concurrency = 1
                        }
                        Text("Single pass, temperature 0, 256-token cap, one page at a time. The model pipeline runs in Swift/Core AI; NVIDIA formatting uses the private Python worker.").font(.caption)
                    }
                    HStack { Button("Test Connection", action: model.testConnection); Button("Check Private Worker") { Task { await model.startServiceIfNeeded() } } }
                    Text(model.serviceStatus).font(.caption).textSelection(.enabled)
                }
                Section("Page rendering & queue") {
                    Picker("Resolution", selection: $model.preferences.dpi) { ForEach([72.0, 96, 144, 200, 300], id: \.self) { Text("\(Int($0)) DPI").tag($0) } }
                    Stepper("Concurrent pages: \(model.preferences.concurrency)", value: $model.preferences.concurrency, in: 1...4)
                    Text("The local Transformers model serializes inference; additional concurrency is mainly useful with vLLM. PDFs are rendered one page at a time.").font(.caption).foregroundStyle(.secondary)
                    TextField("Preferred export directory", text: $model.preferences.outputDirectory)
                }
                Section("Parsing defaults") {
                    Picker("Mode", selection: $model.preferences.settings.mode) { Text("Single pass").tag("single"); Text("Double pass · table/chart crops").tag("double"); Text("Triple pass · independent detection").tag("triple") }
                    Picker("Triple-pass regions", selection: $model.preferences.settings.triple_regions) { Text("Detector + parser union").tag("union"); Text("Detector only").tag("detector") }
                    Toggle("Predict text inside pictures", isOn: $model.preferences.settings.predict_text_in_pic)
                    Text("Full structure control-token prompting is always used. Geometry and class prediction remain enabled for overlays. The pinned helper uses guarded structural crop replacement.").font(.caption).foregroundStyle(.secondary)
                    TextField("Maximum generation tokens", value: $model.preferences.settings.max_new_tokens, format: .number)
                    TextField("Temperature", value: $model.preferences.settings.temperature, format: .number)
                    TextField("Top P", value: $model.preferences.settings.top_p, format: .number)
                    TextField("Top K", value: $model.preferences.settings.top_k, format: .number)
                    TextField("Repetition penalty", value: $model.preferences.settings.repetition_penalty, format: .number)
                    TextField("Random seed", value: $model.preferences.settings.seed, format: .number)
                    Stepper("Maximum crop regions: \(model.preferences.settings.max_crops)", value: $model.preferences.settings.max_crops, in: 0...200)
                    TextField("Crop margin (fraction)", value: $model.preferences.settings.crop_margin, format: .number)
                    TextField("Extra crop margin (pixels)", value: $model.preferences.settings.crop_margin_pixels, format: .number)
                    Text("Full-page settings above; detection token cap 4,096, crop temperature 0.1 and repetition penalty 1.0 follow the pinned helper.").font(.caption).foregroundStyle(.secondary)
                }
                Section("Postprocessing defaults") {
                    OptionToggles(options: $model.preferences.options)
                    Button("Reset Postprocessing Defaults") { model.preferences.options = PostprocessOptions() }
                }
                Section("Local setup") {
                    Text(model.localSetupInstructions).font(.caption).textSelection(.enabled)
                    Text("Workspace: \(model.projectRoot.path)").font(.caption.monospaced()).textSelection(.enabled)
                }
            }.formStyle(.grouped)
        }
    }
}
