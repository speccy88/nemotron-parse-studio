import Foundation
import ParseCore
@main struct NativeValidate {
    static func main() async throws {
        guard #available(macOS 27.0, *) else { throw StudioError.message("macOS 27 required") }
        let arguments = CommandLine.arguments
        guard arguments.count >= 4 else { throw StudioError.message("Usage: NativeValidate MODEL_DIRECTORY IMAGE OUTPUT_DIRECTORY [MAX_TOKENS]") }
        let root = URL(fileURLWithPath: arguments[1]), image = URL(fileURLWithPath: arguments[2]), out = URL(fileURLWithPath: arguments[3])
        let result = try await NativeCoreAIEngine().generate(imageURL: image, directory: root, maxTokens: arguments.count > 4 ? Int(arguments[4]) ?? 256 : 256, requireValidated: false, validationDirectory: out)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(result).write(to: out.appendingPathComponent("prediction.json"), options: .atomic)
        print(result.raw)
        print(result.metrics)
    }
}
