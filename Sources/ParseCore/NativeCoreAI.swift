import Foundation
import ImageIO
import CoreGraphics
#if canImport(CoreAI)
import CoreAI
#endif

public struct NativeProfile: Codable, Sendable {
    public var id: String
    public var imageSizeHW: [Int]
    public var contentSizeHW: [Int]
    public var patchCount: Int
    public var prefixLength: Int
    public var cacheCapacity: Int
    public var promptIDs: [Int]
    public var eosID: Int
    public var encoder: String
    public var prefill: String
    public var step: String
    public var validated: Bool
}
public struct NativeManifest: Codable, Sendable {
    public var schemaVersion: Int
    public var revision: String
    public var profiles: [NativeProfile]
    public var tokenBytes: String
    public var limitations: [String]
}
public struct NativePrediction: Codable, Sendable {
    public var raw: String
    public var tokenIDs: [Int]
    public var profile: String
    public var metrics: [String: Double]
}

/// Inference is entirely Swift + macOS Core AI. NVIDIA formatting remains a
/// separate local service. Only validated geometry-specific profiles are enabled.
@available(macOS 27.0, *)
public actor NativeCoreAIEngine {
    private struct Loaded {
        let root: URL
        let profile: NativeProfile
        let encoder: InferenceFunction
        let prefill: InferenceFunction
        let step: InferenceFunction
        let vocabulary: [Data]
    }
    private var loaded: Loaded?
    private var busy = false
    public init() {}

    public func generate(imageURL: URL, directory: URL, maxTokens: Int, repetitionPenalty: Double = 1.05, requireValidated: Bool = true, validationDirectory: URL? = nil) async throws -> NativePrediction {
        guard !busy else { throw StudioError.message("Native Core AI runs one page at a time. Set batch concurrency to 1.") }
        busy = true; defer { busy = false }
        let totalStart = Date()
        let manifest = try JSONDecoder().decode(NativeManifest.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard manifest.schemaVersion == 1, manifest.revision == pinnedRevision else { throw StudioError.message("Native model manifest does not match the pinned NVIDIA revision.") }
        guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw StudioError.message("Cannot open native input image.") }
        guard let profile = manifest.profiles.first(where: { $0.contentSizeHW == [image.height, image.width] }) else {
            let available = manifest.profiles.filter(\.validated).map(\.id).joined(separator: ", ")
            throw StudioError.message("No verified Core AI geometry for \(image.width)×\(image.height). Available: \(available). Use local Transformers for other page sizes. The native backend never silently resizes a page.")
        }
        guard !requireValidated || profile.validated else { throw StudioError.message("This Core AI profile has not passed end-to-end validation.") }
        guard profile.imageSizeHW.count == 2, profile.imageSizeHW[0] % 16 == 0, profile.imageSizeHW[1] % 64 == 0, profile.patchCount == (profile.imageSizeHW[0]/16)*(profile.imageSizeHW[1]/16), profile.contentSizeHW[0] <= profile.imageSizeHW[0], profile.contentSizeHW[1] <= profile.imageSizeHW[1] else { throw StudioError.message("Invalid native profile geometry.") }
        guard maxTokens > 0, maxTokens <= profile.cacheCapacity - profile.prefixLength, repetitionPenalty.isFinite, repetitionPenalty > 0 else { throw StudioError.message("This native profile supports at most \(profile.cacheCapacity - profile.prefixLength) generated tokens. Reduce Maximum generation tokens; the cache will never be silently truncated.") }
        let loadStart = Date()
        let value: Loaded
        if let existing = loaded, existing.root == directory, existing.profile.id == profile.id { value = existing }
        else {
            loaded = nil
            func function(_ relative: String) async throws -> InferenceFunction {
                let url = directory.appendingPathComponent(relative).standardizedFileURL
                guard url.path.hasPrefix(directory.standardizedFileURL.path + "/") else { throw StudioError.message("Native asset path escapes the selected model directory.") }
                let model = try await AIModel(contentsOf: url, options: .cpuOnly)
                guard let fn = try model.loadFunction(named: "main") else { throw StudioError.message("Native asset has no main function.") }; return fn
            }
            let encoder = try await function(profile.encoder)
            let prefill = try await function(profile.prefill)
            let step = try await function(profile.step)
            let encoded = try JSONDecoder().decode([String].self, from: Data(contentsOf: directory.appendingPathComponent(manifest.tokenBytes)))
            let vocab = try encoded.map { token -> Data in guard let bytes = Data(base64Encoded: token) else { throw StudioError.message("Invalid native tokenizer bytes.") }; return bytes }
            value = Loaded(root: directory, profile: profile, encoder: encoder, prefill: prefill, step: step, vocabulary: vocab); loaded = value
        }
        let loadSeconds = Date().timeIntervalSince(loadStart)
        let prepareStart = Date()
        let patches = try Self.prepare(image: image, profile: profile)
        if let validationDirectory { try FileManager.default.createDirectory(at: validationDirectory, withIntermediateDirectories: true); try patches.withUnsafeBytes { Data($0) }.write(to: validationDirectory.appendingPathComponent("patches.f32")) }
        let prepareSeconds = Date().timeIntervalSince(prepareStart)
        try Task.checkCancellation()
        let inferStart = Date()
        var encoderOut = try await value.encoder.run(inputs: ["patches": NDArray(scalars: patches, shape: [profile.patchCount, 768])])
        guard let visual = encoderOut.remove("visual")?.ndArray else { throw StudioError.message("Native encoder produced no visual prefix.") }
        if let validationDirectory { try Self.floats(visual).withUnsafeBytes { Data($0) }.write(to: validationDirectory.appendingPathComponent("visual.f32")) }
        var prefillOut = try await value.prefill.run(inputs: ["visual": visual])
        guard var logits = prefillOut.remove("logits")?.ndArray, var keys = prefillOut.remove("keys")?.ndArray, var values = prefillOut.remove("values")?.ndArray else { throw StudioError.message("Native prefill outputs are incomplete.") }
        if let validationDirectory { try Self.floats(logits).withUnsafeBytes { Data($0) }.write(to: validationDirectory.appendingPathComponent("logits.f32")) }
        var ids = profile.promptIDs
        var seen = Set(ids)
        var generated = 0
        var finished = false
        for offset in 0..<maxTokens {
            try Task.checkCancellation()
            var scores = Self.floats(logits)
            for token in seen where token >= 0 && token < scores.count {
                scores[token] = scores[token] < 0 ? scores[token] * Float(repetitionPenalty) : scores[token] / Float(repetitionPenalty)
            }
            guard scores.allSatisfy(\.isFinite), let token = scores.indices.max(by: { scores[$0] < scores[$1] }) else { throw StudioError.message("Native decoder produced invalid logits.") }
            ids.append(token); seen.insert(token); generated += 1
            if token == profile.eosID { finished = true; break }
            if offset + 1 == maxTokens { break }
            var output = try await value.step.run(inputs: [
                "input_ids": NDArray(scalars: [Int32(token)], shape: [1,1]),
                "past_keys": keys, "past_values": values,
                "position": NDArray(scalars: [Int32(profile.prefixLength + offset)], shape: [1])
            ])
            guard let l = output.remove("logits")?.ndArray, let k = output.remove("keys")?.ndArray, let v = output.remove("values")?.ndArray else { throw StudioError.message("Native decoder step outputs are incomplete.") }
            logits = l; keys = k; values = v
        }
        var bytes = Data()
        for id in ids { guard value.vocabulary.indices.contains(id) else { throw StudioError.message("Native decoder returned an unknown token ID.") }; bytes.append(value.vocabulary[id]) }
        let raw = String(decoding: bytes, as: UTF8.self)
        guard finished else {
            let failure: [String: Any] = ["error": "Native generation reached its token limit; partial prediction retained.", "raw": raw, "backend": "coreai", "profile": profile.id]
            throw StudioError.backendFailure("Native generation reached its token limit.", String(decoding: try JSONSerialization.data(withJSONObject: failure), as: UTF8.self))
        }
        let inferenceSeconds = Date().timeIntervalSince(inferStart)
        return NativePrediction(raw: raw, tokenIDs: ids, profile: profile.id, metrics: ["model_load_seconds": loadSeconds, "preprocess_seconds": prepareSeconds, "inference_seconds": inferenceSeconds, "total_seconds": Date().timeIntervalSince(totalStart), "request_count": 1, "generated_tokens": Double(generated), "pages_per_second": 1 / max(0.001, inferenceSeconds)])
    }

    private static func floats(_ array: NDArray) -> [Float] {
        precondition(array.scalarType == .float32)
        let count = array.shape.reduce(1, *)
        // Exported model outputs are dense; gather by actual strides to also
        // support runtime delegates which return a non-contiguous view.
        let dims = array.shape, steps = array.strides
        return array.view(as: Float.self).withUnsafePointer { pointer, _, _ in
            var result = [Float](); result.reserveCapacity(count)
            for flat in 0..<count {
                var rest = flat, offset = 0
                for axis in dims.indices.reversed() { offset += (rest % dims[axis]) * steps[axis]; rest /= dims[axis] }
                result.append(pointer[offset])
            }
            return result
        }
    }

    private static func prepare(image: CGImage, profile: NativeProfile) throws -> [Float] {
        let width = image.width, height = image.height
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height)); return true
        }
        guard drawn else { throw StudioError.message("Cannot allocate native RGB page buffer.") }
        let mean: [Float] = [0.48145466,0.4578275,0.40821073]
        let std: [Float] = [0.26862954,0.26130258,0.27577711]
        var patches = [Float](); patches.reserveCapacity(profile.patchCount * 768)
        for row in stride(from: 0, to: profile.imageSizeHW[0], by: 16) {
            for column in stride(from: 0, to: profile.imageSizeHW[1], by: 16) {
                for channel in 0..<3 { for dy in 0..<16 { for dx in 0..<16 {
                    let x = column + dx, y = row + dy
                    let byte = (x < width && y < height) ? rgba[(y * width + x) * 4 + channel] : 255
                    let unit = Float(byte) / 255
                    patches.append((unit - mean[channel]) / std[channel])
                } } }
            }
        }
        return patches
    }
}

@available(macOS 27.0, *)
public struct NativeCoreAIBackend: InferenceBackend {
    public let directory: URL
    public let postprocessor: ReferenceBackend
    private let engine: NativeCoreAIEngine
    public init(directory: URL, postprocessor: ReferenceBackend, engine: NativeCoreAIEngine) { self.directory = directory; self.postprocessor = postprocessor; self.engine = engine }
    public func parse(imageURL: URL, width: Int, height: Int, settings: InferenceSettings, options: PostprocessOptions, requestID: String) async throws -> ParseResult {
        guard settings.mode == "single", settings.temperature == 0, settings.predict_text_in_pic else { throw StudioError.message("The validated Core AI profiles support single-pass, greedy decoding (temperature 0) with full structure and picture text. Use Transformers for other modes.") }
        let prediction = try await engine.generate(imageURL: imageURL, directory: directory, maxTokens: settings.max_new_tokens, repetitionPenalty: settings.repetition_penalty)
        let input = ParseResult(raw: prediction.raw, mergedRaw: prediction.raw, markdown: "", blocks: [], warnings: [], stages: [StageResponse(name: "fullpage", raw: prediction.raw, finishReason: "stop", cropBBoxPixels: nil)], metrics: prediction.metrics, backend: "coreai", revision: pinnedRevision)
        var result = try await postprocessor.postprocess(result: input, width: width, height: height, options: options)
        result.warnings.append("Native Core AI CPU · geometry \(prediction.profile) · single-pass greedy decoding; NVIDIA postprocessing runs locally in Python.")
        let metadata: [String: JSONValue] = ["backend": .string("native Swift Core AI"), "profile": .string(prediction.profile), "compute": .string("CPU float32"), "raw": .string(prediction.raw), "settings": try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(settings))]
        result.sourceResponse = .object(metadata)
        return result
    }
    public func postprocess(result: ParseResult, width: Int, height: Int, options: PostprocessOptions) async throws -> ParseResult { try await postprocessor.postprocess(result: result, width: width, height: height, options: options) }
}
