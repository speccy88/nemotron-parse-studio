import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import PDFKit

public struct ImportedDocument: Identifiable, Codable, Sendable {
    public var id: UUID
    /// A private, unchanged copy of the imported file. No persistent access to the source is required.
    public var originalURL: URL
    public var workingURL: URL
    public var name: String
    public var pageCount: Int
    public var kind: String
    public var fileSize: Int64
    public init(id: UUID = UUID(), originalURL: URL, workingURL: URL, name: String, pageCount: Int, kind: String, fileSize: Int64) {
        self.id = id; self.originalURL = originalURL; self.workingURL = workingURL
        self.name = name; self.pageCount = pageCount; self.kind = kind; self.fileSize = fileSize
    }
}

public struct RenderedPage: Codable, Sendable {
    public var imageURL: URL
    public var thumbnailURL: URL
    public var width: Int
    public var height: Int
    /// Displayed document dimensions in PDF points, including page rotation.
    public var documentWidth: Double
    public var documentHeight: Double
    public var rotation: Int
    public var renderSeconds: Double
    public init(imageURL: URL, thumbnailURL: URL, width: Int, height: Int, documentWidth: Double, documentHeight: Double, rotation: Int, renderSeconds: Double) {
        self.imageURL = imageURL; self.thumbnailURL = thumbnailURL; self.width = width; self.height = height
        self.documentWidth = documentWidth; self.documentHeight = documentHeight; self.rotation = rotation; self.renderSeconds = renderSeconds
    }
}

public enum DocumentIOError: LocalizedError {
    case unsupported(String), unreadable(String), invalidPage, resolutionTooLarge, converterMissing, conversionFailed(String), unsafeHTML
    public var errorDescription: String? {
        switch self {
        case .unsupported(let ext): "Unsupported document format: \(ext). Import a PDF or an ImageIO-supported image."
        case .unreadable(let message): message
        case .invalidPage: "The requested page does not exist."
        case .resolutionTooLarge: "This rendering would exceed 64 million pixels. Reduce the rendering DPI."
        case .converterMissing: "This file requires LibreOffice. Install LibreOffice in Applications, or set NEMOTRON_SOFFICE to its executable."
        case .conversionFailed(let message): "Local LibreOffice conversion failed: \(message)"
        case .unsafeHTML: "HTML containing scripts, embedded objects, styles, frames, or external resources cannot be safely converted. Export it to PDF first."
        }
    }
}

/// Loads metadata and renders exactly one page at a time, away from the main actor.
public actor DocumentLoader {
    public let storageURL: URL
    private let fileManager = FileManager.default
    private var renderCache: [String: RenderedPage] = [:]

    public init(storageURL: URL? = nil) {
        self.storageURL = storageURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Nemotron Parse Studio/Imports", isDirectory: true)
    }

    public func load(url: URL) async throws -> ImportedDocument {
        try Task.checkCancellation()
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        let ext = url.pathExtension.lowercased()
        let id = UUID()
        let directory = storageURL.appendingPathComponent(id.uuidString, isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let original = directory.appendingPathComponent(url.lastPathComponent)
        do {
            try fileManager.copyItem(at: url, to: original)
            let attributes = try fileManager.attributesOfItem(atPath: original.path)
            let fileSize = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            let office: Set<String> = ["docx", "pptx", "xlsx", "rtf", "html", "htm", "odt", "odp", "ods"]
            var working = original
            if office.contains(ext) {
                if ext == "html" || ext == "htm" { try validateLocalHTML(original) }
                working = try await convertToPDF(original, in: directory)
            }
            try Task.checkCancellation()
            let count: Int
            let kind: String
            if working.pathExtension.lowercased() == "pdf" {
                guard let pdf = CGPDFDocument(working as CFURL) else { throw DocumentIOError.unreadable("The PDF cannot be opened.") }
                guard !pdf.isEncrypted || pdf.isUnlocked else { throw DocumentIOError.unreadable("This PDF is password protected. Save an unlocked copy before importing.") }
                count = pdf.numberOfPages; kind = ext == "pdf" ? "PDF" : "\(ext.uppercased()) → PDF"
            } else if let source = CGImageSourceCreateWithURL(working as CFURL, nil) {
                count = CGImageSourceGetCount(source); kind = ext.uppercased()
            } else { throw DocumentIOError.unsupported(ext) }
            guard count > 0 else { throw DocumentIOError.unreadable("The document has no readable pages.") }
            return ImportedDocument(id: id, originalURL: original, workingURL: working, name: url.lastPathComponent, pageCount: count, kind: kind, fileSize: fileSize)
        } catch {
            try? fileManager.removeItem(at: directory)
            throw error
        }
    }

    public func render(document: ImportedDocument, page: Int, dpi: Double) async throws -> RenderedPage {
        try Task.checkCancellation()
        guard page >= 0, page < document.pageCount else { throw DocumentIOError.invalidPage }
        guard dpi.isFinite, dpi >= 36, dpi <= 600 else { throw DocumentIOError.unreadable("Rendering DPI must be between 36 and 600.") }
        let dpiKey = Int((dpi * 1000).rounded())
        let key = "\(document.id.uuidString)-\(page)-\(dpiKey)"
        if let cached = renderCache[key], fileManager.fileExists(atPath: cached.imageURL.path) { return cached }
        let started = Date()
        let image: CGImage
        let logicalWidth: Double, logicalHeight: Double
        let rotation: Int
        if document.workingURL.pathExtension.lowercased() == "pdf" {
            guard let pdf = CGPDFDocument(document.workingURL as CFURL), let pdfPage = pdf.page(at: page + 1) else { throw DocumentIOError.invalidPage }
            let bounds = pdfPage.getBoxRect(.cropBox)
            rotation = (Int(pdfPage.rotationAngle) % 360 + 360) % 360
            let swapped = rotation == 90 || rotation == 270
            logicalWidth = swapped ? bounds.height : bounds.width
            logicalHeight = swapped ? bounds.width : bounds.height
            let width = max(1, Int(ceil(logicalWidth * dpi / 72)))
            let height = max(1, Int(ceil(logicalHeight * dpi / 72)))
            let context = try Self.rgbContext(width: width, height: height)
            let rect = CGRect(x: 0, y: 0, width: width, height: height)
            context.concatenate(pdfPage.getDrawingTransform(.cropBox, rect: rect, rotate: 0, preserveAspectRatio: true))
            context.drawPDFPage(pdfPage)
            guard let rendered = context.makeImage() else { throw DocumentIOError.unreadable("The PDF page could not be rendered.") }
            image = rendered
        } else {
            guard let source = CGImageSourceCreateWithURL(document.workingURL as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
                  let properties = CGImageSourceCopyPropertiesAtIndex(source, page, nil) as? [CFString: Any],
                  let width = (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue,
                  let height = (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue else { throw DocumentIOError.unreadable("This image frame cannot be decoded.") }
            guard width > 0, height > 0, Int64(width) * Int64(height) <= 64_000_000 else { throw DocumentIOError.resolutionTooLarge }
            // ImageIO applies EXIF orientation before inference and preview. Native pixel resolution is preserved.
            let options: [CFString: Any] = [kCGImageSourceCreateThumbnailFromImageAlways: true, kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: max(width, height), kCGImageSourceShouldCacheImmediately: true]
            guard let decoded = CGImageSourceCreateThumbnailAtIndex(source, page, options as CFDictionary) else { throw DocumentIOError.unreadable("The image cannot be decoded.") }
            let context = try Self.rgbContext(width: decoded.width, height: decoded.height)
            context.draw(decoded, in: CGRect(x: 0, y: 0, width: decoded.width, height: decoded.height))
            guard let rendered = context.makeImage() else { throw DocumentIOError.unreadable("The RGB image could not be created.") }
            image = rendered; rotation = 0
            let imageDPI = (properties[kCGImagePropertyDPIWidth] as? NSNumber)?.doubleValue ?? 72
            logicalWidth = Double(image.width) * 72 / max(1, imageDPI)
            logicalHeight = Double(image.height) * 72 / max(1, imageDPI)
        }
        try Task.checkCancellation()
        let directory = document.originalURL.deletingLastPathComponent().appendingPathComponent("Rendered", isDirectory: true)
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let output = directory.appendingPathComponent("page-\(page + 1)-\(dpiKey).png")
        let thumbURL = directory.appendingPathComponent("page-\(page + 1)-\(dpiKey)-thumb.png")
        try Self.writePNG(image, to: output)
        let scale = min(1, 180.0 / Double(max(image.width, image.height)))
        let thumbContext = try Self.rgbContext(width: max(1, Int(Double(image.width) * scale)), height: max(1, Int(Double(image.height) * scale)))
        thumbContext.interpolationQuality = .high
        thumbContext.draw(image, in: CGRect(x: 0, y: 0, width: thumbContext.width, height: thumbContext.height))
        if let thumb = thumbContext.makeImage() { try Self.writePNG(thumb, to: thumbURL) }
        let result = RenderedPage(imageURL: output, thumbnailURL: thumbURL, width: image.width, height: image.height,
                                  documentWidth: logicalWidth, documentHeight: logicalHeight, rotation: rotation, renderSeconds: Date().timeIntervalSince(started))
        renderCache[key] = result
        return result
    }

    private static func rgbContext(width: Int, height: Int) throws -> CGContext {
        guard width > 0, height > 0, Int64(width) * Int64(height) <= 64_000_000 else { throw DocumentIOError.resolutionTooLarge }
        guard let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { throw DocumentIOError.unreadable("Not enough memory to render this page. Reduce the DPI.") }
        context.setFillColor(CGColor(gray: 1, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context
    }

    private static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw DocumentIOError.unreadable("Cannot create the page image.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw DocumentIOError.unreadable("Cannot save the page image.") }
    }

    private func validateLocalHTML(_ url: URL) throws {
        let html = try String(contentsOf: url, encoding: .utf8)
        // Only a deliberately restricted passive HTML subset is accepted by the local converter.
        let forbidden = #"(?is)<\s*(script|style|iframe|object|embed|link|base|meta|img|video|audio|svg|math)\b|\bon\w+\s*=|\b(?:src|href|background|style)\s*=|url\s*\("#
        if html.range(of: forbidden, options: .regularExpression) != nil { throw DocumentIOError.unsafeHTML }
    }

    static func officeConverterExecutable() -> String? {
        let fileManager = FileManager.default
        let bundledRuntime = fileManager.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/codex-runtimes/codex-primary-runtime/dependencies/bin/override/soffice")
        let candidates = [
            ProcessInfo.processInfo.environment["NEMOTRON_SOFFICE"],
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "/Applications/LibreOfficeDev.app/Contents/MacOS/soffice",
            "/opt/homebrew/bin/soffice",
            "/usr/local/bin/soffice",
            bundledRuntime.path
        ].compactMap { $0 }
        return candidates.first(where: { fileManager.isExecutableFile(atPath: $0) })
    }

    private func convertToPDF(_ input: URL, in directory: URL) async throws -> URL {
        guard let executable = Self.officeConverterExecutable() else { throw DocumentIOError.converterMissing }
        let output = directory.appendingPathComponent("Converted", isDirectory: true)
        let profile = directory.appendingPathComponent("ConverterProfile", isDirectory: true)
        try fileManager.createDirectory(at: output, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: profile.appendingPathComponent("user"), withIntermediateDirectories: true)
        // A fresh profile disables macros and link updates; no user LibreOffice profile is modified.
        let policy = #"<?xml version="1.0" encoding="UTF-8"?><oor:items xmlns:oor="http://openoffice.org/2001/registry"><item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item><item oor:path="/org.openoffice.Office.Writer/Content/Update"><prop oor:name="Link" oor:op="fuse"><value>2</value></prop></item></oor:items>"#
        try policy.write(to: profile.appendingPathComponent("user/registrymodifications.xcu"), atomically: true, encoding: .utf8)
        defer { try? fileManager.removeItem(at: profile) }
        let process = Process()
        let arguments = ["-env:UserInstallation=\(profile.absoluteString)", "--headless", "--nologo", "--nodefault", "--nofirststartwizard", "--convert-to", "pdf", "--outdir", output.path, input.path]
        // The system sandbox prevents embedded external links from reaching the network.
        if fileManager.isExecutableFile(atPath: "/usr/bin/sandbox-exec") {
            process.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            process.arguments = ["-p", "(version 1)(allow default)(deny network*)", executable] + arguments
        } else {
            process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        }
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        let box = ConverterProcess(process)
        let timeout = Task {
            try? await Task.sleep(for: .seconds(120))
            if !Task.isCancelled { box.terminate() }
        }
        defer { timeout.cancel() }
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
                process.terminationHandler = { completed in
                    if completed.terminationStatus == 0 { continuation.resume() }
                    else { continuation.resume(throwing: DocumentIOError.conversionFailed("converter exited with status \(completed.terminationStatus).")) }
                }
                do { try Task.checkCancellation(); try box.run() } catch { continuation.resume(throwing: error) }
            }
        } onCancel: { box.terminate() }
        try Task.checkCancellation()
        let pdf = output.appendingPathComponent(input.deletingPathExtension().lastPathComponent).appendingPathExtension("pdf")
        guard fileManager.fileExists(atPath: pdf.path), CGPDFDocument(pdf as CFURL) != nil else { throw DocumentIOError.conversionFailed("no readable PDF was produced. The format may not be supported by this LibreOffice installation.") }
        return pdf
    }
}

private final class ConverterProcess: @unchecked Sendable {
    private let process: Process
    private let lock = NSLock()
    private var cancelled = false
    init(_ process: Process) { self.process = process }
    func run() throws {
        lock.lock(); defer { lock.unlock() }
        if cancelled { throw CancellationError() }
        try process.run()
    }
    func terminate() {
        lock.lock(); defer { lock.unlock() }
        cancelled = true
        if process.isRunning { process.terminate() }
    }
}
