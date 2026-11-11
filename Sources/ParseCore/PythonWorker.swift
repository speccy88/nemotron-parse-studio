import Foundation

/// One long-lived child process. stdin and stdout are anonymous private pipes;
/// this transport neither binds nor connects a network socket.
public actor PythonWorker {
    private let root: URL
    private let logURL: URL
    private var child: Process?
    private var input: FileHandle?
    private var reader: Task<Void, Never>?
    private var buffer = Data()
    private var pending: [String: CheckedContinuation<Data, Error>] = [:]
    private var generation = UUID()
    public init(projectRoot: URL, logURL: URL? = nil) {
        root = projectRoot
        self.logURL = logURL ?? FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0].appendingPathComponent("NemotronParseStudio/worker.log")
    }
    public func request(method: String, body: Data? = nil) async throws -> Data {
        try Task.checkCancellation()
        try startIfNeeded()
        let id = UUID().uuidString
        let params = try body.map { try JSONSerialization.jsonObject(with: $0) } ?? [:]
        let jobID = (params as? [String: Any])?["request_id"] as? String
        let frame = try JSONSerialization.data(withJSONObject: ["id": id, "method": method, "params": params])
        guard frame.count <= 64 * 1024 * 1024 else { throw StudioError.message("Rendered page exceeds the worker's 64 MB request limit. Reduce DPI.") }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { continuation in
                pending[id] = continuation
                do { try input?.write(contentsOf: frame + Data([10])) }
                catch { pending.removeValue(forKey: id)?.resume(throwing: error) }
            }
        } onCancel: {
            Task { await self.cancelPending(id: id, jobID: jobID) }
        }
    }
    private func cancelPending(id: String, jobID: String?) {
        guard let continuation = pending.removeValue(forKey: id) else { return }
        continuation.resume(throwing: CancellationError())
        if let jobID, let frame = try? JSONSerialization.data(withJSONObject: ["id": UUID().uuidString, "method": "cancel", "params": ["request_id": jobID]]) { try? input?.write(contentsOf: frame + Data([10])) }
    }
    private func startIfNeeded() throws {
        if let child, child.isRunning { return }
        let candidates = [".venv-model/bin/python", "Backend/.venv/bin/python", ".venv-backend/bin/python"]
        guard let executable = candidates.map({ root.appendingPathComponent($0) }).first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) else { throw StudioError.message("Python worker dependencies are missing. Run Backend/setup.sh --model in the project folder.") }
        let script = root.appendingPathComponent("Backend/studio_worker.py")
        guard FileManager.default.fileExists(atPath: script.path) else { throw StudioError.message("Cannot find Backend/studio_worker.py. Keep the app with its source workspace or rebuild from the new location.") }
        let process = Process(), stdinPipe = Pipe(), stdoutPipe = Pipe()
        process.executableURL = executable; process.arguments = [script.path, "--stdio"]; process.currentDirectoryURL = root
        process.standardInput = stdinPipe; process.standardOutput = stdoutPipe
        try FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        if !FileManager.default.fileExists(atPath: logURL.path) { FileManager.default.createFile(atPath: logURL.path, contents: nil) }
        let log = try FileHandle(forWritingTo: logURL); _ = try log.seekToEnd(); process.standardError = log
        var env = ProcessInfo.processInfo.environment; env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"; env["PYTHONUNBUFFERED"] = "1"; process.environment = env
        let current = UUID(); generation = current; buffer = Data()
        process.terminationHandler = { [weak self] process in Task { await self?.ended(generation: current, status: process.terminationStatus) } }
        try process.run()
        child = process; input = stdinPipe.fileHandleForWriting
        let output = stdoutPipe.fileHandleForReading
        reader = Task.detached { [weak self] in
            while !Task.isCancelled {
                let data = output.availableData
                if data.isEmpty { break }
                await self?.receive(data, generation: current)
            }
            try? output.close()
            await self?.ended(generation: current, status: nil)
        }
    }
    private func receive(_ data: Data, generation expected: UUID) {
        guard expected == generation else { return }
        buffer.append(data)
        guard buffer.count <= 128 * 1024 * 1024 else { failAll(StudioError.message("Python worker response exceeded 128 MB.")); stop(); return }
        while let newline = buffer.firstIndex(of: 10) {
            let frame = Data(buffer[..<newline]); buffer.removeSubrange(...newline)
            do {
                guard let envelope = try JSONSerialization.jsonObject(with: frame) as? [String: Any], let id = envelope["id"] as? String else { throw StudioError.message("Invalid worker protocol response.") }
                guard let continuation = pending.removeValue(forKey: id) else { continue }
                if let error = envelope["error"] as? [String: Any] {
                    let payload = try JSONSerialization.data(withJSONObject: error)
                    continuation.resume(throwing: StudioError.backendFailure(error["message"] as? String ?? "Python worker failed.", String(decoding: payload, as: UTF8.self)))
                } else if let result = envelope["result"] {
                    continuation.resume(returning: try JSONSerialization.data(withJSONObject: result, options: .fragmentsAllowed))
                } else { continuation.resume(throwing: StudioError.message("Worker response has no result.")) }
            } catch { failAll(error); stop(); return }
        }
    }
    private func ended(generation expected: UUID, status: Int32?) {
        guard expected == generation else { return }
        // EOF is observed after all complete frames; termination can race their
        // delivery, so only EOF (status nil) fails unfinished requests.
        if status == nil {
            failAll(StudioError.message("Python worker exited. Retry to start a fresh worker. Operational log: \(logURL.path)"))
            try? input?.close(); input = nil; child = nil; reader = nil
        }
    }
    private func failAll(_ error: Error) { let waiting = pending.values; pending = [:]; for continuation in waiting { continuation.resume(throwing: error) } }
    public func stop() {
        generation = UUID(); failAll(CancellationError())
        try? input?.close(); input = nil
        if child?.isRunning == true { child?.terminate() }
        child = nil; reader?.cancel(); reader = nil; buffer = Data()
    }
}
