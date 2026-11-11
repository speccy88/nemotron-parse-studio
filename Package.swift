// swift-tools-version: 6.0
import PackageDescription
let package = Package(
    name: "NemotronParseStudio",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "NativeValidate", targets: ["NativeValidate"]), .library(name: "ParseCore", targets: ["ParseCore"]), .executable(name: "NemotronParseStudio", targets: ["ParseStudio"])],
    targets: [.target(name: "ParseCore"), .executableTarget(name: "NativeValidate", dependencies: ["ParseCore"]), .executableTarget(name: "ParseStudio", dependencies: ["ParseCore"]), .testTarget(name: "ParseCoreTests", dependencies: ["ParseCore"])],
    swiftLanguageModes: [.v5]
)
