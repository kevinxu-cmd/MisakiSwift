// swift-tools-version: 6.2
import PackageDescription

let package = Package(
  name: "MisakiSwift",
  platforms: [
    .iOS(.v18), .macOS(.v15),
  ],
  products: [
    // Static: the app that links this copies one executable into its bundle, and a
    // dynamic product would have to be embedded beside it.
    .library(name: "MisakiSwift", type: .static, targets: ["MisakiSwift"]),
  ],
  targets: [
    .target(
      name: "MisakiSwift",
      resources: [
        .copy("../../MisakiData/")
      ]
    ),
    .testTarget(
      name: "MisakiSwiftTests",
      dependencies: ["MisakiSwift"],
      resources: [
        .copy("Fixtures")
      ]
    ),
  ]
)
