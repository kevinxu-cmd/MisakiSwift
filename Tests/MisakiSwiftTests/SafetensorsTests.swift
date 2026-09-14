import Foundation
import Testing

@testable import MisakiSwift

@Suite struct SafetensorsTests {
  /// The test target's `Bundle.module` holds `Fixtures`, not the library's `MisakiData`, so
  /// the weights are reached through the library's own accessor.
  private func resource(_ name: String) throws -> URL {
    try #require(Safetensors.bundledWeights(named: name))
  }

  @Test func readsEveryTensorWithItsShape() throws {
    let tensors = try Safetensors.load(resource("us_bart"))
    #expect(tensors.count == 50)
    #expect(tensors["model.shared.weight"]?.shape == [63, 128])
    #expect(tensors["model.shared.weight"]?.data.count == 63 * 128)
    #expect(tensors["model.encoder.embed_positions.weight"]?.shape == [66, 128])
    #expect(tensors["final_logits_bias"]?.shape == [1, 63])
    #expect(tensors["model.decoder.layers.0.fc1.weight"]?.shape == [1024, 128])
    #expect(tensors["__metadata__"] == nil)
  }

  @Test func bothDialectsShareOneLayout() throws {
    let us = try Safetensors.load(resource("us_bart"))
    let gb = try Safetensors.load(resource("gb_bart"))
    #expect(Set(us.keys) == Set(gb.keys))
    for (name, t) in us { #expect(gb[name]?.shape == t.shape, "\(name)") }
  }

  @Test func valuesAreFinite() throws {
    let tensors = try Safetensors.load(resource("gb_bart"))
    for (name, t) in tensors { #expect(t.data.allSatisfy { $0.isFinite }, "\(name)") }
  }

  @Test func rejectsATruncatedFile() throws {
    let whole = try Data(contentsOf: resource("us_bart"))
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("truncated.safetensors")
    try whole.prefix(whole.count - 100).write(to: tmp)
    #expect(throws: Safetensors.Error.self) { try Safetensors.load(tmp) }
  }

  @Test func rejectsAnOversizedHeaderLength() throws {
    let bytes = Data(repeating: 0xFF, count: 16)
    let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("oversized-header.safetensors")
    try bytes.write(to: tmp)
    #expect(throws: Safetensors.Error.self) { try Safetensors.load(tmp) }
  }
}
