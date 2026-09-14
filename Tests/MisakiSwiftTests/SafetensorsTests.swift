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

  /// Writes a hand-built file: eight bytes of little-endian header length, the header, the body.
  private func file(header: String, body: Data, named name: String) throws -> URL {
    let headerBytes = Data(header.utf8)
    var bytes = Data()
    withUnsafeBytes(of: UInt64(headerBytes.count).littleEndian) { bytes.append(contentsOf: $0) }
    bytes.append(headerBytes)
    bytes.append(body)
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(name)
    try bytes.write(to: url)
    return url
  }

  @Test func rejectsAHeaderThatIsNotJSON() throws {
    let url = try file(header: "not json at all", body: Data([0, 0, 0, 0]), named: "not-json.safetensors")
    #expect(throws: Safetensors.Error.badHeader) { try Safetensors.load(url) }
  }

  @Test func rejectsADTypeOtherThanF32() throws {
    let header = #"{"t":{"dtype":"F16","shape":[1,1],"data_offsets":[0,2]}}"#
    let url = try file(header: header, body: Data([0, 0]), named: "half-precision.safetensors")
    #expect(throws: Safetensors.Error.unsupportedDType("F16")) { try Safetensors.load(url) }
  }

  /// A shape whose product runs past `Int.max` is rejected, not multiplied into a trap.
  @Test func rejectsAShapeThatOverflows() throws {
    let header = #"{"t":{"dtype":"F32","shape":[4611686018427387904,4],"data_offsets":[0,4]}}"#
    let url = try file(header: header, body: Data([0, 0, 0, 0]), named: "overflowing-shape.safetensors")
    #expect(throws: Safetensors.Error.self) { try Safetensors.load(url) }
  }

  @Test func rejectsOffsetsPastTheBody() throws {
    let header = #"{"t":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]}}"#
    let url = try file(header: header, body: Data([0, 0, 0, 0]), named: "short-body.safetensors")
    #expect(throws: Safetensors.Error.badOffsets("t")) { try Safetensors.load(url) }
  }
}
