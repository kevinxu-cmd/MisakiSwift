import Foundation

/// Reads the fallback network's weights. The safetensors layout is eight bytes of
/// little-endian header length, a JSON header naming each tensor's dtype, shape and byte
/// range, then the tensor bytes back to back. Only float32 is stored here, and only
/// float32 is read.
enum Safetensors {
  struct Tensor: Sendable {
    let shape: [Int]
    let data: [Float]
  }

  enum Error: Swift.Error, Equatable {
    case truncated
    case badHeader
    case unsupportedDType(String)
    case badOffsets(String)
  }

  private struct Entry: Decodable {
    let dtype: String
    let shape: [Int]
    let data_offsets: [Int]
  }

  static func load(_ url: URL) throws -> [String: Tensor] {
    let bytes = try Data(contentsOf: url)
    guard bytes.count >= 8 else { throw Error.truncated }
    let headerLength = bytes.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self).littleEndian) }
    guard bytes.count >= 8 + headerLength else { throw Error.truncated }
    let headerData = bytes.subdata(in: 8..<(8 + headerLength))
    guard let raw = try? JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
      throw Error.badHeader
    }
    let body = bytes.subdata(in: (8 + headerLength)..<bytes.count)
    var out: [String: Tensor] = [:]
    for (name, value) in raw where name != "__metadata__" {
      let entryData = try JSONSerialization.data(withJSONObject: value)
      guard let entry = try? JSONDecoder().decode(Entry.self, from: entryData) else { throw Error.badHeader }
      guard entry.dtype == "F32" else { throw Error.unsupportedDType(entry.dtype) }
      guard entry.data_offsets.count == 2 else { throw Error.badOffsets(name) }
      let start = entry.data_offsets[0]
      let end = entry.data_offsets[1]
      let count = entry.shape.reduce(1, *)
      guard start >= 0, end >= start, end <= body.count, end - start == count * 4 else {
        throw Error.badOffsets(name)
      }
      let data = body.subdata(in: start..<end).withUnsafeBytes { buffer -> [Float] in
        var floats = [Float](repeating: 0, count: count)
        for i in 0..<count {
          floats[i] = Float(bitPattern: buffer.loadUnaligned(fromByteOffset: i * 4, as: UInt32.self).littleEndian)
        }
        return floats
      }
      out[name] = Tensor(shape: entry.shape, data: data)
    }
    return out
  }

  /// The bundled weight file for one dialect, "us_bart" or "gb_bart".
  static func bundledWeights(named name: String) -> URL? {
    Bundle.module.url(forResource: name, withExtension: "safetensors", subdirectory: "MisakiData")
  }

  /// The bundled network config for one dialect, "us_bart_config" or "gb_bart_config".
  static func bundledConfig(named name: String) -> URL? {
    Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "MisakiData")
  }
}
