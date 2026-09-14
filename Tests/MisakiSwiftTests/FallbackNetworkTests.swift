import Foundation
import Testing

@testable import MisakiSwift

@Suite struct MatrixTests {
  @Test func matmulMatchesByHand() {
    let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
    let b = Matrix(rows: 3, cols: 2, data: [7, 8, 9, 10, 11, 12])
    let c = a.matmul(b)
    #expect(c.rows == 2 && c.cols == 2)
    #expect(c.data == [58, 64, 139, 154])
  }

  @Test func matmulTransposedIsAgainstRows() {
    let a = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 4, 5, 6])
    let b = Matrix(rows: 2, cols: 3, data: [1, 0, 0, 0, 1, 0])
    let c = a.matmulTransposed(b)
    #expect(c.data == [1, 2, 4, 5])
  }

  @Test func softmaxRowsSumToOne() {
    let m = Matrix(rows: 2, cols: 3, data: [1, 2, 3, 1000, 1000, 1000]).softmaxRows()
    #expect(abs(m.row(0).reduce(0, +) - 1) < 1e-5)
    #expect(m.row(1).allSatisfy { abs($0 - 1.0 / 3) < 1e-5 })
    #expect(m[0, 2] > m[0, 1] && m[0, 1] > m[0, 0])
  }

  @Test func layerNormZeroMeansUnitVariance() {
    let m = Matrix(rows: 1, cols: 4, data: [1, 2, 3, 4])
      .layerNormRows(weight: [1, 1, 1, 1], bias: [0, 0, 0, 0], eps: 0)
    let r = m.row(0)
    #expect(abs(r.reduce(0, +)) < 1e-5)
    let variance = r.map { $0 * $0 }.reduce(0, +) / 4
    #expect(abs(variance - 1) < 1e-4)
  }

  @Test func layerNormAppliesWeightBiasAndEps() {
    // mean 2.5, population variance 1.25, so 1 / sqrt(1.25 + 1) is 2 / 3.
    let m = Matrix(rows: 1, cols: 4, data: [1, 2, 3, 4])
      .layerNormRows(weight: [2, 0.5, 1, 3], bias: [1, -1, 0.5, 0], eps: 1)
    let expected: [Float] = [-1, -7.0 / 6, 5.0 / 6, 3]
    for (i, want) in expected.enumerated() { #expect(abs(m[0, i] - want) < 1e-5, "column \(i)") }
  }

  @Test func geluIsTheExactForm() {
    let m = Matrix(rows: 1, cols: 3, data: [-1, 0, 1]).gelu()
    #expect(abs(m[0, 0] - (-0.15865526)) < 1e-6)
    #expect(m[0, 1] == 0)
    #expect(abs(m[0, 2] - 0.84134474) < 1e-6)
  }

  @Test func columnsRoundTrip() {
    let m = Matrix(rows: 2, cols: 4, data: [1, 2, 3, 4, 5, 6, 7, 8])
    let block = m.columns(from: 1, count: 2)
    #expect(block.data == [2, 3, 6, 7])
    var target = Matrix(rows: 2, cols: 4)
    target.setColumns(from: 1, block)
    #expect(target.data == [0, 2, 3, 0, 0, 6, 7, 0])
  }
}

@Suite struct BARTNetworkTests {
  @Test func generatesSomethingAndStops() throws {
    let url = try #require(Safetensors.bundledWeights(named: "us_bart"))
    let configURL = try #require(Safetensors.bundledConfig(named: "us_bart_config"))
    let config = try JSONDecoder().decode(BARTConfig.self, from: Data(contentsOf: configURL))
    let net = try BARTNetwork(config: config, tensors: Safetensors.load(url))
    // "b", "l", "o", "r", "p" spelled through grapheme_chars, wrapped in BOS and EOS.
    let graphemes = Array(config.graphemeChars)
    let ids = [config.bosTokenId] + "blorp".map { graphemes.firstIndex(of: $0)! } + [config.eosTokenId]
    let out = net.generate(ids)
    #expect(!out.isEmpty)
    #expect(out.count < 50)
    #expect(!out.contains(config.eosTokenId))
    // The model re-emits BOS as its first id; the caller drops ids of 3 and below, as the
    // PyTorch reference does.
    #expect(out.allSatisfy { $0 < config.vocabSize })
    #expect(out.contains { $0 > 3 })
  }
}

@Suite struct FallbackGoldenTests {
  private func golden() throws -> [String: [String: String]] {
    let url = try #require(Bundle.module.url(forResource: "fallback-golden", withExtension: "json", subdirectory: "Fixtures"))
    return try JSONDecoder().decode([String: [String: String]].self, from: Data(contentsOf: url))
  }

  private func token(_ word: String) -> MToken {
    MToken(text: word, tokenRange: word.startIndex..<word.endIndex, whitespace: "")
  }

  @Test func americanMatchesThePyTorchModel() throws {
    let expected = try #require(try golden()["us"])
    #expect(expected.count == 32)
    let net = EnglishFallbackNetwork(british: false)
    for (word, phonemes) in expected.sorted(by: { $0.key < $1.key }) {
      #expect(net(token(word)).phoneme == phonemes, "us \(word)")
    }
  }

  @Test func britishMatchesThePyTorchModel() throws {
    let expected = try #require(try golden()["gb"])
    #expect(expected.count == 32)
    let net = EnglishFallbackNetwork(british: true)
    for (word, phonemes) in expected.sorted(by: { $0.key < $1.key }) {
      #expect(net(token(word)).phoneme == phonemes, "gb \(word)")
    }
  }

  @Test func ratingIsOne() {
    #expect(EnglishFallbackNetwork(british: false)(token("blorptastic")).rating == 1)
  }

  /// A character outside `grapheme_chars` becomes the unknown token rather than a crash,
  /// and the network still pronounces the rest of the word.
  @Test func unknownGraphemesMapToTheUnknownToken() {
    #expect(!EnglishFallbackNetwork(british: false)(token("café")).phoneme.isEmpty)
  }
}
