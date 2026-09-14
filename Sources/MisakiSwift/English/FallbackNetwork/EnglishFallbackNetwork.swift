import Foundation

/// Pronounces a word the dictionaries did not know, with the small BART network shipped
/// beside them. Loaded once per dialect; the network is a value and the class only caches it.
final class EnglishFallbackNetwork {
  static let unknownTokenId = 3

  private let configuration: BARTConfig
  private let network: BARTNetwork
  private let graphemeToToken: [Character: Int]
  private let tokenToPhoneme: [Int: Character]

  init(british: Bool) {
    let prefix = british ? "gb" : "us"
    guard let configURL = Safetensors.bundledConfig(named: "\(prefix)_bart_config"),
      let configData = try? Data(contentsOf: configURL),
      let configuration = try? JSONDecoder().decode(BARTConfig.self, from: configData)
    else {
      fatalError("MisakiSwift: the bundled \(prefix)_bart_config.json is missing or unreadable")
    }
    guard let weightsURL = Safetensors.bundledWeights(named: "\(prefix)_bart"),
      let tensors = try? Safetensors.load(weightsURL),
      let network = try? BARTNetwork(config: configuration, tensors: tensors)
    else {
      fatalError("MisakiSwift: the bundled \(prefix)_bart.safetensors is missing or unreadable")
    }
    self.configuration = configuration
    self.network = network

    var graphemes: [Character: Int] = [:]
    for (index, grapheme) in configuration.graphemeChars.enumerated() { graphemes[grapheme] = index }
    self.graphemeToToken = graphemes

    var phonemes: [Int: Character] = [:]
    for (index, phoneme) in configuration.phonemeChars.enumerated() { phonemes[index] = phoneme }
    self.tokenToPhoneme = phonemes
  }

  private func graphemesToTokens(_ graphemes: String) -> [Int] {
    var tokens = [configuration.bosTokenId]
    for char in graphemes {
      tokens.append(graphemeToToken[char] ?? Self.unknownTokenId)
    }
    tokens.append(configuration.eosTokenId)
    return tokens
  }

  private func tokensToPhonemes(_ tokens: [Int]) -> String {
    var phonemes = ""
    for token in tokens where token > Self.unknownTokenId {
      if let phoneme = tokenToPhoneme[token] { phonemes.append(phoneme) }
    }
    return phonemes
  }

  func callAsFunction(_ word: MToken) -> (phoneme: String, rating: Int) {
    let generated = network.generate(graphemesToTokens(word.text))
    return (tokensToPhonemes(generated), 1)
  }
}
