import Foundation

/// Stand-in while the Accelerate network is built. Replaced in Task 6.
final class EnglishFallbackNetwork {
  static let unknownTokenId = 3
  init(british: Bool) {}
  func callAsFunction(_ word: MToken) -> (phoneme: String, rating: Int) { ("", 1) }
}
