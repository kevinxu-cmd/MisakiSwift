import Foundation

/// The fallback grapheme-to-phoneme network: a one-layer BART encoder and decoder with a
/// tied output head, run greedily. It reproduces the PyTorch model the weights were
/// trained as, and the MLX port that preceded this file, token for token.
struct BARTNetwork: Sendable {
  enum WeightError: Error { case missing(String) }

  struct Dense: Sendable {
    let weight: Matrix  // [out, in]
    let bias: [Float]
    func callAsFunction(_ x: Matrix) -> Matrix {
      x.matmulTransposed(weight).adding(bias: bias)
    }
  }

  struct Norm: Sendable {
    let weight: [Float]
    let bias: [Float]
    static let eps: Float = 1e-5
    func callAsFunction(_ x: Matrix) -> Matrix {
      x.layerNormRows(weight: weight, bias: bias, eps: Self.eps)
    }
  }

  struct Attention: Sendable {
    let q: Dense
    let k: Dense
    let v: Dense
    let out: Dense
    let heads: Int

    func callAsFunction(_ query: Matrix, keyValue: Matrix) -> Matrix {
      let qp = q(query)
      let kp = k(keyValue)
      let vp = v(keyValue)
      let dim = qp.cols / heads
      let scale = 1 / Float(dim).squareRoot()
      var merged = Matrix(rows: qp.rows, cols: qp.cols)
      for h in 0..<heads {
        let qh = qp.columns(from: h * dim, count: dim)
        let kh = kp.columns(from: h * dim, count: dim)
        let vh = vp.columns(from: h * dim, count: dim)
        let weights = qh.matmulTransposed(kh).scaled(by: scale).softmaxRows()
        merged.setColumns(from: h * dim, weights.matmul(vh))
      }
      return out(merged)
    }
  }

  struct Layer: Sendable {
    let selfAttention: Attention
    let selfNorm: Norm
    let crossAttention: Attention?
    let crossNorm: Norm?
    let fc1: Dense
    let fc2: Dense
    let finalNorm: Norm

    func callAsFunction(_ x: Matrix, encoder: Matrix?) -> Matrix {
      var h = selfNorm(x.adding(selfAttention(x, keyValue: x)))
      if let crossAttention, let crossNorm, let encoder {
        h = crossNorm(h.adding(crossAttention(h, keyValue: encoder)))
      }
      return finalNorm(h.adding(fc2(fc1(h).gelu())))
    }
  }

  let config: BARTConfig
  let shared: Matrix
  let encoderPositions: Matrix
  let decoderPositions: Matrix
  let encoderNorm: Norm
  let decoderNorm: Norm
  let encoderLayers: [Layer]
  let decoderLayers: [Layer]
  let logitBias: [Float]

  init(config: BARTConfig, tensors: [String: Safetensors.Tensor]) throws {
    func tensor(_ name: String) throws -> Safetensors.Tensor {
      guard let t = tensors[name] else { throw WeightError.missing(name) }
      return t
    }
    func matrix(_ name: String) throws -> Matrix { try Matrix(try tensor(name)) }
    func vector(_ name: String) throws -> [Float] { try tensor(name).data }
    func dense(_ prefix: String) throws -> Dense {
      Dense(weight: try matrix(prefix + ".weight"), bias: try vector(prefix + ".bias"))
    }
    func norm(_ prefix: String) throws -> Norm {
      Norm(weight: try vector(prefix + ".weight"), bias: try vector(prefix + ".bias"))
    }
    func attention(_ prefix: String, heads: Int) throws -> Attention {
      Attention(
        q: try dense(prefix + ".q_proj"), k: try dense(prefix + ".k_proj"),
        v: try dense(prefix + ".v_proj"), out: try dense(prefix + ".out_proj"), heads: heads)
    }
    func layer(_ prefix: String, heads: Int, cross: Bool) throws -> Layer {
      Layer(
        selfAttention: try attention(prefix + ".self_attn", heads: heads),
        selfNorm: try norm(prefix + ".self_attn_layer_norm"),
        crossAttention: cross ? try attention(prefix + ".encoder_attn", heads: heads) : nil,
        crossNorm: cross ? try norm(prefix + ".encoder_attn_layer_norm") : nil,
        fc1: try dense(prefix + ".fc1"),
        fc2: try dense(prefix + ".fc2"),
        finalNorm: try norm(prefix + ".final_layer_norm"))
    }

    self.config = config
    self.shared = try matrix("model.shared.weight")
    self.encoderPositions = try matrix("model.encoder.embed_positions.weight")
    self.decoderPositions = try matrix("model.decoder.embed_positions.weight")
    self.encoderNorm = try norm("model.encoder.layernorm_embedding")
    self.decoderNorm = try norm("model.decoder.layernorm_embedding")
    self.encoderLayers = try (0..<config.encoderLayers).map {
      try layer("model.encoder.layers.\($0)", heads: config.encoderAttentionHeads, cross: false)
    }
    self.decoderLayers = try (0..<config.decoderLayers).map {
      try layer("model.decoder.layers.\($0)", heads: config.decoderAttentionHeads, cross: true)
    }
    self.logitBias = try vector("final_logits_bias")
    precondition(
      config.decoderLayers == 1,
      "BARTNetwork runs the decoder without a causal mask, which is exact only for one decoder layer")
  }

  /// BART's position table starts two rows in.
  private static let positionOffset = 2

  private func embed(_ ids: [Int], positions table: Matrix, norm: Norm) -> Matrix {
    var h = Matrix(rows: ids.count, cols: shared.cols)
    for (i, id) in ids.enumerated() {
      let token = shared.row(id)
      let position = table.row(i + Self.positionOffset)
      for c in 0..<shared.cols { h.data[i * shared.cols + c] = token[c] + position[c] }
    }
    return norm(h)
  }

  func encode(_ ids: [Int]) -> Matrix {
    var h = embed(ids, positions: encoderPositions, norm: encoderNorm)
    for layer in encoderLayers { h = layer(h, encoder: nil) }
    return h
  }

  /// Logits over the vocabulary for the last decoder position.
  func decodeLast(_ ids: [Int], encoder: Matrix) -> [Float] {
    var h = embed(ids, positions: decoderPositions, norm: decoderNorm)
    // No causal mask is applied, so every row attends to every row; exact only because
    // `init` guarantees a single decoder layer.
    for layer in decoderLayers { h = layer(h, encoder: encoder) }
    let last = Matrix(rows: 1, cols: h.cols, data: h.row(h.rows - 1))
    return last.matmulTransposed(shared).adding(bias: logitBias).row(0)
  }

  /// Greedy decoding, the way the MLX port did it: at most 49 generated tokens, stopping at
  /// EOS, ties to the lowest index. The model re-emits BOS as its first id, as trained BART
  /// checkpoints do; callers drop ids of 3 and below, as the PyTorch reference does.
  func generate(_ inputIds: [Int]) -> [Int] {
    let encoder = encode(inputIds)
    var decoded = [config.bosTokenId]
    var out: [Int] = []
    let maxLength = 50
    for i in 0..<maxLength {
      if i == maxLength - 1 { break }
      let logits = decodeLast(decoded, encoder: encoder)
      var best = 0
      for (j, v) in logits.enumerated() where v > logits[best] { best = j }
      if best == config.eosTokenId { break }
      out.append(best)
      decoded.append(best)
    }
    return out
  }
}
