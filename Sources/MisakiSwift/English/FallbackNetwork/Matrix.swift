import Accelerate
import Foundation

/// A row-major float matrix with the few operations the fallback network needs. Every
/// operation returns a new matrix; the network is small enough that copying is nothing.
struct Matrix: Sendable {
  let rows: Int
  let cols: Int
  var data: [Float]

  init(rows: Int, cols: Int) {
    self.rows = rows
    self.cols = cols
    self.data = [Float](repeating: 0, count: rows * cols)
  }

  init(rows: Int, cols: Int, data: [Float]) {
    precondition(data.count == rows * cols, "Matrix \(rows)x\(cols) given \(data.count) values")
    self.rows = rows
    self.cols = cols
    self.data = data
  }

  enum ShapeError: Error { case notTwoDimensional([Int]) }

  init(_ tensor: Safetensors.Tensor) throws {
    guard tensor.shape.count == 2 else { throw ShapeError.notTwoDimensional(tensor.shape) }
    self.init(rows: tensor.shape[0], cols: tensor.shape[1], data: tensor.data)
  }

  subscript(row: Int, col: Int) -> Float {
    data[row * cols + col]
  }

  func row(_ i: Int) -> [Float] {
    Array(data[(i * cols)..<((i + 1) * cols)])
  }

  func columns(from: Int, count: Int) -> Matrix {
    precondition(from >= 0 && from + count <= cols)
    var out = Matrix(rows: rows, cols: count)
    for r in 0..<rows {
      for c in 0..<count { out.data[r * count + c] = data[r * cols + from + c] }
    }
    return out
  }

  mutating func setColumns(from: Int, _ block: Matrix) {
    precondition(block.rows == rows)
    precondition(from >= 0 && from + block.cols <= cols)
    for r in 0..<rows {
      for c in 0..<block.cols { data[r * cols + from + c] = block.data[r * block.cols + c] }
    }
  }

  /// `self` [n,k] times `b` [k,m], via `vDSP_mmul` on the row-major buffers directly.
  func matmul(_ b: Matrix) -> Matrix {
    precondition(cols == b.rows)
    var out = Matrix(rows: rows, cols: b.cols)
    data.withUnsafeBufferPointer { a in
      b.data.withUnsafeBufferPointer { bp in
        out.data.withUnsafeMutableBufferPointer { c in
          vDSP_mmul(
            a.baseAddress!, 1, bp.baseAddress!, 1, c.baseAddress!, 1,
            vDSP_Length(rows), vDSP_Length(b.cols), vDSP_Length(cols))
        }
      }
    }
    return out
  }

  /// `self` [n,d] times the transpose of `b` [m,d], giving [n,m]. This is both a linear
  /// layer against a `[out,in]` weight and the query-key product of attention. `b` is
  /// transposed into a scratch buffer with `vDSP_mtrans` first, since `vDSP_mmul` has no
  /// transposed-operand option the way `cblas_sgemm` did.
  func matmulTransposed(_ b: Matrix) -> Matrix {
    precondition(cols == b.cols)
    var transposed = [Float](repeating: 0, count: b.cols * b.rows)
    b.data.withUnsafeBufferPointer { bp in
      transposed.withUnsafeMutableBufferPointer { t in
        vDSP_mtrans(bp.baseAddress!, 1, t.baseAddress!, 1, vDSP_Length(b.cols), vDSP_Length(b.rows))
      }
    }
    var out = Matrix(rows: rows, cols: b.rows)
    data.withUnsafeBufferPointer { a in
      transposed.withUnsafeBufferPointer { t in
        out.data.withUnsafeMutableBufferPointer { c in
          vDSP_mmul(
            a.baseAddress!, 1, t.baseAddress!, 1, c.baseAddress!, 1,
            vDSP_Length(rows), vDSP_Length(b.rows), vDSP_Length(cols))
        }
      }
    }
    return out
  }

  func adding(_ b: Matrix) -> Matrix {
    precondition(rows == b.rows && cols == b.cols)
    return Matrix(rows: rows, cols: cols, data: vDSP.add(data, b.data))
  }

  func adding(bias: [Float]) -> Matrix {
    precondition(bias.count == cols)
    var out = self
    for r in 0..<rows {
      for c in 0..<cols { out.data[r * cols + c] += bias[c] }
    }
    return out
  }

  func scaled(by s: Float) -> Matrix {
    Matrix(rows: rows, cols: cols, data: vDSP.multiply(s, data))
  }

  func softmaxRows() -> Matrix {
    var out = self
    for r in 0..<rows {
      let range = (r * cols)..<((r + 1) * cols)
      let top = out.data[range].max() ?? 0
      var sum: Float = 0
      for i in range {
        let e = expf(out.data[i] - top)
        out.data[i] = e
        sum += e
      }
      for i in range { out.data[i] /= sum }
    }
    return out
  }

  func layerNormRows(weight: [Float], bias: [Float], eps: Float) -> Matrix {
    precondition(weight.count == cols && bias.count == cols)
    var out = self
    for r in 0..<rows {
      let range = (r * cols)..<((r + 1) * cols)
      var mean: Float = 0
      var meanOfSquares: Float = 0
      vDSP_measqv(Array(data[range]), 1, &meanOfSquares, vDSP_Length(cols))
      vDSP_meanv(Array(data[range]), 1, &mean, vDSP_Length(cols))
      let variance = max(meanOfSquares - mean * mean, 0)
      let inv = 1 / (variance + eps).squareRoot()
      for (j, i) in range.enumerated() {
        out.data[i] = (data[i] - mean) * inv * weight[j] + bias[j]
      }
    }
    return out
  }

  /// The exact gelu, `x * (1 + erf(x / sqrt 2)) / 2`, the form MLXNN and Transformers use here.
  func gelu() -> Matrix {
    Matrix(rows: rows, cols: cols, data: data.map { $0 * (1 + erff($0 / Float(2).squareRoot())) / 2 })
  }
}
