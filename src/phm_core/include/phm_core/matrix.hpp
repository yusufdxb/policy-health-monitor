// Copyright 2026 Yusuf Guenena. MIT License.
// Minimal row-major float64 matrix used by the offline (calibration, sim,
// benchmark) paths. The runtime OOD path never allocates one per frame.
#ifndef PHM_CORE__MATRIX_HPP_
#define PHM_CORE__MATRIX_HPP_

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace phm_core
{

struct Matrix
{
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> data;  // rows * cols, row-major

  Matrix() = default;
  Matrix(std::size_t r, std::size_t c, double fill = 0.0)
  : rows(r), cols(c), data(r * c, fill) {}

  double * row(std::size_t r) {return data.data() + r * cols;}
  const double * row(std::size_t r) const {return data.data() + r * cols;}
  double & operator()(std::size_t r, std::size_t c) {return data[r * cols + c];}
  double operator()(std::size_t r, std::size_t c) const {return data[r * cols + c];}

  // Rows [begin, end) as a new matrix.
  Matrix slice_rows(std::size_t begin, std::size_t end) const
  {
    if (begin > end || end > rows) {
      throw std::out_of_range("Matrix::slice_rows");
    }
    Matrix out(end - begin, cols);
    for (std::size_t i = 0; i < out.data.size(); ++i) {
      out.data[i] = data[begin * cols + i];
    }
    return out;
  }
};

// Stack matrices with equal column counts (np.concatenate(..., axis=0)).
inline Matrix vstack(const std::vector<const Matrix *> & parts)
{
  Matrix out;
  if (parts.empty()) {
    return out;
  }
  out.cols = parts.front()->cols;
  for (const Matrix * m : parts) {
    if (m->cols != out.cols) {
      throw std::invalid_argument("vstack: column counts differ");
    }
    out.rows += m->rows;
    out.data.insert(out.data.end(), m->data.begin(), m->data.end());
  }
  return out;
}

}  // namespace phm_core

#endif  // PHM_CORE__MATRIX_HPP_
