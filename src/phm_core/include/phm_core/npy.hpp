// Copyright 2026 Yusuf Guenena. MIT License.
// Reading and writing NumPy .npy arrays and .npz archives.
//
// Reading accepts format versions 1-3, little-endian numeric dtypes and both
// stored and deflate-compressed archive members (np.savez and
// np.savez_compressed), including ZIP64 records. Writing produces a stored
// (uncompressed) archive that np.load reads, the format np.savez writes. This
// keeps calibration files interchangeable between this code and NumPy.
#ifndef PHM_CORE__NPY_HPP_
#define PHM_CORE__NPY_HPP_

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace phm_core
{
namespace npy
{

struct Array
{
  std::string descr;                 // NumPy dtype string, e.g. "<f8", "<f4", "<i8"
  std::vector<std::size_t> shape;    // empty for a scalar
  bool fortran_order = false;
  std::vector<uint8_t> bytes;        // raw little-endian element data

  std::size_t size() const;          // number of elements (1 for a scalar)
  std::size_t itemsize() const;
  // Elements converted to double, in storage order. Throws for non-numeric
  // dtypes.
  std::vector<double> as_doubles() const;
  // The single element of a size-1 array as a double.
  double scalar() const;

  static Array from_doubles(const std::vector<double> & values, std::vector<std::size_t> shape);
  static Array from_floats(const std::vector<float> & values, std::vector<std::size_t> shape);
  static Array scalar_double(double value);
  static Array scalar_int64(int64_t value);
};

// Parse one .npy image. Throws std::runtime_error on malformed input.
Array parse_npy(const uint8_t * data, std::size_t size);
// Serialize as a version 1.0 .npy image (64-byte aligned header).
std::vector<uint8_t> serialize_npy(const Array & array);

Array load_npy(const std::string & path);
// Members of an .npz archive keyed by name without the ".npy" suffix.
std::map<std::string, Array> load_npz(const std::string & path);
// Write an uncompressed .npz archive (np.savez layout); member order is kept.
void save_npz(const std::string & path, const std::vector<std::pair<std::string, Array>> & arrays);

// Whole-file helpers shared by the tools (gzip input is decompressed).
std::vector<uint8_t> read_file_bytes(const std::string & path);
std::string gunzip(const std::vector<uint8_t> & compressed);

}  // namespace npy
}  // namespace phm_core

#endif  // PHM_CORE__NPY_HPP_
