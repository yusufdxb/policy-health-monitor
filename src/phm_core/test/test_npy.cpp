// Copyright 2026 Yusuf Guenena. MIT License.
// .npy/.npz I/O: reads archives written by NumPy's np.savez and
// np.savez_compressed (test/data, made with NumPy 1.26.4), and round-trips
// archives this code writes.
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/npy.hpp"

namespace npy = phm_core::npy;

namespace
{

std::string temp_path()
{
  const char * dir = std::getenv("TMPDIR");
  std::string tmpl = std::string(dir != nullptr && *dir != '\0' ? dir : "/tmp") +
    "/phm_npz_XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  const int fd = mkstemp(buf.data());
  if (fd >= 0) {
    close(fd);
  }
  return buf.data();
}

void write_bytes(const std::string & path, const std::vector<uint8_t> & bytes)
{
  std::ofstream f(path, std::ios::binary);
  f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

// A valid two-member archive written by save_npz, as bytes.
std::vector<uint8_t> valid_archive()
{
  const std::string path = temp_path();
  std::vector<std::pair<std::string, npy::Array>> arrays;
  arrays.emplace_back("threshold", npy::Array::scalar_double(0.5));
  arrays.emplace_back("latents", npy::Array::from_floats({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}));
  npy::save_npz(path, arrays);
  auto bytes = npy::read_file_bytes(path);
  std::remove(path.c_str());
  return bytes;
}

std::size_t find_signature(const std::vector<uint8_t> & b, uint32_t sig)
{
  for (std::size_t i = 0; i + 4 <= b.size(); ++i) {
    if ((b[i] | (b[i + 1] << 8) | (b[i + 2] << 16) | (uint32_t{b[i + 3]} << 24)) == sig) {
      return i;
    }
  }
  return b.size();
}

void put16(std::vector<uint8_t> & b, std::size_t at, uint16_t v)
{
  b[at] = static_cast<uint8_t>(v);
  b[at + 1] = static_cast<uint8_t>(v >> 8);
}

void put32(std::vector<uint8_t> & b, std::size_t at, uint32_t v)
{
  for (int i = 0; i < 4; ++i) {
    b[at + i] = static_cast<uint8_t>(v >> (8 * i));
  }
}

// load_npz on the given bytes must throw std::runtime_error (and, under
// AddressSanitizer, must not read outside the buffer while doing so).
void expect_rejected(const std::vector<uint8_t> & bytes)
{
  const std::string path = temp_path();
  write_bytes(path, bytes);
  EXPECT_THROW(npy::load_npz(path), std::runtime_error);
  std::remove(path.c_str());
}

}  // namespace

TEST(Npz, ReadsNumpySavez)
{
  // np.savez(threshold=0.0123456789, window=30, percentile=1.0,
  //          latents=float32 arange(12).reshape(4, 3) / 8)
  const auto z = npy::load_npz("data/calib_savez.npz");
  EXPECT_EQ(z.at("threshold").scalar(), 0.0123456789);
  EXPECT_EQ(z.at("window").scalar(), 30.0);
  EXPECT_EQ(z.at("window").descr, "<i8");
  EXPECT_EQ(z.at("percentile").scalar(), 1.0);
  const auto & lat = z.at("latents");
  EXPECT_EQ(lat.descr, "<f4");
  EXPECT_EQ(lat.shape, (std::vector<std::size_t>{4, 3}));
  const auto v = lat.as_doubles();
  ASSERT_EQ(v.size(), 12u);
  for (std::size_t i = 0; i < 12; ++i) {
    EXPECT_EQ(v[i], static_cast<double>(static_cast<float>(i) / 8.0f));
  }
}

TEST(Npz, ReadsNumpySavezCompressed)
{
  const auto z = npy::load_npz("data/calib_compressed.npz");
  EXPECT_EQ(z.at("threshold").scalar(), 2.5);
  EXPECT_EQ(z.at("window").scalar(), 20.0);
  EXPECT_EQ(z.at("latents").shape, (std::vector<std::size_t>{4, 3}));
}

TEST(Npz, RoundTripsWhatItWrites)
{
  const std::string path = temp_path();
  std::vector<std::pair<std::string, npy::Array>> out;
  out.emplace_back("threshold", npy::Array::scalar_double(0x1.c6e60a05a04dcp+5));
  out.emplace_back("window", npy::Array::scalar_int64(30));
  out.emplace_back("latents", npy::Array::from_floats({1.5f, -2.0f, 3.25f, 0.0f}, {2, 2}));
  out.emplace_back("one_d", npy::Array::from_doubles({1.0, 2.0, 3.0}, {3}));
  npy::save_npz(path, out);
  const auto z = npy::load_npz(path);
  std::remove(path.c_str());
  EXPECT_EQ(z.at("threshold").scalar(), 0x1.c6e60a05a04dcp+5);
  EXPECT_EQ(z.at("window").scalar(), 30.0);
  EXPECT_EQ(z.at("latents").as_doubles(), (std::vector<double>{1.5, -2.0, 3.25, 0.0}));
  EXPECT_EQ(z.at("one_d").shape, (std::vector<std::size_t>{3}));
}

TEST(Npy, HeaderIsAlignedAndShapeTextMatchesNumpy)
{
  const auto bytes = npy::serialize_npy(npy::Array::from_doubles({1.0, 2.0, 3.0}, {3}));
  const std::size_t header_len = bytes[8] | (bytes[9] << 8);
  EXPECT_EQ((10 + header_len) % 64, 0u);
  const std::string header(bytes.begin() + 10, bytes.begin() + 10 + header_len);
  EXPECT_NE(header.find("'shape': (3,)"), std::string::npos);
  EXPECT_EQ(header.back(), '\n');
  const auto back = npy::parse_npy(bytes.data(), bytes.size());
  EXPECT_EQ(back.as_doubles(), (std::vector<double>{1.0, 2.0, 3.0}));
  EXPECT_THROW(npy::parse_npy(bytes.data(), 5), std::runtime_error);
}

TEST(Npz, RejectsMalformedArchivesWithoutReadingPastThem)
{
  const std::vector<uint8_t> good = valid_archive();
  const std::size_t cd = find_signature(good, 0x02014b50u);
  const std::size_t eocd = find_signature(good, 0x06054b50u);
  ASSERT_LT(cd, good.size());
  ASSERT_LT(eocd, good.size());

  auto b = good;  // central-directory name longer than the archive
  put16(b, cd + 28, 0xffff);
  expect_rejected(b);

  b = good;  // extra field longer than the archive
  put16(b, cd + 30, 0xffff);
  expect_rejected(b);

  b = good;  // local header offset past the end
  put32(b, cd + 42, 0xfffffff0u);
  expect_rejected(b);

  b = good;  // member size past the end
  put32(b, cd + 20, 0x7fffffffu);
  expect_rejected(b);

  b = good;  // central directory offset past the end
  put32(b, eocd + 16, 0xfffffff0u);
  expect_rejected(b);

  b = good;  // more entries than the archive could hold
  put16(b, eocd + 10, 0xfffe);
  put16(b, eocd + 8, 0xfffe);
  expect_rejected(b);

  b = good;  // a ZIP64 marker whose locator points outside the archive
  put32(b, eocd + 16, 0xffffffffu);
  expect_rejected(b);

  b.assign(good.begin() + static_cast<std::ptrdiff_t>(eocd), good.end());  // EOCD alone
  expect_rejected(b);
}

TEST(Npy, RejectsShapesWhoseByteCountOverflows)
{
  const std::string header =
    "{'descr': '<f8', 'fortran_order': False, 'shape': (4294967296, 4294967296, 16), }";
  std::vector<uint8_t> bytes = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
    static_cast<uint8_t>(header.size()), 0};
  bytes.insert(bytes.end(), header.begin(), header.end());
  bytes.resize(bytes.size() + 64, 0);
  EXPECT_THROW(npy::parse_npy(bytes.data(), bytes.size()), std::runtime_error);

  const std::string bad_dtype = "{'descr': '<fx', 'fortran_order': False, 'shape': (2,), }";
  std::vector<uint8_t> b2 = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0,
    static_cast<uint8_t>(bad_dtype.size()), 0};
  b2.insert(b2.end(), bad_dtype.begin(), bad_dtype.end());
  b2.resize(b2.size() + 64, 0);
  EXPECT_THROW(npy::parse_npy(b2.data(), b2.size()), std::runtime_error);
}
