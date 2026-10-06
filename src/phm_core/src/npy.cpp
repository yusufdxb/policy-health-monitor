// Copyright 2026 Yusuf Guenena. MIT License.
#include "phm_core/npy.hpp"

#include <zlib.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phm_core
{
namespace npy
{
namespace
{

[[noreturn]] void fail(const std::string & what)
{
  throw std::runtime_error("npy: " + what);
}

// True when [offset, offset + length) lies inside [0, limit), without
// overflowing. Every read of untrusted archive or header data is checked with
// this before the bytes are touched.
bool fits(uint64_t offset, uint64_t length, uint64_t limit)
{
  return offset <= limit && length <= limit - offset;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char * what)
{
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) {
    fail(std::string(what) + " overflows");
  }
  return a * b;
}

uint16_t rd16(const uint8_t * p) {return static_cast<uint16_t>(p[0] | (p[1] << 8));}
uint32_t rd32(const uint8_t * p)
{
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
         (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t rd64(const uint8_t * p)
{
  return static_cast<uint64_t>(rd32(p)) | (static_cast<uint64_t>(rd32(p + 4)) << 32);
}
void wr16(std::vector<uint8_t> & o, uint16_t v)
{
  o.push_back(static_cast<uint8_t>(v & 0xff));
  o.push_back(static_cast<uint8_t>(v >> 8));
}
void wr32(std::vector<uint8_t> & o, uint32_t v)
{
  for (int i = 0; i < 4; ++i) {
    o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
  }
}

// Value of `key` in a Python dict literal header, as raw text.
std::string header_field(const std::string & header, const std::string & key)
{
  const std::string quoted1 = "'" + key + "'";
  const std::string quoted2 = "\"" + key + "\"";
  std::size_t pos = header.find(quoted1);
  std::size_t len = quoted1.size();
  if (pos == std::string::npos) {
    pos = header.find(quoted2);
    len = quoted2.size();
  }
  if (pos == std::string::npos) {
    fail("header lacks '" + key + "'");
  }
  pos = header.find(':', pos + len);
  if (pos == std::string::npos) {
    fail("malformed header");
  }
  ++pos;
  while (pos < header.size() && header[pos] == ' ') {
    ++pos;
  }
  if (pos < header.size() && header[pos] == '(') {
    const std::size_t end = header.find(')', pos);
    if (end == std::string::npos) {
      fail("malformed shape");
    }
    return header.substr(pos, end - pos + 1);
  }
  if (pos < header.size() && (header[pos] == '\'' || header[pos] == '"')) {
    const char q = header[pos];
    const std::size_t end = header.find(q, pos + 1);
    if (end == std::string::npos) {
      fail("malformed string field");
    }
    return header.substr(pos + 1, end - pos - 1);
  }
  std::size_t end = pos;
  while (end < header.size() && header[end] != ',' && header[end] != '}') {
    ++end;
  }
  return header.substr(pos, end - pos);
}

std::vector<std::size_t> parse_shape(const std::string & text)
{
  std::vector<std::size_t> shape;
  std::string num;
  for (char c : text) {
    if (c >= '0' && c <= '9') {
      num += c;
    } else if (!num.empty()) {
      if (num.size() > 19) {
        fail("shape dimension too large");
      }
      shape.push_back(static_cast<std::size_t>(std::stoull(num)));
      num.clear();
    }
  }
  return shape;
}

std::string shape_text(const std::vector<std::size_t> & shape)
{
  std::string s = "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (shape.size() == 1 || i + 1 < shape.size()) {
      s += ",";
    }
    if (i + 1 < shape.size()) {
      s += " ";
    }
  }
  return s + ")";
}

std::vector<uint8_t> inflate_raw(const uint8_t * data, std::size_t size, std::size_t out_size)
{
  if (size > std::numeric_limits<uInt>::max() || out_size > std::numeric_limits<uInt>::max()) {
    fail("deflate member too large");
  }
  std::vector<uint8_t> out(out_size);
  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) {
    fail("inflateInit2 failed");
  }
  zs.next_in = const_cast<Bytef *>(data);
  zs.avail_in = static_cast<uInt>(size);
  zs.next_out = out.data();
  zs.avail_out = static_cast<uInt>(out.size());
  const int rc = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (rc != Z_STREAM_END || zs.total_out != out_size) {
    fail("deflate member did not decompress cleanly");
  }
  return out;
}

}  // namespace

std::size_t Array::size() const
{
  std::size_t n = 1;
  for (std::size_t d : shape) {
    n = checked_mul(n, d, "array element count");
  }
  return n;
}

std::size_t Array::itemsize() const
{
  if (descr.size() < 3 || descr.size() > 4 ||
    descr.find_first_not_of("0123456789", 2) != std::string::npos)
  {
    fail("unsupported dtype '" + descr + "'");
  }
  const std::size_t width = static_cast<std::size_t>(std::stoul(descr.substr(2)));
  if (width == 0) {
    fail("unsupported dtype '" + descr + "'");
  }
  return width;
}

std::vector<double> Array::as_doubles() const
{
  const std::size_t n = size();
  if (descr.size() < 3 || (descr[0] != '<' && descr[0] != '|')) {
    fail("unsupported byte order in dtype '" + descr + "'");
  }
  const char kind = descr[1];
  const std::size_t width = itemsize();
  if (bytes.size() / width < n) {
    fail("array data shorter than its shape");
  }
  std::vector<double> out(n);
  const uint8_t * p = bytes.data();
  for (std::size_t i = 0; i < n; ++i, p += width) {
    if (kind == 'f' && width == 8) {
      double v;
      std::memcpy(&v, p, 8);
      out[i] = v;
    } else if (kind == 'f' && width == 4) {
      float v;
      std::memcpy(&v, p, 4);
      out[i] = static_cast<double>(v);
    } else if (kind == 'i' && width == 8) {
      out[i] = static_cast<double>(static_cast<int64_t>(rd64(p)));
    } else if (kind == 'i' && width == 4) {
      out[i] = static_cast<double>(static_cast<int32_t>(rd32(p)));
    } else if (kind == 'i' && width == 2) {
      out[i] = static_cast<double>(static_cast<int16_t>(rd16(p)));
    } else if (kind == 'i' && width == 1) {
      out[i] = static_cast<double>(static_cast<int8_t>(p[0]));
    } else if (kind == 'u' && width == 8) {
      out[i] = static_cast<double>(rd64(p));
    } else if (kind == 'u' && width == 4) {
      out[i] = static_cast<double>(rd32(p));
    } else if (kind == 'u' && width == 2) {
      out[i] = static_cast<double>(rd16(p));
    } else if ((kind == 'u' || kind == 'b') && width == 1) {
      out[i] = static_cast<double>(p[0]);
    } else {
      fail("unsupported dtype '" + descr + "'");
    }
  }
  return out;
}

double Array::scalar() const
{
  if (size() != 1) {
    fail("array is not a scalar");
  }
  return as_doubles()[0];
}

Array Array::from_doubles(const std::vector<double> & values, std::vector<std::size_t> shape)
{
  Array a;
  a.descr = "<f8";
  a.shape = std::move(shape);
  if (a.size() != values.size()) {
    fail("from_doubles: shape does not match value count");
  }
  a.bytes.resize(values.size() * 8);
  if (!values.empty()) {
    std::memcpy(a.bytes.data(), values.data(), a.bytes.size());
  }
  return a;
}

Array Array::from_floats(const std::vector<float> & values, std::vector<std::size_t> shape)
{
  Array a;
  a.descr = "<f4";
  a.shape = std::move(shape);
  if (a.size() != values.size()) {
    fail("from_floats: shape does not match value count");
  }
  a.bytes.resize(values.size() * 4);
  if (!values.empty()) {
    std::memcpy(a.bytes.data(), values.data(), a.bytes.size());
  }
  return a;
}

Array Array::scalar_double(double value)
{
  return from_doubles({value}, {});
}

Array Array::scalar_int64(int64_t value)
{
  Array a;
  a.descr = "<i8";
  a.bytes.resize(8);
  std::memcpy(a.bytes.data(), &value, 8);
  return a;
}

Array parse_npy(const uint8_t * data, std::size_t size)
{
  static const char kMagic[] = "\x93NUMPY";
  if (size < 10 || std::memcmp(data, kMagic, 6) != 0) {
    fail("bad magic");
  }
  const uint8_t major = data[6];
  std::size_t header_len;
  std::size_t offset;
  if (major == 1) {
    header_len = rd16(data + 8);
    offset = 10;
  } else if (major == 2 || major == 3) {
    if (size < 12) {
      fail("truncated header");
    }
    header_len = rd32(data + 8);
    offset = 12;
  } else {
    fail("unsupported format version " + std::to_string(major));
  }
  if (!fits(offset, header_len, size)) {
    fail("truncated header");
  }
  const std::string header(reinterpret_cast<const char *>(data + offset), header_len);
  Array a;
  a.descr = header_field(header, "descr");
  a.fortran_order = header_field(header, "fortran_order").find("True") != std::string::npos;
  a.shape = parse_shape(header_field(header, "shape"));
  const std::size_t start = offset + header_len;
  const std::size_t nbytes = checked_mul(a.size(), a.itemsize(), "array byte count");
  if (!fits(start, nbytes, size)) {
    fail("truncated data");
  }
  a.bytes.assign(data + start, data + start + nbytes);
  return a;
}

std::vector<uint8_t> serialize_npy(const Array & array)
{
  std::string header = "{'descr': '" + array.descr + "', 'fortran_order': " +
    (array.fortran_order ? "True" : "False") + ", 'shape': " + shape_text(array.shape) + ", }";
  // Pad with spaces so magic + length + header + '\n' is a multiple of 64.
  const std::size_t base = 10 + header.size() + 1;
  header.append((64 - base % 64) % 64, ' ');
  header += '\n';
  std::vector<uint8_t> out;
  out.reserve(10 + header.size() + array.bytes.size());
  const char magic[] = "\x93NUMPY";
  out.insert(out.end(), magic, magic + 6);
  out.push_back(1);
  out.push_back(0);
  wr16(out, static_cast<uint16_t>(header.size()));
  out.insert(out.end(), header.begin(), header.end());
  out.insert(out.end(), array.bytes.begin(), array.bytes.end());
  return out;
}

std::vector<uint8_t> read_file_bytes(const std::string & path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string gunzip(const std::vector<uint8_t> & compressed)
{
  z_stream zs{};
  if (inflateInit2(&zs, 16 + MAX_WBITS) != Z_OK) {
    fail("inflateInit2 failed");
  }
  zs.next_in = const_cast<Bytef *>(compressed.data());
  zs.avail_in = static_cast<uInt>(compressed.size());
  std::string out;
  char buf[1 << 16];
  int rc;
  do {
    zs.next_out = reinterpret_cast<Bytef *>(buf);
    zs.avail_out = sizeof(buf);
    rc = inflate(&zs, Z_NO_FLUSH);
    if (rc != Z_OK && rc != Z_STREAM_END) {
      inflateEnd(&zs);
      fail("gzip data is corrupt");
    }
    out.append(buf, sizeof(buf) - zs.avail_out);
  } while (rc != Z_STREAM_END && zs.avail_in > 0);
  inflateEnd(&zs);
  if (rc != Z_STREAM_END) {
    fail("gzip data is truncated");
  }
  return out;
}

Array load_npy(const std::string & path)
{
  const std::vector<uint8_t> bytes = read_file_bytes(path);
  return parse_npy(bytes.data(), bytes.size());
}

std::map<std::string, Array> load_npz(const std::string & path)
{
  const std::vector<uint8_t> buf = read_file_bytes(path);
  const std::size_t n = buf.size();
  if (n < 22) {
    fail(path + " is not a zip archive");
  }
  // End of central directory: scan back over a possible comment.
  std::size_t eocd = n;
  for (std::size_t i = n - 22 + 1; i-- > 0 && n - i <= 22 + 65535; ) {
    if (rd32(buf.data() + i) == 0x06054b50u) {
      eocd = i;
      break;
    }
  }
  if (eocd == n) {
    fail(path + " has no end-of-central-directory record");
  }
  uint64_t entries = rd16(buf.data() + eocd + 10);
  uint64_t cd_offset = rd32(buf.data() + eocd + 16);
  if ((entries == 0xffff || cd_offset == 0xffffffffu) && eocd >= 20 &&
    rd32(buf.data() + eocd - 20) == 0x07064b50u)
  {
    const uint64_t z64 = rd64(buf.data() + eocd - 20 + 8);
    if (!fits(z64, 56, n) || rd32(buf.data() + z64) != 0x06064b50u) {
      fail("bad ZIP64 end-of-central-directory record");
    }
    entries = rd64(buf.data() + z64 + 32);
    cd_offset = rd64(buf.data() + z64 + 48);
  }
  // Each central-directory entry takes at least 46 bytes, so a larger count
  // cannot be genuine.
  if (entries > n / 46) {
    fail("bad central directory entry count");
  }
  std::map<std::string, Array> out;
  uint64_t p = cd_offset;
  for (uint64_t e = 0; e < entries; ++e) {
    if (!fits(p, 46, n) || rd32(buf.data() + p) != 0x02014b50u) {
      fail("bad central directory entry");
    }
    const uint16_t method = rd16(buf.data() + p + 10);
    uint64_t comp = rd32(buf.data() + p + 20);
    uint64_t uncomp = rd32(buf.data() + p + 24);
    const uint16_t name_len = rd16(buf.data() + p + 28);
    const uint16_t extra_len = rd16(buf.data() + p + 30);
    const uint16_t comment_len = rd16(buf.data() + p + 32);
    uint64_t local = rd32(buf.data() + p + 42);
    if (!fits(p + 46, uint64_t{name_len} + extra_len + comment_len, n)) {
      fail("central directory entry runs past the end of the archive");
    }
    std::string name(reinterpret_cast<const char *>(buf.data() + p + 46), name_len);
    // ZIP64 extended information extra field (id 0x0001).
    uint64_t x = p + 46 + name_len;
    const uint64_t x_end = x + extra_len;
    while (x_end - x >= 4) {
      const uint16_t id = rd16(buf.data() + x);
      const uint16_t len = rd16(buf.data() + x + 2);
      if (len > x_end - x - 4) {
        fail("bad extra field for " + name);
      }
      if (id == 0x0001) {
        uint64_t q = x + 4;
        const uint64_t q_end = q + len;
        const auto next64 = [&]() {
            if (q_end - q < 8) {
              fail("short ZIP64 extra field for " + name);
            }
            const uint64_t v = rd64(buf.data() + q);
            q += 8;
            return v;
          };
        if (uncomp == 0xffffffffu) {
          uncomp = next64();
        }
        if (comp == 0xffffffffu) {
          comp = next64();
        }
        if (local == 0xffffffffu) {
          local = next64();
        }
      }
      x += 4 + uint64_t{len};
    }
    p = x_end + comment_len;
    if (!fits(local, 30, n) || rd32(buf.data() + local) != 0x04034b50u) {
      fail("bad local file header for " + name);
    }
    const uint64_t data_start = local + 30 +
      uint64_t{rd16(buf.data() + local + 26)} + rd16(buf.data() + local + 28);
    if (!fits(data_start, comp, n)) {
      fail("truncated member " + name);
    }
    std::vector<uint8_t> member;
    if (method == 0) {
      member.assign(buf.begin() + data_start, buf.begin() + data_start + comp);
    } else if (method == 8) {
      // Deflate cannot expand more than about 1032:1; a larger declared size
      // is corrupt, and refusing it avoids a huge allocation.
      if (uncomp > comp * 1032 + 1024) {
        fail("implausible uncompressed size for " + name);
      }
      member = inflate_raw(buf.data() + data_start, comp, uncomp);
    } else {
      fail("unsupported compression method " + std::to_string(method) + " for " + name);
    }
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) {
      name.resize(name.size() - 4);
    }
    out[name] = parse_npy(member.data(), member.size());
  }
  return out;
}

void save_npz(const std::string & path, const std::vector<std::pair<std::string, Array>> & arrays)
{
  std::vector<uint8_t> out;
  std::vector<uint8_t> cd;
  // DOS time/date fixed at 1980-01-01 00:00 so identical inputs give
  // identical files.
  const uint16_t dos_time = 0;
  const uint16_t dos_date = (0 << 9) | (1 << 5) | 1;
  for (const auto & [key, array] : arrays) {
    const std::string name = key + ".npy";
    const std::vector<uint8_t> payload = serialize_npy(array);
    if (payload.size() >= 0xffffffffu || out.size() >= 0xffffffffu) {
      fail("archive too large for the non-ZIP64 writer");
    }
    const uint32_t crc = static_cast<uint32_t>(
      crc32(0L, payload.data(), static_cast<uInt>(payload.size())));
    const uint32_t offset = static_cast<uint32_t>(out.size());
    const uint32_t size = static_cast<uint32_t>(payload.size());
    // Local file header.
    wr32(out, 0x04034b50u);
    wr16(out, 20);
    wr16(out, 0);
    wr16(out, 0);
    wr16(out, dos_time);
    wr16(out, dos_date);
    wr32(out, crc);
    wr32(out, size);
    wr32(out, size);
    wr16(out, static_cast<uint16_t>(name.size()));
    wr16(out, 0);
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), payload.begin(), payload.end());
    // Central directory entry.
    wr32(cd, 0x02014b50u);
    wr16(cd, 20);
    wr16(cd, 20);
    wr16(cd, 0);
    wr16(cd, 0);
    wr16(cd, dos_time);
    wr16(cd, dos_date);
    wr32(cd, crc);
    wr32(cd, size);
    wr32(cd, size);
    wr16(cd, static_cast<uint16_t>(name.size()));
    wr16(cd, 0);
    wr16(cd, 0);
    wr16(cd, 0);
    wr16(cd, 0);
    wr32(cd, 0);
    wr32(cd, offset);
    cd.insert(cd.end(), name.begin(), name.end());
  }
  const uint32_t cd_offset = static_cast<uint32_t>(out.size());
  out.insert(out.end(), cd.begin(), cd.end());
  wr32(out, 0x06054b50u);
  wr16(out, 0);
  wr16(out, 0);
  wr16(out, static_cast<uint16_t>(arrays.size()));
  wr16(out, static_cast<uint16_t>(arrays.size()));
  wr32(out, static_cast<uint32_t>(cd.size()));
  wr32(out, cd_offset);
  wr16(out, 0);
  std::ofstream f(path, std::ios::binary | std::ios::trunc);
  if (!f) {
    throw std::runtime_error("cannot write " + path);
  }
  f.write(reinterpret_cast<const char *>(out.data()), static_cast<std::streamsize>(out.size()));
  if (!f) {
    throw std::runtime_error("write failed for " + path);
  }
}

}  // namespace npy
}  // namespace phm_core
