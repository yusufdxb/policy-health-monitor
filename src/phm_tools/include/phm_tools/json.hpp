// Copyright 2026 Yusuf Guenena. MIT License.
// Minimal JSON value, parser and writer for the offline tools.
//
// The writer reproduces Python's json.dumps byte-for-byte for the values the
// tools produce: integers and floats are distinct (1 vs 1.0), floats use the
// shortest round-trip repr ("1e-05", "1e+16"), NaN / Infinity are written as
// Python writes them, non-ASCII text is \u-escaped, and object keys keep
// insertion order. indent < 0 gives the compact form ", " / ": "; indent >= 0
// gives the pretty form with that many spaces per level.
//
// The parser accepts standard JSON plus NaN / Infinity / -Infinity (as Python's
// json module does). Header-only.
#ifndef PHM_TOOLS__JSON_HPP_
#define PHM_TOOLS__JSON_HPP_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "phm_core/format.hpp"

namespace phm_tools
{
namespace json
{

class Value
{
public:
  enum class Type { kNull, kBool, kInt, kDouble, kString, kArray, kObject };
  using Array = std::vector<Value>;
  using Object = std::vector<std::pair<std::string, Value>>;

  Value() = default;
  Value(std::nullptr_t) {}  // NOLINT(runtime/explicit)
  Value(bool b)  // NOLINT(runtime/explicit)
  : type_(Type::kBool), b_(b) {}
  Value(int i)  // NOLINT(runtime/explicit)
  : type_(Type::kInt), i_(i) {}
  Value(int64_t i)  // NOLINT(runtime/explicit)
  : type_(Type::kInt), i_(i) {}
  Value(uint64_t i)  // NOLINT(runtime/explicit)
  : type_(Type::kInt), i_(static_cast<int64_t>(i)) {}
  Value(double d)  // NOLINT(runtime/explicit)
  : type_(Type::kDouble), d_(d) {}
  Value(const char * s)  // NOLINT(runtime/explicit)
  : type_(Type::kString), s_(s) {}
  Value(std::string s)  // NOLINT(runtime/explicit)
  : type_(Type::kString), s_(std::move(s)) {}
  Value(Array a)  // NOLINT(runtime/explicit)
  : type_(Type::kArray), a_(std::make_shared<Array>(std::move(a))) {}
  Value(Object o)  // NOLINT(runtime/explicit)
  : type_(Type::kObject), o_(std::make_shared<Object>(std::move(o))) {}

  static Value array() {return Value(Array{});}
  static Value object() {return Value(Object{});}

  Type type() const {return type_;}
  bool is_null() const {return type_ == Type::kNull;}
  bool is_number() const {return type_ == Type::kInt || type_ == Type::kDouble;}
  bool is_string() const {return type_ == Type::kString;}
  bool is_array() const {return type_ == Type::kArray;}
  bool is_object() const {return type_ == Type::kObject;}

  bool as_bool() const
  {
    if (type_ == Type::kBool) {
      return b_;
    }
    if (type_ == Type::kInt) {
      return i_ != 0;
    }
    throw std::runtime_error("json: not a bool");
  }
  double as_double() const
  {
    if (type_ == Type::kDouble) {
      return d_;
    }
    if (type_ == Type::kInt) {
      return static_cast<double>(i_);
    }
    if (type_ == Type::kBool) {
      return b_ ? 1.0 : 0.0;
    }
    throw std::runtime_error("json: not a number");
  }
  int64_t as_int() const
  {
    if (type_ == Type::kInt) {
      return i_;
    }
    if (type_ == Type::kBool) {
      return b_ ? 1 : 0;
    }
    if (type_ == Type::kDouble) {
      return static_cast<int64_t>(d_);
    }
    throw std::runtime_error("json: not an integer");
  }
  const std::string & as_string() const
  {
    if (type_ != Type::kString) {
      throw std::runtime_error("json: not a string");
    }
    return s_;
  }
  const Array & as_array() const
  {
    if (type_ != Type::kArray) {
      throw std::runtime_error("json: not an array");
    }
    return *a_;
  }
  Array & as_array()
  {
    if (type_ != Type::kArray) {
      throw std::runtime_error("json: not an array");
    }
    return *a_;
  }
  const Object & as_object() const
  {
    if (type_ != Type::kObject) {
      throw std::runtime_error("json: not an object");
    }
    return *o_;
  }

  // Object access. find() returns nullptr when absent.
  const Value * find(const std::string & key) const
  {
    if (type_ != Type::kObject) {
      return nullptr;
    }
    for (const auto & kv : *o_) {
      if (kv.first == key) {
        return &kv.second;
      }
    }
    return nullptr;
  }
  bool has(const std::string & key) const {return find(key) != nullptr;}
  const Value & at(const std::string & key) const
  {
    const Value * v = find(key);
    if (v == nullptr) {
      throw std::runtime_error("json: missing key '" + key + "'");
    }
    return *v;
  }
  // Set (append or replace) a key, keeping first-insertion order.
  Value & set(const std::string & key, Value v)
  {
    if (type_ == Type::kNull) {
      *this = object();
    }
    if (type_ != Type::kObject) {
      throw std::runtime_error("json: not an object");
    }
    for (auto & kv : *o_) {
      if (kv.first == key) {
        kv.second = std::move(v);
        return kv.second;
      }
    }
    o_->emplace_back(key, std::move(v));
    return o_->back().second;
  }
  void push_back(Value v)
  {
    if (type_ == Type::kNull) {
      *this = array();
    }
    as_array().push_back(std::move(v));
  }

private:
  Type type_ = Type::kNull;
  bool b_ = false;
  int64_t i_ = 0;
  double d_ = 0.0;
  std::string s_;
  std::shared_ptr<Array> a_;
  std::shared_ptr<Object> o_;
};

namespace detail
{

inline void write_string(std::string & out, const std::string & s)
{
  out += '"';
  for (std::size_t i = 0; i < s.size(); ++i) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else if (c < 0x80) {
          out += static_cast<char>(c);
        } else {
          // Decode UTF-8 and escape as Python's ensure_ascii does.
          uint32_t cp = 0;
          int extra = 0;
          if ((c & 0xe0) == 0xc0) {
            cp = c & 0x1f;
            extra = 1;
          } else if ((c & 0xf0) == 0xe0) {
            cp = c & 0x0f;
            extra = 2;
          } else {
            cp = c & 0x07;
            extra = 3;
          }
          for (int k = 0; k < extra && i + 1 < s.size(); ++k) {
            cp = (cp << 6) | (static_cast<unsigned char>(s[++i]) & 0x3f);
          }
          char buf[16];
          if (cp >= 0x10000) {
            cp -= 0x10000;
            std::snprintf(
              buf, sizeof(buf), "\\u%04x\\u%04x", 0xd800 + (cp >> 10), 0xdc00 + (cp & 0x3ff));
          } else {
            std::snprintf(buf, sizeof(buf), "\\u%04x", cp);
          }
          out += buf;
        }
    }
  }
  out += '"';
}

inline void write_double(std::string & out, double d)
{
  if (std::isnan(d)) {
    out += "NaN";
  } else if (std::isinf(d)) {
    out += d > 0 ? "Infinity" : "-Infinity";
  } else {
    phm_core::fmt::append_repr(out, d);
  }
}

inline void write(std::string & out, const Value & v, int indent, int level)
{
  const auto newline = [&out, indent](int lvl) {
      out += '\n';
      out.append(static_cast<std::size_t>(indent * lvl), ' ');
    };
  switch (v.type()) {
    case Value::Type::kNull: out += "null"; break;
    case Value::Type::kBool: out += v.as_bool() ? "true" : "false"; break;
    case Value::Type::kInt: out += std::to_string(v.as_int()); break;
    case Value::Type::kDouble: write_double(out, v.as_double()); break;
    case Value::Type::kString: write_string(out, v.as_string()); break;
    case Value::Type::kArray: {
        const auto & a = v.as_array();
        if (a.empty()) {
          out += "[]";
          break;
        }
        out += '[';
        for (std::size_t i = 0; i < a.size(); ++i) {
          if (indent >= 0) {
            newline(level + 1);
          }
          write(out, a[i], indent, level + 1);
          if (i + 1 < a.size()) {
            out += indent >= 0 ? "," : ", ";
          }
        }
        if (indent >= 0) {
          newline(level);
        }
        out += ']';
        break;
      }
    case Value::Type::kObject: {
        const auto & o = v.as_object();
        if (o.empty()) {
          out += "{}";
          break;
        }
        out += '{';
        for (std::size_t i = 0; i < o.size(); ++i) {
          if (indent >= 0) {
            newline(level + 1);
          }
          write_string(out, o[i].first);
          out += ": ";
          write(out, o[i].second, indent, level + 1);
          if (i + 1 < o.size()) {
            out += indent >= 0 ? "," : ", ";
          }
        }
        if (indent >= 0) {
          newline(level);
        }
        out += '}';
        break;
      }
  }
}

class Parser
{
public:
  explicit Parser(const std::string & text)
  : t_(text) {}

  Value parse_document()
  {
    Value v = parse_value();
    skip_ws();
    if (p_ != t_.size()) {
      fail("trailing characters");
    }
    return v;
  }

private:
  [[noreturn]] void fail(const std::string & what) const
  {
    throw std::runtime_error("json parse error at " + std::to_string(p_) + ": " + what);
  }
  void skip_ws()
  {
    while (p_ < t_.size() && (t_[p_] == ' ' || t_[p_] == '\n' || t_[p_] == '\r' ||
      t_[p_] == '\t'))
    {
      ++p_;
    }
  }
  bool consume(const char * lit)
  {
    const std::size_t n = std::char_traits<char>::length(lit);
    if (t_.compare(p_, n, lit) == 0) {
      p_ += n;
      return true;
    }
    return false;
  }
  Value parse_value()
  {
    skip_ws();
    if (p_ >= t_.size()) {
      fail("unexpected end");
    }
    const char c = t_[p_];
    if (c == '{') {
      return parse_object();
    }
    if (c == '[') {
      return parse_array();
    }
    if (c == '"') {
      return Value(parse_string());
    }
    if (consume("true")) {
      return Value(true);
    }
    if (consume("false")) {
      return Value(false);
    }
    if (consume("null")) {
      return Value();
    }
    if (consume("NaN")) {
      return Value(std::numeric_limits<double>::quiet_NaN());
    }
    if (consume("Infinity")) {
      return Value(std::numeric_limits<double>::infinity());
    }
    if (consume("-Infinity")) {
      return Value(-std::numeric_limits<double>::infinity());
    }
    return parse_number();
  }
  Value parse_number()
  {
    const std::size_t start = p_;
    bool is_float = false;
    if (t_[p_] == '-') {
      ++p_;
    }
    while (p_ < t_.size()) {
      const char c = t_[p_];
      if (c >= '0' && c <= '9') {
        ++p_;
      } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        is_float = true;
        ++p_;
      } else {
        break;
      }
    }
    if (p_ == start) {
      fail("expected a value");
    }
    const std::string tok = t_.substr(start, p_ - start);
    if (!is_float) {
      return Value(static_cast<int64_t>(std::strtoll(tok.c_str(), nullptr, 10)));
    }
    return Value(std::strtod(tok.c_str(), nullptr));
  }
  static void append_utf8(std::string & out, uint32_t cp)
  {
    if (cp < 0x80) {
      out += static_cast<char>(cp);
    } else if (cp < 0x800) {
      out += static_cast<char>(0xc0 | (cp >> 6));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    } else if (cp < 0x10000) {
      out += static_cast<char>(0xe0 | (cp >> 12));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    } else {
      out += static_cast<char>(0xf0 | (cp >> 18));
      out += static_cast<char>(0x80 | ((cp >> 12) & 0x3f));
      out += static_cast<char>(0x80 | ((cp >> 6) & 0x3f));
      out += static_cast<char>(0x80 | (cp & 0x3f));
    }
  }
  std::string parse_string()
  {
    ++p_;  // opening quote
    std::string out;
    while (p_ < t_.size() && t_[p_] != '"') {
      char c = t_[p_++];
      if (c != '\\') {
        out += c;
        continue;
      }
      if (p_ >= t_.size()) {
        fail("bad escape");
      }
      c = t_[p_++];
      switch (c) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u': {
            uint32_t cp = static_cast<uint32_t>(std::stoul(t_.substr(p_, 4), nullptr, 16));
            p_ += 4;
            if (cp >= 0xd800 && cp < 0xdc00 && t_.compare(p_, 2, "\\u") == 0) {
              const uint32_t lo =
                static_cast<uint32_t>(std::stoul(t_.substr(p_ + 2, 4), nullptr, 16));
              p_ += 6;
              cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
            }
            append_utf8(out, cp);
            break;
          }
        default:
          fail("bad escape");
      }
    }
    if (p_ >= t_.size()) {
      fail("unterminated string");
    }
    ++p_;
    return out;
  }
  Value parse_array()
  {
    ++p_;
    Value v = Value::array();
    skip_ws();
    if (p_ < t_.size() && t_[p_] == ']') {
      ++p_;
      return v;
    }
    for (;;) {
      v.push_back(parse_value());
      skip_ws();
      if (p_ < t_.size() && t_[p_] == ',') {
        ++p_;
        continue;
      }
      if (p_ < t_.size() && t_[p_] == ']') {
        ++p_;
        return v;
      }
      fail("expected , or ]");
    }
  }
  Value parse_object()
  {
    ++p_;
    Value v = Value::object();
    skip_ws();
    if (p_ < t_.size() && t_[p_] == '}') {
      ++p_;
      return v;
    }
    for (;;) {
      skip_ws();
      if (p_ >= t_.size() || t_[p_] != '"') {
        fail("expected a key");
      }
      std::string key = parse_string();
      skip_ws();
      if (p_ >= t_.size() || t_[p_] != ':') {
        fail("expected :");
      }
      ++p_;
      v.set(key, parse_value());
      skip_ws();
      if (p_ < t_.size() && t_[p_] == ',') {
        ++p_;
        continue;
      }
      if (p_ < t_.size() && t_[p_] == '}') {
        ++p_;
        return v;
      }
      fail("expected , or }");
    }
  }

  const std::string & t_;
  std::size_t p_ = 0;
};

}  // namespace detail

inline Value parse(const std::string & text)
{
  return detail::Parser(text).parse_document();
}

// json.dumps(value) with indent < 0, json.dumps(value, indent=N) otherwise.
inline std::string dumps(const Value & v, int indent = -1)
{
  std::string out;
  detail::write(out, v, indent, 0);
  return out;
}

}  // namespace json
}  // namespace phm_tools

#endif  // PHM_TOOLS__JSON_HPP_
