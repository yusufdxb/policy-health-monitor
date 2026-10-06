// Copyright 2026 Yusuf Guenena. MIT License.
// Small dependency-free SVG chart writer for the offline figures (GO2 session
// plots, the alpha-sweep demo). It covers what those figures use: linear and
// log axes, lines, steps, markers, horizontal / vertical rules, shaded spans,
// filled rectangles in data coordinates, histogram bars, text boxes, legends,
// custom ticks, a horizontal legend row above the axes (set_legend("above_row"))
// and a twin y-axis. Header-only; output is plain SVG 1.1.
#ifndef PHM_TOOLS__SVG_HPP_
#define PHM_TOOLS__SVG_HPP_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace phm_tools
{
namespace svg
{

struct Style
{
  std::string color = "#2a78d6";
  double width = 1.5;
  std::string dash;          // SVG stroke-dasharray, e.g. "6,4"; empty = solid
  double alpha = 1.0;
  char marker = 0;           // 0 none, 'o' circle, 's' square, '^' triangle
  std::string label;         // legend entry; empty = not in legend
};

inline std::string escape(const std::string & s)
{
  std::string o;
  for (char c : s) {
    switch (c) {
      case '&': o += "&amp;"; break;
      case '<': o += "&lt;"; break;
      case '>': o += "&gt;"; break;
      case '"': o += "&quot;"; break;
      default: o += c;
    }
  }
  return o;
}

inline std::string num(double v)
{
  char b[32];
  std::snprintf(b, sizeof(b), "%.2f", v);
  return b;
}

// Tick label text: compact, without trailing zeros.
inline std::string tick_label(double v)
{
  if (v == 0.0) {
    return "0";
  }
  char b[32];
  const double a = std::fabs(v);
  if (a >= 1e5 || a < 1e-3) {
    std::snprintf(b, sizeof(b), "%.0e", v);
  } else {
    std::snprintf(b, sizeof(b), "%.6g", v);
  }
  return b;
}

class Axes
{
public:
  Axes(double x, double y, double w, double h)
  : px_(x), py_(y), pw_(w), ph_(h) {}

  void set_xlim(double lo, double hi) {xlo_ = lo; xhi_ = hi; xset_ = true;}
  void set_ylim(double lo, double hi) {ylo_ = lo; yhi_ = hi; yset_ = true;}
  void set_ylim_low(double lo) {ylo_force_ = lo;}
  void set_log_y(bool on) {logy_ = on;}
  void set_xlabel(std::string s) {xlabel_ = std::move(s);}
  void set_ylabel(std::string s) {ylabel_ = std::move(s);}
  void set_title(std::string s) {title_ = std::move(s);}
  void set_grid(bool on) {grid_ = on;}
  void set_xticks_visible(bool on) {xticks_visible_ = on;}
  void set_yticks(std::vector<double> v, std::vector<std::string> labels)
  {
    yticks_ = std::move(v);
    ytick_labels_ = std::move(labels);
  }
  void set_legend(std::string loc) {legend_loc_ = std::move(loc);}

  void line(const std::vector<double> & x, const std::vector<double> & y, const Style & s)
  {
    series_.push_back({x, y, s, false});
    extend(x, y);
  }
  // Step plot, value held until the next x (matplotlib where="post").
  void step(const std::vector<double> & x, const std::vector<double> & y, const Style & s)
  {
    series_.push_back({x, y, s, true});
    extend(x, y);
  }
  void hline(double y, const Style & s) {hlines_.push_back({y, s});}
  void vline(double x, const Style & s) {vlines_.push_back({x, s});}
  // Shaded x-span over the full y range.
  void axvspan(double x0, double x1, const std::string & color, double alpha)
  {
    spans_.push_back({x0, x1, color, alpha});
  }
  // Filled rectangle in data coordinates.
  void rect(double x, double y, double w, double h, const std::string & color)
  {
    rects_.push_back({x, y, w, h, color});
  }
  // Histogram of `values` in `bins` equal bins (matplotlib ax.hist defaults).
  void hist(const std::vector<double> & values, int bins, const std::string & color, double alpha)
  {
    if (values.empty()) {
      return;
    }
    const auto mm = std::minmax_element(values.begin(), values.end());
    double lo = *mm.first;
    double hi = *mm.second;
    if (lo == hi) {
      lo -= 0.5;
      hi += 0.5;
    }
    std::vector<double> counts(static_cast<std::size_t>(bins), 0.0);
    for (double v : values) {
      int b = static_cast<int>((v - lo) / (hi - lo) * bins);
      b = std::min(std::max(b, 0), bins - 1);
      counts[static_cast<std::size_t>(b)] += 1.0;
    }
    const double w = (hi - lo) / bins;
    double top = 0.0;
    for (int i = 0; i < bins; ++i) {
      rects_.push_back({lo + i * w, 0.0, w, counts[static_cast<std::size_t>(i)], color, alpha});
      top = std::max(top, counts[static_cast<std::size_t>(i)]);
    }
    extend({lo, hi}, {0.0, top});
  }
  // Text at (x, y) in axes fractions (0..1), anchor "start" | "middle" | "end".
  void text(
    double fx, double fy, const std::string & s, const std::string & anchor = "start",
    double size = 10, bool box = false, const std::string & color = "#0b0b0b",
    bool top_aligned = false)
  {
    texts_.push_back({fx, fy, s, anchor, size, box, color, top_aligned});
  }

  // A twin axes sharing this one's x range, drawing its y axis on the right.
  Axes & twinx()
  {
    twin_ = std::make_unique<Axes>(px_, py_, pw_, ph_);
    twin_->is_twin_ = true;
    twin_->grid_ = false;
    return *twin_;
  }

  void render(std::string & out) const
  {
    finalize_into(out, nullptr);
  }

private:
  struct Series
  {
    std::vector<double> x, y;
    Style s;
    bool step;
  };
  struct Rule
  {
    double v;
    Style s;
  };
  struct Span
  {
    double x0, x1;
    std::string color;
    double alpha;
  };
  struct Rect
  {
    double x, y, w, h;
    std::string color;
    double alpha = 1.0;
  };
  struct Text
  {
    double fx, fy;
    std::string s, anchor;
    double size;
    bool box;
    std::string color;
    bool top;
  };

  void extend(const std::vector<double> & x, const std::vector<double> & y)
  {
    for (double v : x) {
      if (std::isfinite(v)) {
        dxlo_ = std::min(dxlo_, v);
        dxhi_ = std::max(dxhi_, v);
      }
    }
    for (double v : y) {
      if (std::isfinite(v) && (!logy_ || v > 0)) {
        dylo_ = std::min(dylo_, v);
        dyhi_ = std::max(dyhi_, v);
      }
    }
  }

  void limits(double & xlo, double & xhi, double & ylo, double & yhi) const
  {
    xlo = xset_ ? xlo_ : dxlo_;
    xhi = xset_ ? xhi_ : dxhi_;
    if (!std::isfinite(xlo) || !std::isfinite(xhi)) {
      xlo = 0;
      xhi = 1;
    }
    if (!xset_) {
      const double pad = (xhi - xlo) * 0.05;
      xlo -= pad;
      xhi += pad;
    }
    if (yset_) {
      ylo = ylo_;
      yhi = yhi_;
    } else {
      ylo = dylo_;
      yhi = dyhi_;
      for (const auto & r : hlines_) {
        if (!logy_ || r.v > 0) {
          ylo = std::min(ylo, r.v);
          yhi = std::max(yhi, r.v);
        }
      }
      if (!std::isfinite(ylo) || !std::isfinite(yhi)) {
        ylo = logy_ ? 1e-3 : 0;
        yhi = 1;
      }
      if (logy_) {
        ylo /= 1.6;
        yhi *= 1.6;
      } else {
        const double pad = (yhi - ylo) * 0.05 + (yhi == ylo ? 0.5 : 0.0);
        ylo -= pad;
        yhi += pad;
      }
    }
    if (std::isfinite(ylo_force_)) {
      ylo = ylo_force_;
    }
    if (xhi == xlo) {
      xhi = xlo + 1;
    }
    if (yhi == ylo) {
      yhi = ylo + 1;
    }
  }

  static std::vector<double> nice_ticks(double lo, double hi)
  {
    const double span = hi - lo;
    const double raw = span / 6.0;
    const double mag = std::pow(10.0, std::floor(std::log10(raw)));
    double step = mag;
    for (double m : {1.0, 2.0, 2.5, 5.0, 10.0}) {
      if (raw <= m * mag) {
        step = m * mag;
        break;
      }
    }
    std::vector<double> t;
    for (double v = std::ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) {
      t.push_back(std::fabs(v) < step * 1e-9 ? 0.0 : v);
    }
    return t;
  }

  void finalize_into(std::string & out, const Axes * parent) const
  {
    double xlo, xhi, ylo, yhi;
    limits(xlo, xhi, ylo, yhi);
    if (parent != nullptr) {
      double a, b, c, d;
      parent->limits(a, b, c, d);
      xlo = a;
      xhi = b;
    }
    const auto X = [&](double v) {return px_ + (v - xlo) / (xhi - xlo) * pw_;};
    const auto Y = [&](double v) {
        if (logy_) {
          const double l = std::log10(std::max(v, 1e-300));
          return py_ + ph_ - (l - std::log10(ylo)) / (std::log10(yhi) - std::log10(ylo)) * ph_;
        }
        return py_ + ph_ - (v - ylo) / (yhi - ylo) * ph_;
      };
    const std::string ink = "#0b0b0b";
    const std::string muted = "#52514e";
    const std::string tick = "#898781";
    const std::string grid = "#e1e0d9";
    // Clip ids are numbered in document order, so a figure renders to the same
    // bytes on every run.
    std::size_t n_clips = 0;
    for (std::size_t pos = out.find("<clipPath id="); pos != std::string::npos;
      pos = out.find("<clipPath id=", pos + 1))
    {
      ++n_clips;
    }
    const std::string clip = "clip" + std::to_string(n_clips);

    out += "<defs><clipPath id=\"" + std::string(clip) + "\"><rect x=\"" + num(px_) + "\" y=\"" +
      num(py_) + "\" width=\"" + num(pw_) + "\" height=\"" + num(ph_) + "\"/></clipPath></defs>\n";

    // Ticks and grid.
    std::vector<double> yt = yticks_;
    std::vector<std::string> ytl = ytick_labels_;
    if (yt.empty()) {
      if (logy_) {
        for (int e = static_cast<int>(std::floor(std::log10(ylo)));
          e <= static_cast<int>(std::ceil(std::log10(yhi))); ++e)
        {
          const double v = std::pow(10.0, e);
          if (v >= ylo && v <= yhi) {
            yt.push_back(v);
            char b[16];
            std::snprintf(b, sizeof(b), "1e%d", e);
            ytl.push_back(b);
          }
        }
      } else {
        yt = nice_ticks(ylo, yhi);
        for (double v : yt) {
          ytl.push_back(tick_label(v));
        }
      }
    }
    const double axis_x = is_twin_ ? px_ + pw_ : px_;
    for (std::size_t i = 0; i < yt.size(); ++i) {
      const double y = Y(yt[i]);
      if (y < py_ - 0.5 || y > py_ + ph_ + 0.5) {
        continue;
      }
      if (grid_) {
        out += "<line x1=\"" + num(px_) + "\" y1=\"" + num(y) + "\" x2=\"" + num(px_ + pw_) +
          "\" y2=\"" + num(y) + "\" stroke=\"" + grid + "\" stroke-width=\"0.8\"/>\n";
      }
      out += "<text x=\"" + num(is_twin_ ? axis_x + 6 : axis_x - 6) + "\" y=\"" + num(y + 3.5) +
        "\" font-size=\"10\" fill=\"" + tick + "\" text-anchor=\"" +
        (is_twin_ ? "start" : "end") + "\">" + escape(i < ytl.size() ? ytl[i] : "") +
        "</text>\n";
    }
    if (!is_twin_) {
      for (double v : nice_ticks(xlo, xhi)) {
        const double x = X(v);
        if (grid_) {
          out += "<line x1=\"" + num(x) + "\" y1=\"" + num(py_) + "\" x2=\"" + num(x) +
            "\" y2=\"" + num(py_ + ph_) + "\" stroke=\"" + grid + "\" stroke-width=\"0.8\"/>\n";
        }
        if (xticks_visible_) {
          out += "<text x=\"" + num(x) + "\" y=\"" + num(py_ + ph_ + 14) +
            "\" font-size=\"10\" fill=\"" + tick + "\" text-anchor=\"middle\">" +
            escape(tick_label(v)) + "</text>\n";
        }
      }
    }

    out += "<g clip-path=\"url(#" + std::string(clip) + ")\">\n";
    for (const auto & s : spans_) {
      out += "<rect x=\"" + num(X(s.x0)) + "\" y=\"" + num(py_) + "\" width=\"" +
        num(X(s.x1) - X(s.x0)) + "\" height=\"" + num(ph_) + "\" fill=\"" + s.color +
        "\" fill-opacity=\"" + num(s.alpha) + "\"/>\n";
    }
    for (const auto & r : rects_) {
      const double y0 = Y(logy_ ? std::max(r.y, ylo) : r.y);
      const double y1 = Y(r.y + r.h);
      out += "<rect x=\"" + num(X(r.x)) + "\" y=\"" + num(std::min(y0, y1)) + "\" width=\"" +
        num(std::max(X(r.x + r.w) - X(r.x), 0.0)) + "\" height=\"" +
        num(std::fabs(y1 - y0)) + "\" fill=\"" + r.color + "\" fill-opacity=\"" + num(r.alpha) +
        "\" stroke=\"white\" stroke-width=\"0.5\"/>\n";
    }
    for (const auto & ser : series_) {
      std::string pts;
      for (std::size_t i = 0; i < ser.x.size(); ++i) {
        if (!std::isfinite(ser.x[i]) || !std::isfinite(ser.y[i]) || (logy_ && ser.y[i] <= 0)) {
          continue;
        }
        if (ser.step && i > 0) {
          pts += num(X(ser.x[i])) + "," + num(Y(ser.y[i - 1])) + " ";
        }
        pts += num(X(ser.x[i])) + "," + num(Y(ser.y[i])) + " ";
      }
      out += "<polyline fill=\"none\" points=\"" + pts + "\" stroke=\"" + ser.s.color +
        "\" stroke-width=\"" + num(ser.s.width) + "\" stroke-opacity=\"" + num(ser.s.alpha) +
        "\"" + (ser.s.dash.empty() ? "" : " stroke-dasharray=\"" + ser.s.dash + "\"") +
        " stroke-linejoin=\"round\"/>\n";
      if (ser.s.marker) {
        for (std::size_t i = 0; i < ser.x.size(); ++i) {
          if (!std::isfinite(ser.y[i])) {
            continue;
          }
          out += marker(ser.s.marker, X(ser.x[i]), Y(ser.y[i]), ser.s.color);
        }
      }
    }
    for (const auto & r : hlines_) {
      out += "<line x1=\"" + num(px_) + "\" y1=\"" + num(Y(r.v)) + "\" x2=\"" + num(px_ + pw_) +
        "\" y2=\"" + num(Y(r.v)) + "\" stroke=\"" + r.s.color + "\" stroke-width=\"" +
        num(r.s.width) + "\"" +
        (r.s.dash.empty() ? "" : " stroke-dasharray=\"" + r.s.dash + "\"") + "/>\n";
    }
    for (const auto & r : vlines_) {
      out += "<line x1=\"" + num(X(r.v)) + "\" y1=\"" + num(py_) + "\" x2=\"" + num(X(r.v)) +
        "\" y2=\"" + num(py_ + ph_) + "\" stroke=\"" + r.s.color + "\" stroke-width=\"" +
        num(r.s.width) + "\"" +
        (r.s.dash.empty() ? "" : " stroke-dasharray=\"" + r.s.dash + "\"") + "/>\n";
    }
    out += "</g>\n";

    // Axis lines (left and bottom; the twin draws the right one).
    if (is_twin_) {
      out += "<line x1=\"" + num(px_ + pw_) + "\" y1=\"" + num(py_) + "\" x2=\"" +
        num(px_ + pw_) + "\" y2=\"" + num(py_ + ph_) + "\" stroke=\"#c3c2b7\"/>\n";
    } else {
      out += "<line x1=\"" + num(px_) + "\" y1=\"" + num(py_) + "\" x2=\"" + num(px_) +
        "\" y2=\"" + num(py_ + ph_) + "\" stroke=\"#c3c2b7\"/>\n";
      out += "<line x1=\"" + num(px_) + "\" y1=\"" + num(py_ + ph_) + "\" x2=\"" +
        num(px_ + pw_) + "\" y2=\"" + num(py_ + ph_) + "\" stroke=\"#c3c2b7\"/>\n";
    }
    if (!xlabel_.empty()) {
      out += "<text x=\"" + num(px_ + pw_ / 2) + "\" y=\"" + num(py_ + ph_ + 32) +
        "\" font-size=\"11\" fill=\"" + muted + "\" text-anchor=\"middle\">" + escape(xlabel_) +
        "</text>\n";
    }
    if (!ylabel_.empty()) {
      const double lx = is_twin_ ? px_ + pw_ + 58 : px_ - 52;
      out += "<text transform=\"translate(" + num(lx) + "," + num(py_ + ph_ / 2) +
        ") rotate(-90)\" font-size=\"11\" fill=\"" + muted + "\" text-anchor=\"middle\">" +
        escape(ylabel_) + "</text>\n";
    }
    if (!title_.empty()) {
      out += "<text x=\"" + num(px_) + "\" y=\"" + num(py_ - 10) + "\" font-size=\"13\" fill=\"" +
        ink + "\">" + escape(title_) + "</text>\n";
    }
    for (const auto & t : texts_) {
      render_text(out, t);
    }
    if (twin_) {
      twin_->finalize_into(out, this);
    }
    if (!legend_loc_.empty()) {
      render_legend(out);
    }
  }

  static std::string marker(char m, double x, double y, const std::string & color)
  {
    if (m == 's') {
      return "<rect x=\"" + num(x - 3) + "\" y=\"" + num(y - 3) +
             "\" width=\"6\" height=\"6\" fill=\"" + color + "\"/>\n";
    }
    if (m == '^') {
      return "<polygon points=\"" + num(x) + "," + num(y - 4) + " " + num(x - 3.5) + "," +
             num(y + 3) + " " + num(x + 3.5) + "," + num(y + 3) + "\" fill=\"" + color + "\"/>\n";
    }
    return "<circle cx=\"" + num(x) + "\" cy=\"" + num(y) + "\" r=\"3\" fill=\"" + color + "\"/>\n";
  }

  void render_text(std::string & out, const Text & t) const
  {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (true) {
      const std::size_t nl = t.s.find('\n', start);
      lines.push_back(t.s.substr(start, nl == std::string::npos ? nl : nl - start));
      if (nl == std::string::npos) {
        break;
      }
      start = nl + 1;
    }
    const double lh = t.size * 1.3;
    const double x = px_ + t.fx * pw_;
    double y = py_ + (1.0 - t.fy) * ph_;
    if (t.top) {
      y += t.size;
    } else {
      y -= lh * static_cast<double>(lines.size() - 1);
    }
    if (t.box) {
      std::size_t longest = 0;
      for (const auto & l : lines) {
        longest = std::max(longest, l.size());
      }
      const double w = static_cast<double>(longest) * t.size * 0.55 + 12;
      const double h = lh * static_cast<double>(lines.size()) + 6;
      double bx = x - 6;
      if (t.anchor == "middle") {
        bx = x - w / 2;
      } else if (t.anchor == "end") {
        bx = x - w + 6;
      }
      out += "<rect x=\"" + num(bx) + "\" y=\"" + num(y - t.size - 2) + "\" width=\"" + num(w) +
        "\" height=\"" + num(h) + "\" fill=\"white\" stroke=\"#e1e0d9\"/>\n";
    }
    for (std::size_t i = 0; i < lines.size(); ++i) {
      out += "<text x=\"" + num(x) + "\" y=\"" + num(y + lh * static_cast<double>(i)) +
        "\" font-size=\"" + num(t.size) + "\" fill=\"" + t.color + "\" text-anchor=\"" +
        t.anchor + "\">" + escape(lines[i]) + "</text>\n";
    }
  }

  void collect_legend(std::vector<Style> & items) const
  {
    for (const auto & s : series_) {
      if (!s.s.label.empty()) {
        items.push_back(s.s);
      }
    }
    for (const auto & r : hlines_) {
      if (!r.s.label.empty()) {
        items.push_back(r.s);
      }
    }
    for (const auto & r : vlines_) {
      if (!r.s.label.empty()) {
        items.push_back(r.s);
      }
    }
    for (const auto & e : extra_legend_) {
      items.push_back(e);
    }
    if (twin_) {
      twin_->collect_legend(items);
    }
  }

public:
  // A legend-only swatch (e.g. state colors for rectangles).
  void legend_patch(const std::string & color, const std::string & label)
  {
    Style s;
    s.color = color;
    s.label = label;
    s.width = -1;  // patch
    extra_legend_.push_back(s);
  }

private:
  // One horizontal row of entries, right-aligned just above the axes.
  void render_legend_row(std::string & out, const std::vector<Style> & items) const
  {
    const auto entry_width = [](const Style & s) {
        return 36.0 + static_cast<double>(s.label.size()) * 5.4;
      };
    double total = 0;
    for (const auto & s : items) {
      total += entry_width(s);
    }
    double x = px_ + pw_ - total;
    const double cy = py_ - 10;
    for (const auto & s : items) {
      if (s.width < 0) {
        out += "<rect x=\"" + num(x) + "\" y=\"" + num(cy - 5) +
          "\" width=\"18\" height=\"10\" fill=\"" + s.color + "\"/>\n";
      } else {
        out += "<line x1=\"" + num(x) + "\" y1=\"" + num(cy) + "\" x2=\"" + num(x + 18) +
          "\" y2=\"" + num(cy) + "\" stroke=\"" + s.color + "\" stroke-width=\"" +
          num(std::max(s.width, 1.0)) + "\"" +
          (s.dash.empty() ? "" : " stroke-dasharray=\"" + s.dash + "\"") + "/>\n";
      }
      out += "<text x=\"" + num(x + 24) + "\" y=\"" + num(cy + 3.5) +
        "\" font-size=\"9.5\" fill=\"#0b0b0b\">" + escape(s.label) + "</text>\n";
      x += entry_width(s);
    }
  }

  void render_legend(std::string & out) const
  {
    std::vector<Style> items;
    collect_legend(items);
    if (items.empty()) {
      return;
    }
    if (legend_loc_ == "above_row") {
      render_legend_row(out, items);
      return;
    }
    const double lh = 14;
    std::size_t longest = 0;
    for (const auto & s : items) {
      longest = std::max(longest, s.label.size());
    }
    const double w = static_cast<double>(longest) * 5.4 + 36;
    const double h = lh * static_cast<double>(items.size()) + 8;
    double x = px_ + 8;
    double y = py_ + 8;
    if (legend_loc_.find("right") != std::string::npos) {
      x = px_ + pw_ - w - 8;
    }
    if (legend_loc_.find("lower") != std::string::npos) {
      y = py_ + ph_ - h - 8;
    } else if (legend_loc_.find("center") != std::string::npos) {
      y = py_ + ph_ / 2 - h / 2;
    } else if (legend_loc_ == "above") {
      x = px_ + pw_ - w;
      y = py_ - h - 2;
    }
    out += "<rect x=\"" + num(x) + "\" y=\"" + num(y) + "\" width=\"" + num(w) + "\" height=\"" +
      num(h) + "\" fill=\"white\" fill-opacity=\"0.9\" stroke=\"#e1e0d9\"/>\n";
    for (std::size_t i = 0; i < items.size(); ++i) {
      const auto & s = items[i];
      const double cy = y + 4 + lh * static_cast<double>(i) + lh / 2;
      if (s.width < 0) {
        out += "<rect x=\"" + num(x + 6) + "\" y=\"" + num(cy - 5) +
          "\" width=\"18\" height=\"10\" fill=\"" + s.color + "\"/>\n";
      } else {
        out += "<line x1=\"" + num(x + 6) + "\" y1=\"" + num(cy) + "\" x2=\"" + num(x + 24) +
          "\" y2=\"" + num(cy) + "\" stroke=\"" + s.color + "\" stroke-width=\"" +
          num(std::max(s.width, 1.0)) + "\"" +
          (s.dash.empty() ? "" : " stroke-dasharray=\"" + s.dash + "\"") + "/>\n";
        if (s.marker) {
          out += marker(s.marker, x + 15, cy, s.color);
        }
      }
      out += "<text x=\"" + num(x + 30) + "\" y=\"" + num(cy + 3.5) +
        "\" font-size=\"9.5\" fill=\"#0b0b0b\">" + escape(s.label) + "</text>\n";
    }
  }

  double px_, py_, pw_, ph_;
  double xlo_ = 0, xhi_ = 1, ylo_ = 0, yhi_ = 1;
  bool xset_ = false, yset_ = false, logy_ = false, grid_ = true, is_twin_ = false;
  bool xticks_visible_ = true;
  double ylo_force_ = std::numeric_limits<double>::quiet_NaN();
  double dxlo_ = std::numeric_limits<double>::infinity();
  double dxhi_ = -std::numeric_limits<double>::infinity();
  double dylo_ = std::numeric_limits<double>::infinity();
  double dyhi_ = -std::numeric_limits<double>::infinity();
  std::string xlabel_, ylabel_, title_, legend_loc_;
  std::vector<double> yticks_;
  std::vector<std::string> ytick_labels_;
  std::vector<Series> series_;
  std::vector<Rule> hlines_, vlines_;
  std::vector<Span> spans_;
  std::vector<Rect> rects_;
  std::vector<Text> texts_;
  std::vector<Style> extra_legend_;
  std::unique_ptr<Axes> twin_;
};

// A figure of fixed pixel size holding axes placed in pixel coordinates.
class Figure
{
public:
  Figure(double width, double height)
  : w_(width), h_(height) {}

  Axes & add_axes(double x, double y, double w, double h)
  {
    axes_.push_back(std::make_unique<Axes>(x, y, w, h));
    return *axes_.back();
  }
  void text(double x, double y, const std::string & s, double size, const std::string & color)
  {
    texts_.push_back("<text x=\"" + num(x) + "\" y=\"" + num(y) + "\" font-size=\"" + num(size) +
      "\" fill=\"" + color + "\">" + escape(s) + "</text>\n");
  }

  std::string render() const
  {
    std::string out = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
      "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"" + num(w_) + "\" height=\"" + num(h_) +
      "\" viewBox=\"0 0 " + num(w_) + " " + num(h_) + "\" font-family=\"DejaVu Sans, Arial, "
      "Helvetica, sans-serif\">\n<rect width=\"100%\" height=\"100%\" fill=\"white\"/>\n";
    for (const auto & a : axes_) {
      a->render(out);
    }
    for (const auto & t : texts_) {
      out += t;
    }
    out += "</svg>\n";
    return out;
  }

  void save(const std::string & path) const
  {
    std::ofstream f(path);
    if (!f) {
      throw std::runtime_error("cannot write " + path);
    }
    f << render();
  }

private:
  double w_, h_;
  std::vector<std::unique_ptr<Axes>> axes_;
  std::vector<std::string> texts_;
};

}  // namespace svg
}  // namespace phm_tools

#endif  // PHM_TOOLS__SVG_HPP_
