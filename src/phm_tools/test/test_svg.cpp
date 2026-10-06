// Copyright 2026 Yusuf Guenena. MIT License.
// The SVG writer renders a figure to the same bytes every time: clip-path ids
// are numbered in document order, not taken from object addresses.
#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "phm_tools/svg.hpp"

namespace
{

std::string render_two_panel_figure()
{
  namespace svg = phm_tools::svg;
  svg::Figure fig(640, 480);
  svg::Style s;
  s.label = "series";
  svg::Axes & top = fig.add_axes(60, 40, 520, 160);
  top.line({0.0, 1.0, 2.0}, {1.0, 3.0, 2.0}, s);
  top.twinx().line({0.0, 1.0, 2.0}, {10.0, 5.0, 0.0}, s);
  svg::Axes & bottom = fig.add_axes(60, 260, 520, 160);
  bottom.line({0.0, 1.0}, {0.0, 1.0}, s);
  bottom.set_legend("upper right");
  return fig.render();
}

}  // namespace

TEST(Svg, RenderIsReproducible)
{
  const std::string a = render_two_panel_figure();
  const std::string b = render_two_panel_figure();
  EXPECT_EQ(a, b);
}

TEST(Svg, ClipIdsAreUniqueAndInDocumentOrder)
{
  const std::string doc = render_two_panel_figure();
  std::vector<std::string> ids;
  const std::string key = "<clipPath id=\"";
  for (std::size_t pos = doc.find(key); pos != std::string::npos; pos = doc.find(key, pos + 1)) {
    const std::size_t start = pos + key.size();
    ids.push_back(doc.substr(start, doc.find('"', start) - start));
  }
  ASSERT_EQ(ids.size(), 3u);
  EXPECT_EQ(ids[0], "clip0");
  EXPECT_EQ(ids[1], "clip1");
  EXPECT_EQ(ids[2], "clip2");
  for (const auto & id : ids) {
    EXPECT_NE(doc.find("url(#" + id + ")"), std::string::npos) << id;
  }
}
