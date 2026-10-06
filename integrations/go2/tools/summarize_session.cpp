// Copyright 2026 Yusuf Guenena. MIT License.
// phm_go2_summarize <session_dir>: reduce an onboard session evidence directory
// to the numbers it supports and print them as one JSON object.
#include <cstdio>
#include <exception>
#include <string>

#include "phm_tools/json.hpp"
#include "session.hpp"

int main(int argc, char ** argv)
{
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <session_dir>\n", argv[0]);
    return 2;
  }
  try {
    const std::string out = phm_tools::json::dumps(phm_go2::summarize_session(argv[1]), 2);
    std::fwrite(out.data(), 1, out.size(), stdout);
    std::fputc('\n', stdout);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
