#!/usr/bin/env python3
"""Reject tracked text that exposes machine-local or private build details."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


def _rules() -> tuple[tuple[str, re.Pattern[str]], ...]:
    hardware_pattern = "|".join(
        (
            "Jetson" + r"\s+" + "Orin",
            r"(?:GeForce\s+)?" + "RTX" + r"\s*\d+",
            "AMD" + r"\s+" + "Ryzen" + r"\s+\d+",
            "Intel" + r"\s+" + "Core" + r"(?:\s+i[3579])?(?:-|\s+)\d+",
            "Apple" + r"\s+M\d+",
            r"\d+\s*(?:GB|GiB)\s+(?:VRAM|GPU\s+memory)",
        )
    )
    return (
        ("Unix user-home path", re.compile("/" + "home/" + r"[^/\s]+/")),
        ("macOS user-home path", re.compile("/" + "Users/" + r"[^/\s]+/")),
        (
            "Windows user-home path",
            re.compile(r"[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+\\"),
        ),
        ("home-directory shorthand", re.compile("~" + r"/")),
        ("private hardware identifier", re.compile(hardware_pattern, re.IGNORECASE)),
        ("U+2014 em dash", re.compile(chr(0x2014))),
    )


def _tracked_files(repo_root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return [repo_root / name.decode() for name in result.stdout.split(b"\0") if name]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    findings: list[str] = []
    for path in _tracked_files(repo_root):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative = path.relative_to(repo_root)
        for line_number, line in enumerate(text.splitlines(), start=1):
            for label, pattern in _rules():
                if pattern.search(line):
                    findings.append(f"{relative}:{line_number}: {label}")

    if findings:
        print("Public repository hygiene check failed:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        return 1

    print("Public repository hygiene check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
