#!/usr/bin/env python3
"""Standalone doc-alignment gate for Mamba-3-Lite.

Thin wrapper around `tests/test_doc_refs.py`, which is the single source of
truth for anchor resolution, public-symbol coverage, line-anchor, Triton
citation, and link checks. This shim exists so the workspace-standard
command works; when no flags are given it defaults to `--coverage --links`.

Usage:
    python3 scripts/check_docs.py                # --coverage --links
    python3 scripts/check_docs.py --coverage     # coverage only
    python3 scripts/check_docs.py                # plus any test_doc_refs flag
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))

import test_doc_refs  # noqa: E402


def main() -> int:
    argv = sys.argv[1:]
    if not any(a in ("--coverage", "--links") for a in argv):
        argv = ["--coverage", "--links", *argv]
    sys.argv = [sys.argv[0], *argv]
    return test_doc_refs.main()


if __name__ == "__main__":
    sys.exit(main())
