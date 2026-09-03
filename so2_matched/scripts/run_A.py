#!/usr/bin/env python3
"""
So2 matched — Condition A rate only.

Usage (from repo root):
  python so2_matched/scripts/run_A.py
  python so2_matched/scripts/run_A.py --limit-users 5
  python so2_matched/scripts/run_A.py --force
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so2_matched.src.paths import DEFAULT_LEXICON, out_dir
from so2_matched.src.random_from_spacy import run_matched


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So2 matched POS degradation — match A")
    p.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out-dir", type=Path, default=out_dir("A"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_matched(
        match="A",
        lexicon_path=args.lexicon,
        limit_users=args.limit_users,
        seed=args.seed,
        force=args.force,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
