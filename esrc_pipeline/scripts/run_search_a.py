#!/usr/bin/env python3
"""Run ESRC Search for Condition A.

Needs Extract query summaries::
  esrc_pipeline/results/a/extract/query/*.txt

Gallery is Extract candidate summaries (same condition by default)::
  esrc_pipeline/results/a/extract/candidates/*.txt

One-sided (A queries vs raw / baseline candidates)::
  python esrc_pipeline/scripts/run_search_a.py --gallery-condition baseline

Writes::
  esrc_pipeline/results/a/search/search_top15.csv
  or, if gallery differs:
  esrc_pipeline/results/a/search_vs_baseline/search_top15.csv

Usage:
  python esrc_pipeline/scripts/run_search_a.py --dry-run
  python esrc_pipeline/scripts/run_search_a.py --limit-queries 5
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ESRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESRC_ROOT))

from search.paths import DEFAULT_EMBED_MODEL, DEFAULT_TOP_K
from search.runner import run_search


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--force-gallery", action="store_true")
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--gallery-condition",
        choices=("baseline", "A", "B", "C"),
        default=None,
        help="Whose candidate extracts to search against (default: A)",
    )
    args = p.parse_args()

    # TODO: load_dotenv if remote embed is added later
    out = run_search(
        "A",
        k=args.k,
        embed_model=args.embed_model,
        force_gallery=args.force_gallery,
        limit_queries=args.limit_queries,
        dry_run=args.dry_run,
        gallery_condition=args.gallery_condition,
    )
    print(out)


if __name__ == "__main__":
    main()
