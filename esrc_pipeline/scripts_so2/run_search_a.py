#!/usr/bin/env python3
"""Run ESRC Search for So2 control A vs a candidate gallery.

Needs Extract::
  esrc_pipeline/results/so2_a/extract/query/*.txt

One-sided default (control queries vs raw candidates)::
  python esrc_pipeline/scripts_so2/run_search_a.py --gallery-condition baseline

Writes::
  esrc_pipeline/results/so2_a/search_vs_baseline/search_top15.csv
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
        default="baseline",
        help="Whose candidate extracts to search against (default: baseline)",
    )
    args = p.parse_args()

    out = run_search(
        "so2_a",
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
