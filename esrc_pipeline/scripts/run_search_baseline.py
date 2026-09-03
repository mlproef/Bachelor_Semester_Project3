#!/usr/bin/env python3
"""Run ESRC Search for baseline (no anonymization).

Needs Extract outputs::
  esrc_pipeline/results/baseline/extract/query/*.txt
  esrc_pipeline/results/baseline/extract/candidates/*.txt

Writes::
  esrc_pipeline/results/baseline/search/search_top15.csv
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
    args = p.parse_args()

    out = run_search(
        "baseline",
        k=args.k,
        embed_model=args.embed_model,
        force_gallery=args.force_gallery,
        limit_queries=args.limit_queries,
        dry_run=args.dry_run,
    )
    print(out)


if __name__ == "__main__":
    main()
