#!/usr/bin/env python3
"""Run ESRC Reason for Condition A.

Needs::
  results/a/extract/query/*.txt
  results/a/extract/candidates/*.txt
  results/a/search/search_top15.csv

Writes::
  results/a/reason/reason_predictions.jsonl
  results/a/reason/reason_metrics.json

Usage (when implemented):
  python esrc_pipeline/scripts/run_reason_a.py --dry-run --limit-queries 2
  python esrc_pipeline/scripts/run_reason_a.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ESRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ESRC_ROOT.parent
sys.path.insert(0, str(ESRC_ROOT))

from reason.paths import DEFAULT_REASON_MODEL
from reason.runner import run_reason
from search.paths import DEFAULT_TOP_K


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ESRC_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_REASON_MODEL)
    p.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--gallery-condition",
        choices=("baseline", "A", "B", "C"),
        default=None,
        help="Whose candidate extracts to pick from (default: A)",
    )
    args = p.parse_args()

    _load_env()
    out = run_reason(
        "A",
        gallery_condition=args.gallery_condition,
        model=args.model,
        k=args.k,
        force=args.force,
        resume=args.resume,
        dry_run=args.dry_run,
        limit_queries=args.limit_queries,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()
