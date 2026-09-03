#!/usr/bin/env python3
"""Run ESRC Extract for So2 control C (POS-matched, query-only).

Reads::
  so2_matched/results/c/user_*_query.jsonl

Writes (does not touch real Condition C)::
  esrc_pipeline/results/so2_c/extract/query/{user_id}.txt

Usage:
  python esrc_pipeline/scripts_so2/run_extract_c.py --dry-run --limit-files 2
  python esrc_pipeline/scripts_so2/run_extract_c.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ESRC_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ESRC_ROOT.parent
sys.path.insert(0, str(ESRC_ROOT))

from extract.runner import run_extract


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ESRC_ROOT / ".env", override=False)
    load_dotenv(REPO_ROOT / ".env", override=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="qwen3.5-4b")
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--timeout", type=float, default=600.0)
    args = p.parse_args()

    _load_env()
    out = run_extract(
        condition="so2_c",
        side="query",
        model=args.model,
        force=args.force,
        resume=args.resume,
        dry_run=args.dry_run,
        limit_files=args.limit_files,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    print(out)


if __name__ == "__main__":
    main()
