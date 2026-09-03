#!/usr/bin/env python3
"""Run Condition B anonymization (LLM generalization) via OpenAI-compatible API.

Writes to anonymized_b/query/ or anonymized_b/candidates/ (same layout as A).

Usage:
  python condition_b/scripts/run_condition_b.py --side query --dry-run
  python condition_b/scripts/run_condition_b.py --side candidate --limit-files 1
  python condition_b/scripts/run_condition_b.py --side both
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(ROOT.parent / ".env")
except ImportError:
    pass

from src.anonymize_b import run_condition_b
from src.paths import ensure_project_dirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--side",
        choices=("query", "candidate", "both"),
        default="query",
        help="Which side(s) to anonymize (default: query)",
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument(
        "--model",
        type=str,
        default=os.getenv("CONDITION_B_MODEL") or "qwen3.5-4b",
        help="Chat model (default: qwen3.5-4b; override with --model or CONDITION_B_MODEL)",
    )
    p.add_argument("--sleep", type=float, default=0.0)
    args = p.parse_args()

    ensure_project_dirs()

    sides = ["query", "candidate"] if args.side == "both" else [args.side]
    outs: list[Path] = []
    for side in sides:
        out = run_condition_b(
            side=side,
            force=args.force,
            dry_run=args.dry_run,
            limit_files=args.limit_files,
            model=args.model,
            sleep_s=args.sleep,
        )
        outs.append(out)
        print(out)


if __name__ == "__main__":
    main()
