#!/usr/bin/env python3
"""Run Condition C anonymization (Staab-style chunked infer→anonymize).

Thin wrapper around run_batch_chunked_anonymize.py with repo-friendly defaults
(same style as run_condition_a / run_condition_b).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.6-35b-a3b-nvfp4")
_DEFAULT_PROFILES = REPO_ROOT / "data" / "profiles" / "profiles_query.jsonl"
_DEFAULT_OUT = ROOT / "results_condition_c"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--profiles",
        type=Path,
        default=_DEFAULT_PROFILES,
        help="Staab profiles jsonl (default: shared data/profiles/profiles_query.jsonl)",
    )
    p.add_argument(
        "--out-root",
        type=Path,
        default=_DEFAULT_OUT,
        help="Output directory (default: condition_c/results_condition_c)",
    )
    p.add_argument("--chunk-size", type=int, default=20)
    p.add_argument("--max-iterations", type=int, default=3)
    p.add_argument("--model", type=str, default=_DEFAULT_MODEL)
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument("--offset-users", type=int, default=0)
    p.add_argument("--usernames", type=str, default=None)
    p.add_argument("--limit-chunks", type=int, default=None)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute chunks even if present",
    )
    p.add_argument(
        "--redo-complete",
        action="store_true",
        help="Do not skip users marked complete",
    )
    args = p.parse_args()

    # Delegate to the existing batch entrypoint with the same CLI flags.
    argv = [
        "run_batch_chunked_anonymize.py",
        "--profiles",
        str(args.profiles),
        "--out-root",
        str(args.out_root),
        "--chunk-size",
        str(args.chunk_size),
        "--max-iterations",
        str(args.max_iterations),
        "--model",
        args.model,
        "--offset-users",
        str(args.offset_users),
    ]
    if args.limit_users is not None:
        argv += ["--limit-users", str(args.limit_users)]
    if args.usernames:
        argv += ["--usernames", args.usernames]
    if args.limit_chunks is not None:
        argv += ["--limit-chunks", str(args.limit_chunks)]
    if args.no_resume:
        argv.append("--no-resume")
    if args.redo_complete:
        argv.append("--redo-complete")

    sys.argv = argv
    from run_batch_chunked_anonymize import main as batch_main

    batch_main()


if __name__ == "__main__":
    main()
