#!/usr/bin/env python3
"""Run Condition A anonymization (spaCy NER) on shared data/splits candidate profiles."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.anonymize_a import run_condition_a
from src.paths import ensure_project_dirs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit-files", type=int, default=None)
    p.add_argument("--spacy-model", type=str, default="en_core_web_lg")
    args = p.parse_args()

    ensure_project_dirs()
    out = run_condition_a(
        side="candidate",
        force=args.force,
        dry_run=args.dry_run,
        limit_files=args.limit_files,
        spacy_model=args.spacy_model,
    )
    print(out)


if __name__ == "__main__":
    main()
