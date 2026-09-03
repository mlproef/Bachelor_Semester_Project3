"""So3 paths — candidate anonymization."""
from __future__ import annotations

from pathlib import Path

from shared.paths import DATA_SPLITS, REPO_ROOT

SO3_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = SO3_ROOT / "results"

# Original candidates (untouched in one-sided runs).
ORIGINAL_CANDIDATES = DATA_SPLITS

OUT_DIRS = {
    "A": RESULTS_ROOT / "a",
    "B": RESULTS_ROOT / "b",
    "C": RESULTS_ROOT / "c",
}


def out_dir(condition: str) -> Path:
    cond = condition.upper()
    if cond not in OUT_DIRS:
        raise ValueError(f"Unknown condition {condition!r}; expected A/B/C")
    return OUT_DIRS[cond]
