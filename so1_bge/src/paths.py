"""So1-specific paths (anonymized inputs + metric output dirs).

Shared corpus roots live in ``shared.paths``; Condition dirs and So1
results folders stay here because they are So1-only.
"""
from __future__ import annotations

from pathlib import Path

from shared.paths import DATA_SPLITS, REPO_ROOT

SO1_ROOT = Path(__file__).resolve().parents[1]

# Anonymized query JSONL (same filenames as data/splits/).
CONDITION_DIRS = {
    "A": REPO_ROOT / "condition_a" / "anonymized_a" / "query",
    "B": REPO_ROOT / "condition_b" / "anonymized_b" / "query",
    "C": REPO_ROOT / "condition_c" / "anonymized_c" / "query",
}

# Per-condition metric outputs: so1_bge/results/{a,b,c}/
RESULTS_ROOT = SO1_ROOT / "results"
RESULTS_DIRS = {
    "A": RESULTS_ROOT / "a",
    "B": RESULTS_ROOT / "b",
    "C": RESULTS_ROOT / "c",
}


def results_dir(condition: str) -> Path:
    """Return so1_bge/results/<a|b|c> for a condition letter."""
    cond = condition.upper()
    if cond not in RESULTS_DIRS:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {sorted(RESULTS_DIRS)}"
        )
    return RESULTS_DIRS[cond]
