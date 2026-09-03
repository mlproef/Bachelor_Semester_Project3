"""Paths for ESRC Extract (github-summer layout).

Conditions:
  baseline     — raw profiles in data/splits/ (no anonymization)
  A|B|C        — anonymized under condition_*/anonymized_*/
  so2_a|so2_b|so2_c — So2 POS-matched fake defence (query JSONL only)
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[2]
ESRC_ROOT = Path(__file__).resolve().parents[1]

Condition = Literal["baseline", "A", "B", "C", "so2_a", "so2_b", "so2_c"]
Side = Literal["query", "candidate"]

SO2_CONDITIONS: tuple[Condition, ...] = ("so2_a", "so2_b", "so2_c")

# Raw (non-anonymized) pool — query + candidate JSONL in one flat dir
SPLITS_DIR = REPO_ROOT / "data" / "splits"

# Where anonymized JSONL live after condition_* runs
ANON_ROOTS = {
    "A": REPO_ROOT / "condition_a" / "anonymized_a",
    "B": REPO_ROOT / "condition_b" / "anonymized_b",
    "C": REPO_ROOT / "condition_c" / "anonymized_c",
}

# So2 control texts: flat user_*_query.jsonl (no candidates yet)
SO2_QUERY_DIRS = {
    "so2_a": REPO_ROOT / "so2_matched" / "results" / "a",
    "so2_b": REPO_ROOT / "so2_matched" / "results" / "b",
    "so2_c": REPO_ROOT / "so2_matched" / "results" / "c",
}

# Side subdirs for anonymized A (and target layout). Baseline is flat in SPLITS_DIR.
SIDE_SUBDIRS = {
    "query": "query",
    "candidate": "candidates",
}

PROMPTS_DIR = ESRC_ROOT / "prompts"
RESULTS_ROOT = ESRC_ROOT / "results"


def so2_condition(match: str) -> Condition:
    """Map So1/So2 letter A|B|C → extract condition so2_a|so2_b|so2_c."""
    m = match.strip().upper()
    if m not in ("A", "B", "C"):
        raise ValueError(f"So2 match must be A, B, or C, got {match!r}")
    return f"so2_{m.lower()}"  # type: ignore[return-value]


def _check_condition(condition: Condition) -> None:
    if (
        condition != "baseline"
        and condition not in ANON_ROOTS
        and condition not in SO2_QUERY_DIRS
    ):
        raise ValueError(
            f"Unknown condition {condition!r}; "
            "expected baseline|A|B|C|so2_a|so2_b|so2_c"
        )


def anonymized_dir(condition: Condition, side: Side) -> Path:
    """Return directory with user_*_{query|candidate}.jsonl for this condition/side.

    Layout:
      baseline      → data/splits/  (flat; both sides in one folder)
      A|B|C         → condition_*/anonymized_*/{query|candidates}/
      so2_a|so2_b|so2_c → so2_matched/results/{a|b|c}/  (query JSONL only)
    """
    _check_condition(condition)
    if side not in SIDE_SUBDIRS:
        raise ValueError(f"Unknown side {side!r}; expected query|candidate")
    if condition == "baseline":
        return SPLITS_DIR
    if condition in SO2_QUERY_DIRS:
        if side != "query":
            raise FileNotFoundError(
                f"So2 condition {condition} is query-only "
                f"(no POS-degraded candidates in {SO2_QUERY_DIRS[condition]})"
            )
        return SO2_QUERY_DIRS[condition]
    root = ANON_ROOTS.get(condition)
    if root is None:
        raise ValueError(f"Unknown condition {condition!r}")
    return root / SIDE_SUBDIRS[side]


def extract_out_dir(condition: Condition, side: Side) -> Path:
    """esrc_pipeline/results/{baseline|a|b|c|so2_a|…}/extract/{query|candidates}/."""
    _check_condition(condition)
    if side not in SIDE_SUBDIRS:
        raise ValueError(f"Unknown side {side!r}; expected query|candidate")
    return RESULTS_ROOT / condition.lower() / "extract" / SIDE_SUBDIRS[side]


def extract_merge_prompt() -> Path:
    """Local merge prompt for chunked extract."""
    return PROMPTS_DIR / "extract_merge.txt"


def summarization_prompt() -> Path:
    """Lermen-style summarization prompt (Extract)."""
    return PROMPTS_DIR / "summarization_lermen_g2.txt"


def record_selection_prompt() -> Path:
    """Lermen-style record selection prompt (Reason, later)."""
    return PROMPTS_DIR / "record_selection_lermen_g2.txt"
