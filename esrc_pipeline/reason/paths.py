"""Paths for ESRC Reason (github-summer layout).

Inputs: Extract summaries + Search top-k CSV.
Output: results/{cond}/reason/reason_predictions.jsonl
"""
from __future__ import annotations

from pathlib import Path

from extract.paths import PROMPTS_DIR, RESULTS_ROOT, Condition, extract_out_dir
from search.paths import DEFAULT_TOP_K, search_hits_csv

# Default Reason model (larger than Extract; matches repo .env OLLAMA_MODEL).
DEFAULT_REASON_MODEL = "qwen3.6-35b-a3b-nvfp4"


def reason_out_dir(
    condition: Condition,
    *,
    gallery_condition: Condition | None = None,
) -> Path:
    """esrc_pipeline/results/{baseline|a|b|c}/reason/.

    A mixed gallery (one-sided attack) writes to ``reason_vs_{gallery}/`` so it
    never overwrites the same-condition run.
    """
    if gallery_condition is not None and gallery_condition != condition:
        return (
            RESULTS_ROOT
            / condition.lower()
            / f"reason_vs_{gallery_condition.lower()}"
        )
    return RESULTS_ROOT / condition.lower() / "reason"


def reason_predictions_jsonl(
    condition: Condition,
    *,
    gallery_condition: Condition | None = None,
) -> Path:
    return (
        reason_out_dir(condition, gallery_condition=gallery_condition)
        / "reason_predictions.jsonl"
    )


def reason_metrics_json(
    condition: Condition,
    *,
    gallery_condition: Condition | None = None,
) -> Path:
    return (
        reason_out_dir(condition, gallery_condition=gallery_condition)
        / "reason_metrics.json"
    )


def record_selection_prompt() -> Path:
    return PROMPTS_DIR / "record_selection_lermen_g2.txt"


def extract_query_dir(condition: Condition) -> Path:
    return extract_out_dir(condition, "query")


def extract_candidate_dir(condition: Condition) -> Path:
    return extract_out_dir(condition, "candidate")


def search_csv(
    condition: Condition,
    *,
    k: int = DEFAULT_TOP_K,
    gallery_condition: Condition | None = None,
) -> Path:
    """Search hits to reason over; mirrors search.runner's mixed-gallery layout."""
    if gallery_condition is not None and gallery_condition != condition:
        return (
            RESULTS_ROOT
            / condition.lower()
            / f"search_vs_{gallery_condition.lower()}"
            / f"search_top{k}.csv"
        )
    return search_hits_csv(condition, k=k)
