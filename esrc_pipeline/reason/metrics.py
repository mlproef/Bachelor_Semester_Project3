"""Metrics from Reason predictions (top1 / hit@15)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extract.resume import load_last_ok_by_user


def evaluate_reason(reason_jsonl: Path) -> dict[str, Any]:
    """Compute top1_accuracy and hit_at_15 from last ok row per query."""
    last_ok = load_last_ok_by_user(reason_jsonl)
    ok = list(last_ok.values())
    n = len(ok)
    top1 = sum(1 for p in ok if p.get("correct") is True)
    hit15 = sum(1 for p in ok if p.get("true_in_top15") is True)
    return {
        "n_reason_ok": n,
        "top1_correct": top1,
        "top1_accuracy": (top1 / n) if n else 0.0,
        "hit_at_15_count": hit15,
        "hit_at_15": (hit15 / n) if n else 0.0,
        "source": str(reason_jsonl.resolve()) if reason_jsonl.exists() else str(reason_jsonl),
    }


def write_reason_metrics(path: Path, metrics: dict[str, Any]) -> Path:
    """Write metrics JSON (pretty)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metrics, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path
