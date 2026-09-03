"""Load profiles as plain text for Extract (baseline or anonymized).

Input format: one JSONL per user (body / b comments).
Output for Extract: newline-joined comment texts (chunking-safe).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from extract.paths import Condition, Side, anonymized_dir


def comment_body(obj: dict) -> str:
    """Return comment text from ``body`` or compact ``b``."""
    b = obj.get("body")
    if isinstance(b, str) and b:
        return b
    short = obj.get("b")
    return short if isinstance(short, str) else ""


def list_profile_files(
    condition: Condition,
    side: Side,
    *,
    limit_files: Optional[int] = None,
) -> List[Path]:
    """Sorted JSONL paths for condition + side (baseline or anonymized)."""
    root = anonymized_dir(condition, side)
    pattern = f"user_*_{side}.jsonl"
    files = sorted(root.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {pattern} in {root}. Expected profile JSONL there."
        )
    if limit_files is not None:
        files = files[:limit_files]
    return files


def user_id_from_path(path: Path) -> str:
    """e.g. user_abc_query.jsonl → user_abc."""
    stem = path.stem  # user_abc_query / user_abc_candidate
    for suffix in ("_query", "_candidate"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_profile_text(path: Path) -> str:
    """Newline-separated comments from one JSONL (empty lines skipped)."""
    comments: List[str] = []
    for line_no, line in enumerate(
        path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
        if not isinstance(obj, dict):
            continue
        text = comment_body(obj).strip()
        if text:
            comments.append(text)
    return "\n".join(comments)
