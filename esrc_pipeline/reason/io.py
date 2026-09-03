"""I/O helpers for Reason: Search CSV + Extract summaries."""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path
from typing import Any


def load_search_hits_by_query(search_csv: Path) -> dict[str, list[dict[str, str]]]:
    """Load search_topK.csv → query_user_id → rows (as written; not sorted).

    Expected columns: query_user_id, rank, candidate_user_id, score.
    """
    if not search_csv.is_file():
        raise FileNotFoundError(
            f"Search CSV not found: {search_csv} (run Search first)"
        )

    by_query: dict[str, list[dict[str, str]]] = {}
    with search_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uid = (row.get("query_user_id") or "").strip()
            if not uid:
                continue
            by_query.setdefault(uid, []).append(dict(row))
    if not by_query:
        raise ValueError(f"No search hits in {search_csv}")
    return by_query


def load_summary_text(summary_dir: Path, user_id: str) -> str:
    """Read ``{user_id}.txt`` from an Extract side dir."""
    path = summary_dir / f"{user_id}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Missing extract summary: {path}")
    return path.read_text(encoding="utf-8").strip()


def append_reason_row(
    path: Path,
    row: dict[str, Any],
    write_lock: threading.Lock,
) -> None:
    """Thread-safe append one JSON object as a line (mirrors extract meta)."""
    line = json.dumps(row, ensure_ascii=True) + "\n"
    with write_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
