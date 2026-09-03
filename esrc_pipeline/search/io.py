"""I/O helpers for Search: load extract summaries, write hits CSV."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

from search.topk import SearchHit


def load_query_summaries(
    query_dir: Path,
    *,
    user_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> tuple[list[str], list[str]]:
    """Load extract query ``{uid}.txt`` → aligned ``(user_ids, texts)``.

    If ``user_ids`` is None, use all ``*.txt`` stems in ``query_dir`` (sorted).
    Missing files are skipped with a warning. Empty directory / no usable
    files raises ``FileNotFoundError``.
    """
    if not query_dir.is_dir():
        raise FileNotFoundError(f"Query extract dir not found: {query_dir}")

    if user_ids is None:
        ids = sorted(p.stem for p in query_dir.glob("*.txt"))
    else:
        ids = list(user_ids)

    if limit is not None:
        ids = ids[:limit]

    ok_ids: list[str] = []
    texts: list[str] = []
    for uid in ids:
        path = query_dir / f"{uid}.txt"
        if not path.exists():
            print(f"WARN skip search query {uid}: no summary at {path}")
            continue
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            print(f"WARN skip search query {uid}: empty summary")
            continue
        ok_ids.append(uid)
        texts.append(text)

    if not ok_ids:
        raise FileNotFoundError(
            f"No usable query summaries in {query_dir} "
            f"(looked at {len(ids)} id(s))"
        )
    return ok_ids, texts


def write_search_hits_csv(
    path: Path,
    query_user_ids: Sequence[str],
    hits_per_query: Sequence[Sequence[SearchHit]],
) -> Path:
    """Atomic write of search_topK.csv.

    Columns: query_user_id, rank, candidate_user_id, score
    """
    if len(query_user_ids) != len(hits_per_query):
        raise ValueError(
            f"query_user_ids ({len(query_user_ids)}) != "
            f"hits_per_query ({len(hits_per_query)})"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["query_user_id", "rank", "candidate_user_id", "score"],
        )
        w.writeheader()
        for uid, hits in zip(query_user_ids, hits_per_query):
            for h in hits:
                w.writerow(
                    {
                        "query_user_id": uid,
                        "rank": h.rank,
                        "candidate_user_id": h.candidate_user_id,
                        "score": f"{h.score:.6f}",
                    }
                )
    tmp.replace(path)
    return path
