"""
So1 step 1 — load and align original ↔ anonymized query texts.

Data class UserPair lives in shared.profiles; file I/O stays here (So1).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

from shared.paths import DATA_SPLITS
from shared.profiles import UserPair

from so1_bge.src.paths import CONDITION_DIRS
from so1_bge.src.reddit_jsonl import comment_body

DELETED_MARKERS = {"", "[deleted]", "[removed]"}


def load_bodies(path: Path) -> List[str]:
    """Read a query JSONL file and return comment texts in file order."""
    bodies: List[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            bodies.append("")
            continue
        bodies.append(comment_body(obj))
    return bodies


def _user_id_from_filename(name: str) -> str:
    # user_0168255c_query.jsonl -> user_0168255c
    stem = Path(name).stem
    if stem.endswith("_query"):
        return stem[: -len("_query")]
    return stem


def _profile_text_equal(original: Sequence[str], anonymized: Sequence[str]) -> bool:
    return "\n".join(original) == "\n".join(anonymized)


def _is_unusable(text: str) -> bool:
    return text.strip().lower() in DELETED_MARKERS


def iter_usable_comment_pairs(
    original: Sequence[str],
    anonymized: Sequence[str],
) -> Iterator[Tuple[int, str, str]]:
    """
    Yield (index, orig, anon) for comments usable in embedding similarity.

    Skips pairs where both sides are empty / [deleted] / [removed].
    If only one side is unusable, the pair is still yielded (real content loss).
    Aligns by index up to min(len(original), len(anonymized)).
    """
    n = min(len(original), len(anonymized))
    for i in range(n):
        o, a = original[i], anonymized[i]
        if _is_unusable(o) and _is_unusable(a):
            continue
        yield i, o, a


def load_condition_pairs(condition: str) -> List[UserPair]:
    """
    Load all original↔anonymized query pairs for one condition ('A', 'B', or 'C').

    Skips files that have no matching original in data/splits/.
    Does not filter identical profiles — use filter_real_anonymizations for that.
    """
    cond = condition.upper()
    if cond not in CONDITION_DIRS:
        raise ValueError(
            f"Unknown condition {condition!r}; expected one of {sorted(CONDITION_DIRS)}"
        )

    anon_dir = CONDITION_DIRS[cond]
    if not anon_dir.is_dir():
        raise FileNotFoundError(f"Missing anonymized dir: {anon_dir}")

    pairs: List[UserPair] = []
    for anon_path in sorted(anon_dir.glob("user_*_query.jsonl")):
        orig_path = DATA_SPLITS / anon_path.name
        if not orig_path.is_file():
            continue
        pairs.append(
            UserPair(
                user_id=_user_id_from_filename(anon_path.name),
                condition=cond,
                original=load_bodies(orig_path),
                anonymized=load_bodies(anon_path),
            )
        )
    return pairs


def filter_real_anonymizations(pairs: Iterable[UserPair]) -> List[UserPair]:
    """
    Keep only users whose anonymized query text differs from the original.

    Needed for Condition C (placeholder copies). Harmless for A/B.
    """
    return [p for p in pairs if not _profile_text_equal(p.original, p.anonymized)]
