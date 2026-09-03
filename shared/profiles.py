"""Shared profile data classes only (no file I/O)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass(frozen=True)
class UserPair:
    """One user: original query bodies aligned with anonymized bodies (So1)."""

    user_id: str
    condition: str
    original: List[str]
    anonymized: List[str]

    @property
    def n_comments(self) -> int:
        return min(len(self.original), len(self.anonymized))


@dataclass(frozen=True)
class UserProfile:
    """
    One user's query side as raw JSONL rows.

    Used for both the original splits and the degraded copy (same shape).
    """

    user_id: str
    path: Path
    # raw JSONL rows (a, b, s, t, v) — keep dicts so we can rewrite field b later
    comments: List[Dict[str, Any]]

    @property
    def n_comments(self) -> int:
        return len(self.comments)

    def to_jsonl_lines(self) -> List[str]:
        """Each comment → one JSON line (same schema as the original splits files)."""
        return [json.dumps(obj, ensure_ascii=True) for obj in self.comments]


@dataclass(frozen=True)
class QueryCorpus:
    """
    Full query-side 'database': many UserProfile rows.

    Example names: "original", "D_matched_A", "D_matched_B".
    Same logical layout as data/splits/user_*_query.jsonl.
    """

    name: str
    profiles: List[UserProfile]

    @property
    def n_users(self) -> int:
        return len(self.profiles)

    @property
    def n_comments(self) -> int:
        return sum(p.n_comments for p in self.profiles)
