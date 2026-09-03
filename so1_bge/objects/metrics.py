"""Metric result objects for So1."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UserSimilarity:
    """Mean BGE cosine similarity for one user (original ↔ anonymized)."""

    user_id: str
    condition: str
    n_comments: int
    mean_cosine: float


@dataclass(frozen=True)
class UserTokenChange:
    """Mean token-change fraction for one user (original ↔ anonymized)."""

    user_id: str
    condition: str
    n_comments: int
    mean_fraction_changed: float
