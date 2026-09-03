"""
So1 — aggregate semantic similarity: anonymized vs original.

Pipeline per user:
  encode usable comments → row-wise cosine similarity → mean over comments

Then mean over users for a condition.
"""
from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np

from shared.profiles import UserPair

from so1_bge.src.cosine_distance import cosine_similarity
from so1_bge.src.embeddings import encode_user_pair
from so1_bge.objects.metrics import UserSimilarity


def mean_cosine(scores: np.ndarray) -> float:
    """Aggregate comment-level similarities to one scalar."""
    if scores.size == 0:
        return float("nan")
    return float(np.mean(scores))


def user_similarity(
    model: Any,
    pair: UserPair,
    *,
    batch_size: int = 32,
) -> UserSimilarity:
    """Encode one UserPair and return mean cosine(original, anonymized)."""
    _indices, emb_o, emb_a = encode_user_pair(model, pair, batch_size=batch_size)
    scores = cosine_similarity(emb_o, emb_a)
    return UserSimilarity(
        user_id=pair.user_id,
        condition=pair.condition,
        n_comments=int(scores.size),
        mean_cosine=mean_cosine(scores),
    )


def condition_similarity(
    model: Any,
    pairs: Sequence[UserPair],
    *,
    batch_size: int = 32,
) -> tuple[List[UserSimilarity], float]:
    """
    Score every user; return (per-user rows, mean of user means).

    Users with no usable comments get mean_cosine=nan and are ignored
    in the condition aggregate.
    """
    rows = [user_similarity(model, p, batch_size=batch_size) for p in pairs]
    values = np.array([r.mean_cosine for r in rows], dtype=np.float64)
    if values.size == 0 or np.all(np.isnan(values)):
        agg = float("nan")
    else:
        agg = float(np.nanmean(values))
    return rows, agg
