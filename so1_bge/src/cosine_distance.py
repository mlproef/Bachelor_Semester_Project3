"""
So1 — pairwise cosine similarity / distance for BGE embeddings.

Embeddings from encode_* are L2-normalized, so for row vectors u, v:

    cosine_similarity(u, v) = u · v
    cosine_distance(u, v)   = 1 − (u · v)
"""
from __future__ import annotations

import numpy as np


def cosine_similarity(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine similarity for L2-normalized embeddings.

    similarity_i = sum_j emb_a[i, j] * emb_b[i, j]
    """
    if emb_a.shape != emb_b.shape:
        raise ValueError(f"shape mismatch: {emb_a.shape} vs {emb_b.shape}")
    return np.sum(emb_a * emb_b, axis=1)


def cosine_distance(emb_a: np.ndarray, emb_b: np.ndarray) -> np.ndarray:
    """
    Row-wise cosine distance for L2-normalized embeddings.

    distance_i = 1 − similarity_i
    """
    return 1.0 - cosine_similarity(emb_a, emb_b)
