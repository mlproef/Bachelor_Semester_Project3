"""
So1 — embedding model for semantic similarity.

Model: BAAI/bge-base-en-v1.5 (sentence-transformers).
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from shared.profiles import UserPair

from so1_bge.src.io_pairs import iter_usable_comment_pairs

DEFAULT_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_model(name: str = DEFAULT_MODEL_NAME) -> Any:
    """
    Load the sentence-transformers embedding model.

    Returns a SentenceTransformer instance. Callers should encode with
    normalize_embeddings=True (required for BGE cosine similarity).
    """
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name)


def encode_texts(
    model: Any,
    texts: Sequence[str],
    *,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts to L2-normalized embeddings (float32), shape (n, dim)."""
    if not texts:
        dim = model.get_sentence_embedding_dimension()
        return np.zeros((0, dim), dtype=np.float32)
    emb = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(emb, dtype=np.float32)


def encode_user_pair(
    model: Any,
    pair: UserPair,
    *,
    batch_size: int = 32,
) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """
    Encode usable original/anonymized comments for one UserPair.

    Returns (comment_indices, emb_original, emb_anonymized), row-aligned.
    Empty / [deleted] / [removed] on both sides are skipped (see io_pairs).
    """
    indices: List[int] = []
    originals: List[str] = []
    anonymized: List[str] = []
    for i, o, a in iter_usable_comment_pairs(pair.original, pair.anonymized):
        indices.append(i)
        # BGE rejects empty strings; keep a space so the row stays aligned.
        originals.append(o if o.strip() else " ")
        anonymized.append(a if a.strip() else " ")

    emb_o = encode_texts(model, originals, batch_size=batch_size)
    emb_a = encode_texts(model, anonymized, batch_size=batch_size)
    return indices, emb_o, emb_a
