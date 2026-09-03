"""Paths for ESRC Search (github-summer layout).

Inputs come from Extract outputs; Search writes under results/{a|b|c}/search/.
"""
from __future__ import annotations

from pathlib import Path

from extract.paths import RESULTS_ROOT, Condition, extract_out_dir

# Default embedding model (local sentence-transformers; BSP-compatible).
# TODO: optional remote embed via OLLAMA_URL / jina-embeddings-v3 on gpu6.
DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_TOP_K = 15


def search_out_dir(condition: Condition) -> Path:
    """esrc_pipeline/results/{a|b|c}/search/."""
    return RESULTS_ROOT / condition.lower() / "search"


def search_hits_csv(condition: Condition, *, k: int = DEFAULT_TOP_K) -> Path:
    """Main Search artifact for Reason: search_top{k}.csv."""
    return search_out_dir(condition) / f"search_top{k}.csv"


def gallery_cache_dir(condition: Condition) -> Path:
    """Cache for candidate embedding matrix + index CSV."""
    return search_out_dir(condition) / "gallery_cache"


def gallery_embeddings_npy(condition: Condition) -> Path:
    return gallery_cache_dir(condition) / "candidate_embeddings.npy"


def gallery_index_csv(condition: Condition) -> Path:
    return gallery_cache_dir(condition) / "candidate_embeddings_index.csv"


def extract_query_dir(condition: Condition) -> Path:
    """Extract summaries used as Search queries."""
    return extract_out_dir(condition, "query")


def extract_candidate_dir(condition: Condition) -> Path:
    """Extract summaries used as Search gallery."""
    return extract_out_dir(condition, "candidate")
