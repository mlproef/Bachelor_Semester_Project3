"""Build / load candidate gallery embeddings from Extract candidate summaries."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from extract.paths import Condition
from search.embed import embed_texts
from search.paths import (
    DEFAULT_EMBED_MODEL,
    extract_candidate_dir,
    gallery_embeddings_npy,
    gallery_index_csv,
)


def list_summary_paths(summary_dir: Path) -> list[Path]:
    """Sorted ``{user_id}.txt`` extract summaries."""
    return sorted(summary_dir.glob("*.txt"))


def user_id_from_summary_path(path: Path) -> str:
    return path.stem


def load_or_build_candidate_gallery(
    condition: Condition,
    *,
    embed_model: str = DEFAULT_EMBED_MODEL,
    force: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Return (candidate_user_ids, embedding_matrix).

    Cache::
      results/{cond}/search/gallery_cache/candidate_embeddings.npy
      results/{cond}/search/gallery_cache/candidate_embeddings_index.csv
    """
    cand_dir = extract_candidate_dir(condition)
    paths = list_summary_paths(cand_dir)
    if not paths:
        raise FileNotFoundError(
            f"No candidate extract summaries in {cand_dir} "
            f"(run Extract for side=candidate first)"
        )

    wanted_ids = [user_id_from_summary_path(p) for p in paths]
    npy_path = gallery_embeddings_npy(condition)
    idx_path = gallery_index_csv(condition)

    if not force and npy_path.exists() and idx_path.exists():
        cached_ids = _read_index_csv(idx_path)
        matrix = np.load(npy_path)
        if (
            matrix.ndim == 2
            and matrix.shape[0] == len(cached_ids)
            and cached_ids == wanted_ids
        ):
            print(
                f"[{condition}/gallery] cache hit "
                f"({len(cached_ids)} candidates) ← {npy_path}"
            )
            return cached_ids, np.asarray(matrix, dtype=np.float32)
        print(
            f"[{condition}/gallery] cache mismatch "
            f"(cached={len(cached_ids)} disk={len(wanted_ids)}); rebuilding"
        )

    texts = [p.read_text(encoding="utf-8").strip() for p in paths]
    print(
        f"[{condition}/gallery] embedding {len(texts)} candidate summaries "
        f"with {embed_model}…"
    )
    matrix = embed_texts(texts, model_name=embed_model)
    if matrix.shape[0] != len(wanted_ids):
        raise RuntimeError(
            f"embed rows {matrix.shape[0]} != candidates {len(wanted_ids)}"
        )

    npy_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(npy_path, matrix)
    _write_index_csv(idx_path, wanted_ids)
    print(f"[{condition}/gallery] wrote cache → {npy_path}")
    return wanted_ids, matrix


def _write_index_csv(path: Path, user_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["row_id", "user_id"])
        w.writeheader()
        for i, uid in enumerate(user_ids):
            w.writerow({"row_id": i, "user_id": uid})


def _read_index_csv(path: Path) -> list[str]:
    user_ids: list[str] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user_ids.append(row["user_id"])
    return user_ids
