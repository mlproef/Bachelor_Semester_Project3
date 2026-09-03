"""Orchestrate ESRC Search for one condition (baseline|A|B|C).

Flow (from esrc_stas phase_search, adapted to github-summer)::

  1. load_or_build_candidate_gallery(condition)   # from extract/candidates/
  2. load_query_summaries(extract/query/)
  3. embed_texts(query summaries)
  4. search_top_k(..., k=15)
  5. write search_top15.csv under results/{cond}/search/
"""
from __future__ import annotations

from pathlib import Path

from extract.paths import Condition
from search.embed import embed_texts
from search.gallery import load_or_build_candidate_gallery
from search.io import load_query_summaries, write_search_hits_csv
from search.paths import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_TOP_K,
    extract_query_dir,
    search_out_dir,
)
from search.topk import search_top_k


def run_search(
    condition: Condition,
    *,
    k: int = DEFAULT_TOP_K,
    embed_model: str = DEFAULT_EMBED_MODEL,
    force_gallery: bool = False,
    limit_queries: int | None = None,
    dry_run: bool = False,
    gallery_condition: Condition | None = None,
    out_dir: Path | None = None,
) -> Path:
    """Run Search for ``condition`` queries; return path to search_top{k}.csv.

    ``gallery_condition`` selects whose Extract candidate summaries form the
    gallery (default: same as ``condition``). One-sided A vs raw candidates::
      run_search("A", gallery_condition="baseline")
    """
    from extract.paths import RESULTS_ROOT

    gallery_condition = gallery_condition or condition
    if out_dir is None:
        if gallery_condition == condition:
            out_dir = search_out_dir(condition)
        else:
            out_dir = (
                RESULTS_ROOT
                / condition.lower()
                / f"search_vs_{gallery_condition.lower()}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"search_top{k}.csv"

    print(
        f"[{condition}/search] building/loading candidate gallery "
        f"(gallery={gallery_condition})…"
    )
    cand_ids, cand_matrix = load_or_build_candidate_gallery(
        gallery_condition,
        embed_model=embed_model,
        force=force_gallery,
    )

    q_dir = extract_query_dir(condition)
    q_ids, q_texts = load_query_summaries(q_dir, limit=limit_queries)
    print(
        f"[{condition}/search] {len(q_ids)} queries × {len(cand_ids)} candidates "
        f"(gallery={gallery_condition}, k={k}, model={embed_model})"
    )

    if dry_run:
        print(f"[{condition}/search] dry_run — not embedding / writing")
        print(f"[{condition}/search] would write → {out_csv}")
        return out_csv

    print(f"[{condition}/search] embedding {len(q_texts)} query summaries…")
    q_vecs = embed_texts(q_texts, model_name=embed_model)
    if q_vecs.shape[0] != len(q_ids):
        raise RuntimeError(
            f"query embed rows {q_vecs.shape[0]} != queries {len(q_ids)}"
        )
    if q_vecs.shape[1] != cand_matrix.shape[1]:
        raise RuntimeError(
            f"query dim {q_vecs.shape[1]} != gallery dim {cand_matrix.shape[1]} "
            f"(use the same embed_model for both)"
        )

    hits = search_top_k(q_vecs, cand_matrix, cand_ids, k=k)
    write_search_hits_csv(out_csv, q_ids, hits)
    print(f"[{condition}/search] wrote {out_csv} ({len(q_ids)} queries)")
    return out_csv
