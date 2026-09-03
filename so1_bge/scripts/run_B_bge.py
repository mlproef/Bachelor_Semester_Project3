#!/usr/bin/env python3
"""
So1 BGE — run Condition B only (all query users by default).

Usage (from repo root):
  python so1_bge/scripts/run_B_bge.py
  python so1_bge/scripts/run_B_bge.py --limit-users 5
  python so1_bge/scripts/run_B_bge.py --no-resume
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so1_bge.objects.metrics import UserSimilarity
from so1_bge.src.embeddings import DEFAULT_MODEL_NAME, load_model
from so1_bge.src.io_pairs import load_condition_pairs
from so1_bge.src.paths import results_dir
from so1_bge.src.similarity_annon_real import user_similarity

OUT_DIR = results_dir("B")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _row_dict(row: UserSimilarity) -> dict:
    sim = row.mean_cosine
    dist = 1.0 - sim if sim == sim else float("nan")
    return {**asdict(row), "mean_cosine_distance": dist}


def _load_done(path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            done[obj["user_id"]] = obj
    return done


def run_condition_B(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "cpu",
    batch_size: int = 32,
    limit_users: int | None = None,
    resume: bool = True,
    out_dir: Path = OUT_DIR,
) -> dict:
    """
    Score Condition B: original ↔ LLM generalization anonymized queries.

    Appends one JSON line per user to bge_condition_B_per_user.jsonl,
    then writes aggregate summary to bge_condition_B.json.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = out_dir / "bge_condition_B_per_user.jsonl"
    summary_path = out_dir / "bge_condition_B.json"

    pairs = load_condition_pairs("B")
    if limit_users is not None:
        pairs = pairs[:limit_users]

    done = _load_done(per_user_path) if resume else {}
    if not resume and per_user_path.is_file():
        per_user_path.unlink()
        done = {}

    pending = [p for p in pairs if p.user_id not in done]
    print(
        f"[B] model={model_name} device={device} "
        f"total={len(pairs)} done={len(done)} pending={len(pending)}"
    )

    model = load_model(model_name)
    try:
        model = model.to(device)
    except Exception:
        pass

    with per_user_path.open("a", encoding="utf-8") as out_f:
        for pair in tqdm(pending, desc="condition B", unit="user"):
            row = user_similarity(model, pair, batch_size=batch_size)
            obj = _row_dict(row)
            done[pair.user_id] = obj
            out_f.write(json.dumps(obj, ensure_ascii=True) + "\n")
            out_f.flush()

    ordered = [done[p.user_id] for p in pairs if p.user_id in done]
    sims = np.array([r["mean_cosine"] for r in ordered], dtype=np.float64)
    mean_sim = (
        float(np.nanmean(sims)) if sims.size and not np.all(np.isnan(sims)) else float("nan")
    )
    mean_dist = float(1.0 - mean_sim) if mean_sim == mean_sim else float("nan")

    try:
        per_user_rel = str(per_user_path.relative_to(REPO_ROOT))
    except ValueError:
        per_user_rel = str(per_user_path)

    summary = {
        "created_at": _utc_now(),
        "condition": "B",
        "model": model_name,
        "device": device,
        "n_users": len(ordered),
        "mean_cosine_similarity": mean_sim,
        "mean_cosine_distance": mean_dist,
        "per_user_path": per_user_rel,
        "per_user": ordered,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2), encoding="utf-8")

    print()
    print("=== Condition B ===")
    print(f"users: {len(ordered)}")
    print(f"mean cosine similarity: {mean_sim:.6f}")
    print(f"mean cosine distance:   {mean_dist:.6f}")
    print(f"per-user → {per_user_path}")
    print(f"summary  → {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So1 BGE Condition B only")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="Ignore checkpoint and restart")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_condition_B(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        limit_users=args.limit_users,
        resume=not args.no_resume,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
