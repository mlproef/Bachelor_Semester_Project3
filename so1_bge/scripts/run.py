#!/usr/bin/env python3
"""
Run So1 BGE similarity for conditions A/B (and optional C).

Prints per-user and condition-level cosine similarity + distance.

Usage (from repo root):
  python so1_bge/scripts/run.py --conditions A,B --limit-users 2
  python so1_bge/scripts/run.py --conditions A,B --device cpu
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

from so1_bge.src.embeddings import DEFAULT_MODEL_NAME, load_model
from so1_bge.src.io_pairs import filter_real_anonymizations, load_condition_pairs
from so1_bge.src.paths import results_dir
from so1_bge.src.similarity_annon_real import user_similarity


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.6f}"


def run_condition(
    model,
    condition: str,
    *,
    limit_users: int | None,
    batch_size: int,
    require_changed: bool,
) -> dict:
    pairs = load_condition_pairs(condition)
    if require_changed or condition.upper() == "C":
        before = len(pairs)
        pairs = filter_real_anonymizations(pairs)
        skipped = before - len(pairs)
    else:
        skipped = 0

    if limit_users is not None:
        pairs = pairs[:limit_users]

    rows = []
    for pair in tqdm(pairs, desc=f"condition {condition}", unit="user"):
        rows.append(user_similarity(model, pair, batch_size=batch_size))

    sims = np.array([r.mean_cosine for r in rows], dtype=np.float64)
    mean_sim = (
        float(np.nanmean(sims)) if sims.size and not np.all(np.isnan(sims)) else float("nan")
    )
    mean_dist = float(1.0 - mean_sim) if mean_sim == mean_sim else float("nan")

    print()
    print(f"=== Condition {condition} ===")
    print(f"users: {len(rows)}  (skipped identical profiles: {skipped})")
    print(f"{'user_id':<20} {'n':>6} {'cosine_sim':>12} {'cosine_dist':>12}")
    print("-" * 54)
    for r in rows:
        dist = 1.0 - r.mean_cosine if r.mean_cosine == r.mean_cosine else float("nan")
        print(
            f"{r.user_id:<20} {r.n_comments:>6} {_fmt(r.mean_cosine):>12} {_fmt(dist):>12}"
        )
    print("-" * 54)
    print(f"{'MEAN':<20} {'':>6} {_fmt(mean_sim):>12} {_fmt(mean_dist):>12}")
    print()

    return {
        "condition": condition.upper(),
        "n_users": len(rows),
        "skipped_identical_profiles": skipped,
        "mean_cosine_similarity": mean_sim,
        "mean_cosine_distance": mean_dist,
        "per_user": [
            {
                **asdict(r),
                "mean_cosine_distance": (
                    1.0 - r.mean_cosine if r.mean_cosine == r.mean_cosine else float("nan")
                ),
            }
            for r in rows
        ],
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So1 BGE: print cosine similarity & distance")
    p.add_argument("--conditions", default="A,B")
    p.add_argument("--model", default=DEFAULT_MODEL_NAME)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument(
        "--require-changed",
        action="store_true",
        help="Skip profiles identical to original (auto for C)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON path (default: so1_bge/results/<cond>/bge_summary.json per condition)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]

    print(f"model={args.model}  device={args.device}  conditions={conditions}")
    if args.limit_users is not None:
        print(f"limit_users={args.limit_users}")

    model = load_model(args.model)
    try:
        model = model.to(args.device)
    except Exception:
        pass

    for cond in conditions:
        summary = run_condition(
            model,
            cond,
            limit_users=args.limit_users,
            batch_size=args.batch_size,
            require_changed=args.require_changed,
        )
        if args.out is not None and len(conditions) == 1:
            out = args.out
        else:
            out = results_dir(cond) / "bge_summary.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": _utc_now(),
            "model": args.model,
            "device": args.device,
            "limit_users": args.limit_users,
            "conditions": [summary],
        }
        out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        print(f"saved → {out}")


if __name__ == "__main__":
    main()
