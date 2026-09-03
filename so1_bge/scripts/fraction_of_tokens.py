#!/usr/bin/env python3
"""
So1 — fraction of tokens changed (original ↔ anonymized).

Whitespace tokens after lowercasing. Per comment:
  edit_distance(tokens_orig, tokens_anon) / max(len_orig, len_anon)

Then mean over comments → one score per user; mean over users → condition.

Usage (from repo root):
  python so1_bge/scripts/fraction_of_tokens.py --conditions A,B
  python so1_bge/scripts/fraction_of_tokens.py --conditions A --limit-users 5
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so1_bge.objects.metrics import UserTokenChange
from so1_bge.src.io_pairs import iter_usable_comment_pairs, load_condition_pairs
from so1_bge.src.paths import results_dir
from shared.profiles import UserPair


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokens."""
    return text.lower().split()


def token_edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Levenshtein distance over token sequences."""
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n

    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev, curr = curr, prev
    return prev[m]


def fraction_changed(original: str, anonymized: str) -> float:
    """
    Fraction of tokens changed between two texts.

    0.0 = identical token sequence, 1.0 = fully rewritten (or empty vs non-empty).
    """
    tok_o = tokenize(original)
    tok_a = tokenize(anonymized)
    denom = max(len(tok_o), len(tok_a))
    if denom == 0:
        return 0.0
    return token_edit_distance(tok_o, tok_a) / denom


def user_fraction_changed(pair: UserPair) -> UserTokenChange:
    """Mean token-change fraction over usable comments for one user."""
    scores: List[float] = []
    for _, o, a in iter_usable_comment_pairs(pair.original, pair.anonymized):
        scores.append(fraction_changed(o, a))
    mean = float(np.mean(scores)) if scores else float("nan")
    return UserTokenChange(
        user_id=pair.user_id,
        condition=pair.condition,
        n_comments=len(scores),
        mean_fraction_changed=mean,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fmt(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.6f}"


def run_condition(
    condition: str,
    *,
    limit_users: int | None,
    out_dir: Path,
) -> dict:
    pairs = load_condition_pairs(condition)
    if limit_users is not None:
        pairs = pairs[:limit_users]

    rows = [
        user_fraction_changed(p)
        for p in tqdm(pairs, desc=f"condition {condition}", unit="user")
    ]
    values = np.array([r.mean_fraction_changed for r in rows], dtype=np.float64)
    mean_frac = (
        float(np.nanmean(values))
        if values.size and not np.all(np.isnan(values))
        else float("nan")
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    per_user_path = out_dir / f"token_change_condition_{condition}_per_user.jsonl"
    summary_path = out_dir / f"token_change_condition_{condition}.json"

    with per_user_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=True) + "\n")

    try:
        per_user_rel = str(per_user_path.relative_to(REPO_ROOT))
    except ValueError:
        per_user_rel = str(per_user_path)

    summary = {
        "created_at": _utc_now(),
        "condition": condition,
        "tokenizer": "lower().split()",
        "metric": "token_edit_distance / max(len_orig, len_anon)",
        "n_users": len(rows),
        "mean_fraction_changed": mean_frac,
        "per_user_path": per_user_rel,
        "per_user": [asdict(r) for r in rows],
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"=== Condition {condition} ===")
    print(f"users: {len(rows)}")
    print(f"{'user_id':<20} {'n':>6} {'frac_changed':>14}")
    print("-" * 44)
    for r in rows:
        print(
            f"{r.user_id:<20} {r.n_comments:>6} {_fmt(r.mean_fraction_changed):>14}"
        )
    print("-" * 44)
    print(f"{'MEAN':<20} {'':>6} {_fmt(mean_frac):>14}")
    print(f"per-user → {per_user_path}")
    print(f"summary  → {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So1 fraction of tokens changed")
    p.add_argument(
        "--conditions",
        default="A,B",
        help="Comma-separated conditions, e.g. A,B",
    )
    p.add_argument("--limit-users", type=int, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output dir (default: so1_bge/results/<a|b|c>/)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    conditions = [c.strip().upper() for c in args.conditions.split(",") if c.strip()]
    for cond in conditions:
        out = args.out_dir if args.out_dir is not None else results_dir(cond)
        run_condition(cond, limit_users=args.limit_users, out_dir=out)


if __name__ == "__main__":
    main()
