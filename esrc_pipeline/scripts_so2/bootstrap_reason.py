#!/usr/bin/env python3
"""Bootstrap CIs + McNemar for So2 Reason top-1 (and hit@15).

Plan recipe:
  - percentile bootstrap on a rate: 10_000 resamples, 2.5/97.5
  - paired conditions on the same users: McNemar (exact binomial)
  - Bonferroni: report raw p and p * k

Reads already-computed reason_predictions.jsonl. No GPU.

Usage:
  python esrc_pipeline/scripts_so2/bootstrap_reason.py
  python esrc_pipeline/scripts_so2/bootstrap_reason.py --n-boot 10000 --seed 0
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

ESRC_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ESRC_ROOT))

from reason.paths import reason_predictions_jsonl

N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 0
METRIC_KEYS = ("correct", "true_in_top15")


@dataclass(frozen=True)
class RateCI:
    n: int
    count: int
    rate: float
    lo: float
    hi: float


def load_ok_rows(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    by_id: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            uid = str(row.get("query_user_id") or row.get("user_id") or "")
            if not uid:
                raise ValueError(f"missing query_user_id in {path}")
            by_id[uid] = row
    if not by_id:
        raise ValueError(f"no ok rows in {path}")
    return by_id


def vec(rows: dict[str, dict[str, Any]], key: str) -> np.ndarray:
    return np.array([bool(rows[u][key]) for u in rows], dtype=np.float64)


def bootstrap_ci(
    x: np.ndarray,
    *,
    n_boot: int,
    seed: int,
) -> RateCI:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    count = int(np.sum(x))
    rate = float(x.mean()) if n else float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return RateCI(n=n, count=count, rate=rate, lo=float(lo), hi=float(hi))


def _binom_cdf_half(k: int, n: int) -> float:
    """P(X <= k) for X ~ Bin(n, 1/2)."""
    if n == 0:
        return 1.0
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    acc = 0.0
    norm = 2.0 ** n
    for i in range(k + 1):
        acc += math.comb(n, i)
    return acc / norm


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar: Bin(b+c, 1/2) on discordant counts.

    b = control correct, real wrong
    c = control wrong, real correct
    """
    n = b + c
    if n == 0:
        return 1.0
    p_le = _binom_cdf_half(min(b, c), n)
    return min(1.0, 2.0 * p_le)


def align(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    key: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("no overlapping query_user_id")
    xl = np.array([bool(left[u][key]) for u in ids], dtype=np.float64)
    xr = np.array([bool(right[u][key]) for u in ids], dtype=np.float64)
    return ids, xl, xr


def paired_stats(
    left: dict[str, dict[str, Any]],
    right: dict[str, dict[str, Any]],
    key: str,
    *,
    n_boot: int,
    seed: int,
    k_bonferroni: int,
) -> dict[str, Any]:
    ids, xl, xr = align(left, right, key)
    delta = xl - xr
    gap = bootstrap_ci(delta, n_boot=n_boot, seed=seed)
    both = int(np.sum((xl == 1) & (xr == 1)))
    neither = int(np.sum((xl == 0) & (xr == 0)))
    b = int(np.sum((xl == 1) & (xr == 0)))
    c = int(np.sum((xl == 0) & (xr == 1)))
    p_raw = mcnemar_exact_p(b, c)
    p_adj = min(1.0, p_raw * k_bonferroni)
    return {
        "n_paired": len(ids),
        "left_rate": float(xl.mean()),
        "right_rate": float(xr.mean()),
        "gap": asdict(gap),
        "table": {
            "both_correct": both,
            "neither": neither,
            "left_only": b,
            "right_only": c,
        },
        "mcnemar_p": p_raw,
        "mcnemar_p_bonferroni": p_adj,
        "k_bonferroni": k_bonferroni,
        "ci_excludes_zero": bool(gap.lo > 0 or gap.hi < 0),
    }


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_pp(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{100.0 * x:.1f} pp"


def fmt_p(p: float) -> str:
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def prediction_path(condition: str) -> Path:
    if condition == "baseline":
        return reason_predictions_jsonl("baseline")
    return reason_predictions_jsonl(condition, gallery_condition="baseline")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument(
        "--out",
        type=Path,
        default=ESRC_ROOT / "results" / "so2_stats" / "reason_bootstrap.json",
    )
    args = p.parse_args()

    labels = {
        "baseline": "Baseline (raw query)",
        "so2_a": "So2 A control",
        "so2_b": "So2 B control",
        "so2_c": "So2 C control",
        "A": "Real A (spaCy NER)",
        "B": "Real B (LLM generalize)",
        "C": "Real C (Staab)",
    }
    order = ["baseline", "so2_a", "so2_b", "so2_c", "A", "B", "C"]
    loaded = {name: load_ok_rows(prediction_path(name)) for name in order}

    rates: dict[str, dict[str, Any]] = {}
    for i, name in enumerate(order):
        rates[name] = {"label": labels[name], "n": len(loaded[name])}
        for key in METRIC_KEYS:
            # Distinct seeds per series so CIs are independent across metrics.
            seed = args.seed + 1000 * i + (0 if key == "correct" else 1)
            ci = bootstrap_ci(vec(loaded[name], key), n_boot=args.n_boot, seed=seed)
            rates[name][key] = asdict(ci)

    c1_pairs = [
        ("so2_a", "A", "A: control vs NER"),
        ("so2_b", "B", "B: control vs generalize"),
        ("so2_c", "C", "C: control vs Staab"),
    ]
    vs_base = [
        ("so2_a", "baseline", "So2 A vs baseline"),
        ("so2_b", "baseline", "So2 B vs baseline"),
        ("so2_c", "baseline", "So2 C vs baseline"),
        ("A", "baseline", "Real A vs baseline"),
        ("B", "baseline", "Real B vs baseline"),
        ("C", "baseline", "Real C vs baseline"),
    ]

    pairs: dict[str, Any] = {"c1_matched": [], "vs_baseline": []}
    for left, right, title in c1_pairs:
        row = paired_stats(
            loaded[left],
            loaded[right],
            "correct",
            n_boot=args.n_boot,
            seed=args.seed,
            k_bonferroni=len(c1_pairs),
        )
        row.update({"left": left, "right": right, "title": title, "metric": "correct"})
        pairs["c1_matched"].append(row)

    for left, right, title in vs_base:
        row = paired_stats(
            loaded[left],
            loaded[right],
            "correct",
            n_boot=args.n_boot,
            seed=args.seed,
            k_bonferroni=len(vs_base),
        )
        row.update({"left": left, "right": right, "title": title, "metric": "correct"})
        pairs["vs_baseline"].append(row)

    out = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "method": "percentile bootstrap 2.5/97.5; McNemar exact two-sided",
        "rates": rates,
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"n_boot={args.n_boot}  seed={args.seed}")
    print(f"wrote {args.out}")
    print()
    print("Reason rates  [bootstrap 95% CI]")
    print(f"{'condition':<28} {'n':>4}  {'top-1':<22}  {'hit@15':<22}")
    for name in order:
        r = rates[name]
        t1 = r["correct"]
        h15 = r["true_in_top15"]
        t1s = f"{fmt_pct(t1['rate'])}  [{fmt_pct(t1['lo'])}, {fmt_pct(t1['hi'])}]"
        h15s = f"{fmt_pct(h15['rate'])}  [{fmt_pct(h15['lo'])}, {fmt_pct(h15['hi'])}]"
        print(f"{r['label']:<28} {r['n']:>4}  {t1s:<22}  {h15s}")

    print()
    print("C1 matched pairs (control − real), McNemar k=3")
    print(
        f"{'pair':<28} {'n':>4}  {'gap':<22}  "
        f"{'p':>8}  {'p_adj':>8}  CI≠0"
    )
    for row in pairs["c1_matched"]:
        g = row["gap"]
        gaps = f"{fmt_pp(g['rate'])}  [{fmt_pp(g['lo'])}, {fmt_pp(g['hi'])}]"
        flag = "yes" if row["ci_excludes_zero"] else "no"
        print(
            f"{row['title']:<28} {row['n_paired']:>4}  {gaps:<22}  "
            f"{fmt_p(row['mcnemar_p']):>8}  {fmt_p(row['mcnemar_p_bonferroni']):>8}  {flag}"
        )
        t = row["table"]
        print(
            f"  discordant: control-only={t['left_only']}  "
            f"real-only={t['right_only']}  both={t['both_correct']}  "
            f"neither={t['neither']}"
        )

    print()
    print("Vs baseline (left − baseline), McNemar k=6")
    for row in pairs["vs_baseline"]:
        g = row["gap"]
        gaps = f"{fmt_pp(g['rate'])}  [{fmt_pp(g['lo'])}, {fmt_pp(g['hi'])}]"
        flag = "yes" if row["ci_excludes_zero"] else "no"
        print(
            f"{row['title']:<28} {row['n_paired']:>4}  {gaps:<22}  "
            f"{fmt_p(row['mcnemar_p']):>8}  {fmt_p(row['mcnemar_p_bonferroni']):>8}  {flag}"
        )


if __name__ == "__main__":
    main()
