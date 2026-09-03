#!/usr/bin/env python3
"""Bootstrap CIs for So1 utility metrics (BGE cosine, token change, hand ratings).

Plan recipe:
  - percentile bootstrap on a mean: 10_000 resamples, 2.5/97.5
  - paired conditions on the same users/pairs: bootstrap the difference
  - Bonferroni: also report k=3 percentile CIs (0.833/99.167) on pairwise gaps
  - no McNemar: these are continuous means, not binary rates

Reads already-computed per-user JSONL and the 50-pair rating table. No GPU.

Usage (from repo root):
  python so1_bge/scripts/bootstrap_utility.py
  python so1_bge/scripts/bootstrap_utility.py --n-boot 10000 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from so1_bge.src.paths import RESULTS_ROOT, results_dir

N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 0
CONDITIONS = ("A", "B", "C")
LABELS = {
    "A": "A (spaCy NER)",
    "B": "B (LLM generalize)",
    "C": "C (Staab)",
}
PAIR_TITLES = (
    ("A", "B", "A − B"),
    ("A", "C", "A − C"),
    ("B", "C", "B − C"),
)
K_BONFERRONI = 3
# 95% with Bonferroni k=3 → α/2k = 0.05/6
BONF_PCTS = (100.0 * (0.05 / (2 * K_BONFERRONI)), 100.0 * (1.0 - 0.05 / (2 * K_BONFERRONI)))


@dataclass(frozen=True)
class MeanCI:
    n: int
    n_used: int
    mean: float
    lo: float
    hi: float


def load_jsonl_map(path: Path, value_key: str) -> dict[str, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    out: dict[str, float] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = str(row.get("user_id") or "")
            if not uid:
                raise ValueError(f"missing user_id in {path}")
            if value_key not in row:
                raise ValueError(f"missing {value_key} in {path} for {uid}")
            out[uid] = float(row[value_key])
    if not out:
        raise ValueError(f"no rows in {path}")
    return out


def parse_score(raw: str) -> float:
    text = (raw or "").strip()
    if text == "":
        return float("nan")
    val = float(text)
    if val == -1:
        return float("nan")
    return val


def load_hand_ratings(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Return per-condition arrays aligned by pair order (n=50)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    meaning: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    fluency: dict[str, list[float]] = {c: [] for c in CONDITIONS}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not (row.get("number") or "").strip():
                continue
            for cond in CONDITIONS:
                meaning[cond].append(parse_score(row.get(f"meaning_{cond}", "")))
                fluency[cond].append(parse_score(row.get(f"fluency_{cond}", "")))
    n = len(meaning["A"])
    if n != 50:
        raise ValueError(f"expected 50 hand-rating pairs, got {n} from {path}")
    for cond in CONDITIONS:
        if len(meaning[cond]) != n or len(fluency[cond]) != n:
            raise ValueError(f"ragged hand ratings for {cond}")
    out: dict[str, dict[str, np.ndarray]] = {}
    for cond in CONDITIONS:
        m = np.asarray(meaning[cond], dtype=np.float64)
        fl = np.asarray(fluency[cond], dtype=np.float64)
        avg = (m + fl) / 2.0
        out[cond] = {"meaning": m, "fluency": fl, "avg": avg}
    return out


def full_sample(x: np.ndarray) -> np.ndarray:
    """Map N/A (−1 already nan) to 5, matching hand_ratings_summary.json."""
    y = np.asarray(x, dtype=np.float64).copy()
    y[np.isnan(y)] = 5.0
    return y


def bootstrap_mean(
    x: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    skip_nan: bool = False,
    percentiles: tuple[float, float] = (2.5, 97.5),
) -> MeanCI:
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    if skip_nan:
        used = x[~np.isnan(x)]
    else:
        used = x
    n_used = int(used.size)
    mean = float(used.mean()) if n_used else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = x[idx]
    if skip_nan:
        means = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            v = samples[i][~np.isnan(samples[i])]
            means[i] = v.mean() if v.size else np.nan
        means = means[np.isfinite(means)]
    else:
        means = samples.mean(axis=1)
    lo, hi = np.percentile(means, list(percentiles))
    return MeanCI(n=n, n_used=n_used, mean=mean, lo=float(lo), hi=float(hi))


def paired_delta(
    left: np.ndarray,
    right: np.ndarray,
    *,
    n_boot: int,
    seed: int,
    skip_nan: bool = False,
) -> dict[str, Any]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size:
        raise ValueError("paired vectors must have the same length")
    n = int(left.size)
    if skip_nan:
        lv = left[~np.isnan(left)]
        rv = right[~np.isnan(right)]
        left_mean = float(lv.mean()) if lv.size else float("nan")
        right_mean = float(rv.mean()) if rv.size else float("nan")
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        deltas = np.empty(n_boot, dtype=np.float64)
        for i in range(n_boot):
            a = left[idx[i]]
            b = right[idx[i]]
            va = a[~np.isnan(a)]
            vb = b[~np.isnan(b)]
            if va.size == 0 or vb.size == 0:
                deltas[i] = np.nan
            else:
                deltas[i] = va.mean() - vb.mean()
        finite = deltas[np.isfinite(deltas)]
        gap_mean = left_mean - right_mean
        lo, hi = np.percentile(finite, [2.5, 97.5])
        blo, bhi = np.percentile(finite, list(BONF_PCTS))
        gap = MeanCI(n=n, n_used=n, mean=float(gap_mean), lo=float(lo), hi=float(hi))
        gap_bonf = MeanCI(n=n, n_used=n, mean=float(gap_mean), lo=float(blo), hi=float(bhi))
    else:
        delta = left - right
        gap = bootstrap_mean(delta, n_boot=n_boot, seed=seed)
        gap_bonf = bootstrap_mean(
            delta, n_boot=n_boot, seed=seed, percentiles=BONF_PCTS
        )
        left_mean = float(left.mean())
        right_mean = float(right.mean())
    return {
        "n_paired": n,
        "left_mean": left_mean,
        "right_mean": right_mean,
        "gap": asdict(gap),
        "gap_bonferroni": asdict(gap_bonf),
        "k_bonferroni": K_BONFERRONI,
        "ci_excludes_zero": bool(gap.lo > 0 or gap.hi < 0),
        "ci_bonf_excludes_zero": bool(gap_bonf.lo > 0 or gap_bonf.hi < 0),
    }


def align_maps(
    left: dict[str, float], right: dict[str, float]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    ids = sorted(set(left) & set(right))
    if not ids:
        raise ValueError("no overlapping user_id")
    xl = np.array([left[u] for u in ids], dtype=np.float64)
    xr = np.array([right[u] for u in ids], dtype=np.float64)
    return ids, xl, xr


def fmt_signed(x: float, digits: int = 3) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.{digits}f}"


def fmt_pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def fmt_pp(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{100.0 * x:.1f} pp"


def print_mean_row(label: str, ci: MeanCI, *, as_pct: bool) -> None:
    if as_pct:
        body = f"{fmt_pct(ci.mean)}  [{fmt_pct(ci.lo)}, {fmt_pct(ci.hi)}]"
    else:
        body = f"{ci.mean:.3f}  [{ci.lo:.3f}, {ci.hi:.3f}]"
    extra = f"  n_used={ci.n_used}" if ci.n_used != ci.n else ""
    print(f"{label:<28} {ci.n:>4}{extra}  {body}")


def print_gap_row(title: str, row: dict[str, Any], *, as_pct: bool, digits: int = 3) -> None:
    g = row["gap"]
    gb = row["gap_bonferroni"]
    if as_pct:
        gaps = f"{fmt_pp(g['mean'])}  [{fmt_pp(g['lo'])}, {fmt_pp(g['hi'])}]"
        bonf = f"[{fmt_pp(gb['lo'])}, {fmt_pp(gb['hi'])}]"
    else:
        gaps = (
            f"{fmt_signed(g['mean'], digits)}  "
            f"[{fmt_signed(g['lo'], digits)}, {fmt_signed(g['hi'], digits)}]"
        )
        bonf = f"[{fmt_signed(gb['lo'], digits)}, {fmt_signed(gb['hi'], digits)}]"
    flag = "yes" if row["ci_excludes_zero"] else "no"
    bflag = "yes" if row["ci_bonf_excludes_zero"] else "no"
    print(
        f"{title:<12} {row['n_paired']:>4}  {gaps:<28}  "
        f"CI≠0 {flag:<3}  bonf {bonf}  ≠0 {bflag}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    p.add_argument("--seed", type=int, default=SEED_DEFAULT)
    p.add_argument(
        "--hand-csv",
        type=Path,
        default=results_dir("A") / "hand_ratings.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=RESULTS_ROOT / "so1_stats" / "utility_bootstrap.json",
    )
    args = p.parse_args()

    bge = {
        c: load_jsonl_map(
            results_dir(c) / f"bge_condition_{c}_per_user.jsonl", "mean_cosine"
        )
        for c in CONDITIONS
    }
    tok = {
        c: load_jsonl_map(
            results_dir(c) / f"token_change_condition_{c}_per_user.jsonl",
            "mean_fraction_changed",
        )
        for c in CONDITIONS
    }
    hand = load_hand_ratings(args.hand_csv)

    rates: dict[str, Any] = {}
    for i, cond in enumerate(CONDITIONS):
        rates[cond] = {
            "label": LABELS[cond],
            "n_users_bge": len(bge[cond]),
            "n_users_token": len(tok[cond]),
            "n_pairs_hand": 50,
        }
        bge_vals = np.array(list(bge[cond].values()), dtype=np.float64)
        tok_vals = np.array(list(tok[cond].values()), dtype=np.float64)
        rates[cond]["bge_cosine"] = asdict(
            bootstrap_mean(bge_vals, n_boot=args.n_boot, seed=args.seed + 1000 * i)
        )
        rates[cond]["token_change"] = asdict(
            bootstrap_mean(
                tok_vals, n_boot=args.n_boot, seed=args.seed + 1000 * i + 1
            )
        )
        for j, key in enumerate(("meaning", "fluency", "avg")):
            raw = hand[cond][key]
            rates[cond][f"hand_{key}_full"] = asdict(
                bootstrap_mean(
                    full_sample(raw),
                    n_boot=args.n_boot,
                    seed=args.seed + 1000 * i + 10 + j,
                )
            )
            rates[cond][f"hand_{key}_given_change"] = asdict(
                bootstrap_mean(
                    raw,
                    n_boot=args.n_boot,
                    seed=args.seed + 1000 * i + 20 + j,
                    skip_nan=True,
                )
            )

    pairs: dict[str, Any] = {
        "bge_cosine": [],
        "token_change": [],
        "hand_avg_full": [],
        "hand_avg_given_change": [],
        "hand_meaning_full": [],
        "hand_fluency_full": [],
    }
    for left, right, title in PAIR_TITLES:
        _, xl, xr = align_maps(bge[left], bge[right])
        row = paired_delta(xl, xr, n_boot=args.n_boot, seed=args.seed)
        row.update({"left": left, "right": right, "title": title, "metric": "bge_cosine"})
        pairs["bge_cosine"].append(row)

        _, tl, tr = align_maps(tok[left], tok[right])
        row = paired_delta(tl, tr, n_boot=args.n_boot, seed=args.seed + 50)
        row.update(
            {"left": left, "right": right, "title": title, "metric": "token_change"}
        )
        pairs["token_change"].append(row)

        row = paired_delta(
            full_sample(hand[left]["avg"]),
            full_sample(hand[right]["avg"]),
            n_boot=args.n_boot,
            seed=args.seed + 100,
        )
        row.update(
            {"left": left, "right": right, "title": title, "metric": "hand_avg_full"}
        )
        pairs["hand_avg_full"].append(row)

        row = paired_delta(
            hand[left]["avg"],
            hand[right]["avg"],
            n_boot=args.n_boot,
            seed=args.seed + 150,
            skip_nan=True,
        )
        row.update(
            {
                "left": left,
                "right": right,
                "title": title,
                "metric": "hand_avg_given_change",
            }
        )
        pairs["hand_avg_given_change"].append(row)

        row = paired_delta(
            full_sample(hand[left]["meaning"]),
            full_sample(hand[right]["meaning"]),
            n_boot=args.n_boot,
            seed=args.seed + 200,
        )
        row.update(
            {
                "left": left,
                "right": right,
                "title": title,
                "metric": "hand_meaning_full",
            }
        )
        pairs["hand_meaning_full"].append(row)

        row = paired_delta(
            full_sample(hand[left]["fluency"]),
            full_sample(hand[right]["fluency"]),
            n_boot=args.n_boot,
            seed=args.seed + 250,
        )
        row.update(
            {
                "left": left,
                "right": right,
                "title": title,
                "metric": "hand_fluency_full",
            }
        )
        pairs["hand_fluency_full"].append(row)

    out = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "method": (
            "percentile bootstrap 2.5/97.5 on user/pair means; "
            "paired difference on overlapping ids; "
            f"Bonferroni k={K_BONFERRONI} CIs at {BONF_PCTS[0]:.3f}/{BONF_PCTS[1]:.3f} percentiles"
        ),
        "hand_csv": str(args.hand_csv),
        "na_coding": "-1 → nan (unchanged / N/A); full_sample maps nan → 5",
        "rates": rates,
        "pairs": pairs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n")

    print(f"n_boot={args.n_boot}  seed={args.seed}")
    print(f"wrote {args.out}")
    print()
    print("BGE cosine  [bootstrap 95% CI]  (higher = more meaning kept)")
    for cond in CONDITIONS:
        print_mean_row(LABELS[cond], MeanCI(**rates[cond]["bge_cosine"]), as_pct=False)
    print()
    print("C1 pairs BGE (left − right), k=3")
    for row in pairs["bge_cosine"]:
        print_gap_row(row["title"], row, as_pct=False, digits=3)

    print()
    print("Token change  [bootstrap 95% CI]  (higher = more text damaged)")
    for cond in CONDITIONS:
        print_mean_row(LABELS[cond], MeanCI(**rates[cond]["token_change"]), as_pct=True)
    print()
    print("C1 pairs token change (left − right), k=3")
    for row in pairs["token_change"]:
        print_gap_row(row["title"], row, as_pct=True)

    print()
    print("Hand ratings, full sample (N/A → 5)  [bootstrap 95% CI]")
    print(f"{'condition':<28} {'n':>4}  {'avg':<22}  {'meaning':<22}  {'fluency'}")
    for cond in CONDITIONS:
        a = rates[cond]["hand_avg_full"]
        m = rates[cond]["hand_meaning_full"]
        fl = rates[cond]["hand_fluency_full"]
        print(
            f"{LABELS[cond]:<28} {a['n']:>4}  "
            f"{a['mean']:.2f}  [{a['lo']:.2f}, {a['hi']:.2f}]   "
            f"{m['mean']:.2f}  [{m['lo']:.2f}, {m['hi']:.2f}]   "
            f"{fl['mean']:.2f}  [{fl['lo']:.2f}, {fl['hi']:.2f}]"
        )
    print()
    print("C1 pairs hand avg full (left − right), k=3")
    for row in pairs["hand_avg_full"]:
        print_gap_row(row["title"], row, as_pct=False, digits=2)

    print()
    print("Hand ratings, given change (−1 excluded)  [bootstrap 95% CI]")
    for cond in CONDITIONS:
        a = rates[cond]["hand_avg_given_change"]
        print(
            f"{LABELS[cond]:<28} n={a['n']} n_used={a['n_used']}  "
            f"{a['mean']:.2f}  [{a['lo']:.2f}, {a['hi']:.2f}]"
        )
    print()
    print("C1 pairs hand avg given-change (left − right), k=3")
    for row in pairs["hand_avg_given_change"]:
        print_gap_row(row["title"], row, as_pct=False, digits=2)


if __name__ == "__main__":
    main()
