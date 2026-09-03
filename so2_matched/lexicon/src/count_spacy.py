#!/usr/bin/env python3
"""
Build a POS → word frequency lexicon from the full splits corpus.

Reads all user_*_query.jsonl and user_*_candidate.jsonl under data/splits/,
runs spaCy, counts (token.text.lower(), pos_) pairs.
Skips space / punct / empty tokens.

Usage (from repo root):
  python so2_matched/lexicon/src/count_spacy.py
  python so2_matched/lexicon/src/count_spacy.py --limit-files 5
  python so2_matched/lexicon/src/count_spacy.py --sides query
  python so2_matched/lexicon/src/count_spacy.py --model en_core_web_sm

Output default: so2_matched/lexicon/result/pos_lexicon.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List

from tqdm import tqdm

# so2_matched/lexicon/src/count_spacy.py → repo root is parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
LEXICON_ROOT = Path(__file__).resolve().parents[1]  # so2_matched/lexicon
OUT_DIR = LEXICON_ROOT / "result"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared.paths import DATA_SPLITS
from so1_bge.src.reddit_jsonl import comment_body  # noqa: E402

DEFAULT_MODEL = "en_core_web_lg"
SKIP_POS = {"SPACE", "PUNCT", "X"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iter_query_bodies(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = comment_body(obj).strip()
            if text:
                yield text


def build_lexicon(
    bodies: Iterable[str],
    nlp,
    *,
    batch_size: int = 64,
) -> Dict[str, Counter]:
    """Return {POS: Counter({word: count})}."""
    by_pos: DefaultDict[str, Counter] = defaultdict(Counter)
    texts = list(bodies)
    for doc in tqdm(
        nlp.pipe(texts, batch_size=batch_size, disable=["ner", "parser", "lemmatizer"]),
        total=len(texts),
        desc="spaCy",
        unit="comment",
    ):
        for tok in doc:
            if tok.is_space or tok.is_punct or not tok.text.strip():
                continue
            pos = tok.pos_
            if pos in SKIP_POS:
                continue
            word = tok.text.lower()
            if not word:
                continue
            by_pos[pos][word] += 1
    return dict(by_pos)


def lexicon_to_jsonable(by_pos: Dict[str, Counter]) -> dict:
    pos_blocks = {}
    total_tokens = 0
    for pos in sorted(by_pos):
        counts = by_pos[pos]
        total = int(sum(counts.values()))
        total_tokens += total
        # most common first — handy for inspection and for weighted sampling later
        words = [
            {"word": w, "count": int(c)}
            for w, c in counts.most_common()
        ]
        pos_blocks[pos] = {
            "n_types": len(words),
            "n_tokens": total,
            "words": words,
        }
    return {
        "created_at": _utc_now(),
        "skip_pos": sorted(SKIP_POS),
        "n_pos_tags": len(pos_blocks),
        "n_tokens": total_tokens,
        "by_pos": pos_blocks,
    }


def collect_files(sides: str, limit_files: int | None) -> List[Path]:
    """Gather query and/or candidate JSONL paths from data/splits/."""
    patterns: List[str] = []
    if sides in {"query", "both"}:
        patterns.append("user_*_query.jsonl")
    if sides in {"candidate", "both"}:
        patterns.append("user_*_candidate.jsonl")

    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(DATA_SPLITS.glob(pat)))
    files = sorted(set(files), key=lambda p: p.name)

    if not files:
        raise SystemExit(f"No files for sides={sides!r} in {DATA_SPLITS}")
    if limit_files is not None:
        # limit applies per side when both are selected
        if sides == "both":
            q = sorted(DATA_SPLITS.glob("user_*_query.jsonl"))[:limit_files]
            c = sorted(DATA_SPLITS.glob("user_*_candidate.jsonl"))[:limit_files]
            files = sorted(set(q + c), key=lambda p: p.name)
        else:
            files = files[:limit_files]
    return files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="So2 POS frequency lexicon from full splits corpus")
    p.add_argument("--model", default=DEFAULT_MODEL, help="spaCy model name")
    p.add_argument(
        "--sides",
        choices=("query", "candidate", "both"),
        default="both",
        help="which splits to include (default: both = full corpus)",
    )
    p.add_argument(
        "--limit-files",
        type=int,
        default=None,
        help="use only first N files per selected side",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "pos_lexicon.json",
        help="output JSON path",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import spacy
    except ImportError as e:
        raise SystemExit(
            "spaCy is not installed. In the project venv run:\n"
            "  pip install spacy && python -m spacy download en_core_web_lg"
        ) from e

    files = collect_files(args.sides, args.limit_files)
    n_query = sum(1 for p in files if p.name.endswith("_query.jsonl"))
    n_cand = sum(1 for p in files if p.name.endswith("_candidate.jsonl"))

    print(f"sides: {args.sides}")
    print(f"files: {len(files)} (query={n_query}, candidate={n_cand})")
    print(f"model: {args.model}")
    nlp = spacy.load(args.model)

    by_pos = build_lexicon(
        iter_query_bodies(files),
        nlp,
        batch_size=args.batch_size,
    )
    payload = lexicon_to_jsonable(by_pos)
    payload["spacy_model"] = args.model
    payload["sides"] = args.sides
    payload["n_files"] = len(files)
    payload["n_query_files"] = n_query
    payload["n_candidate_files"] = n_cand
    payload["corpus"] = (
        "data/splits/user_*_{query,candidate}.jsonl"
        if args.sides == "both"
        else f"data/splits/user_*_{args.sides}.jsonl"
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    print()
    print(f"tokens: {payload['n_tokens']}")
    print(f"POS tags: {payload['n_pos_tags']}")
    print(f"{'POS':<8} {'types':>8} {'tokens':>10}")
    print("-" * 30)
    for pos, block in payload["by_pos"].items():
        print(f"{pos:<8} {block['n_types']:>8} {block['n_tokens']:>10}")
    print("-" * 30)
    print(f"wrote → {args.out}")


if __name__ == "__main__":
    main()
