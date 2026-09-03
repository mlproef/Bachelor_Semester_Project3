"""So2 paths: lexicon input + matched fake-anonymization outputs."""
from __future__ import annotations

from pathlib import Path

from shared.paths import REPO_ROOT

SO2_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_LEXICON = SO2_ROOT / "lexicon" / "result" / "pos_lexicon.json"

# Token-change rates matched to So1 (so1_bge/results/{a,b,c}/token_change_*.json).
# B is the current qwen anonymization, not the old gpt-4o-mini 0.306.
RATES = {
    "A": 0.08480899809946209,
    "B": 0.10933542232536979,
    "C": 0.13803875679535127,
}

# Fake anonymized query JSONL → so2_matched/results/{a,b,c}/
RESULTS_ROOT = SO2_ROOT / "results"
OUT_DIRS = {
    "A": RESULTS_ROOT / "a",
    "B": RESULTS_ROOT / "b",
    "C": RESULTS_ROOT / "c",
}

SPACY_MODEL = "en_core_web_lg"


def out_dir(match: str) -> Path:
    m = match.upper()
    if m not in OUT_DIRS:
        raise ValueError(f"match must be A, B, or C, got {match!r}")
    return OUT_DIRS[m]
