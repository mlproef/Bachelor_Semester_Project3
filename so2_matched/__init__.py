"""So2 — matched POS degradation control (null defence)."""

from so2_matched.src.random_from_spacy import run_matched
from so2_matched.src.paths import DEFAULT_LEXICON, OUT_DIRS, RATES

__all__ = ["run_matched", "DEFAULT_LEXICON", "OUT_DIRS", "RATES"]
