"""Shared package: paths + data classes (I/O stays in So1/So2 scripts)."""

from shared.lexicon import LexiconEntry, PosBucket, PosLexicon
from shared.paths import (
    BSP3_SUMMER,
    DATA_SPLITS,
    EXPERIMENTS,
    PROJECT,
    REPO_ROOT,
    RESULTS_TABLES,
)
from shared.profiles import QueryCorpus, UserPair, UserProfile

__all__ = [
    "REPO_ROOT",
    "BSP3_SUMMER",
    "PROJECT",
    "DATA_SPLITS",
    "EXPERIMENTS",
    "RESULTS_TABLES",
    "UserPair",
    "UserProfile",
    "QueryCorpus",
    "LexiconEntry",
    "PosBucket",
    "PosLexicon",
]
