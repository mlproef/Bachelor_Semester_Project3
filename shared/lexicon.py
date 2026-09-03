"""Shared lexicon data classes only (no file I/O — loading stays in So2 scripts)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class LexiconEntry:
    """One word type inside a POS bucket."""

    id: int
    word: str
    count: int


@dataclass(frozen=True)
class PosBucket:
    """All lexicon entries for one part-of-speech tag."""

    pos: str
    entries: Tuple[LexiconEntry, ...]

    @property
    def n_types(self) -> int:
        return len(self.entries)

    @property
    def n_tokens(self) -> int:
        return sum(e.count for e in self.entries)

    @property
    def words(self) -> Tuple[str, ...]:
        return tuple(e.word for e in self.entries)

    @property
    def counts(self) -> Tuple[int, ...]:
        return tuple(e.count for e in self.entries)

    def entry_by_id(self, entry_id: int) -> LexiconEntry:
        return self.entries[entry_id]


@dataclass(frozen=True)
class PosLexicon:
    """Full POS lexicon built from count_spacy output (in memory)."""

    by_pos: Dict[str, PosBucket]
    path: Optional[Path] = None
    spacy_model: str = ""
    sides: str = ""
    n_tokens: int = 0

    @property
    def pos_tags(self) -> Tuple[str, ...]:
        return tuple(sorted(self.by_pos))

    def get(self, pos: str) -> Optional[PosBucket]:
        return self.by_pos.get(pos)
