"""Common candidate anonymization pipeline (So3).

Stubs only — fill implementations one function at a time.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def comment_body(obj: dict) -> str:
    """Return comment text from ``body`` or compact ``b``."""
    ...


def list_candidate_files(*, limit_users: Optional[int] = None) -> List[Path]:
    """Sorted ``user_*_candidate.jsonl`` from data/splits."""
    ...


def iter_src_objects(
    src_file: Path,
    *,
    limit_lines: Optional[int] = None,
) -> List[dict]:
    """Parse JSON objects from one candidate JSONL."""
    ...


def apply_anonymizer_to_obj(
    obj: dict,
    anonymizer: Callable[[str], str],
) -> dict:
    """Replace comment text in-place via anonymizer; return the object."""
    ...


def process_candidate_file(
    src_file: Path,
    dst_file: Path,
    anonymizer: Callable[[str], str],
    *,
    force: bool = False,
    limit_lines: Optional[int] = None,
) -> Dict[str, Any]:
    """Anonymize one candidate file (resume / force). Return simple stats."""
    ...


def run_candidate_anonymization(
    *,
    condition: str,
    make_anonymizer: Callable[[str], Callable[[str], str]],
    force: bool = False,
    limit_users: Optional[int] = None,
    limit_lines: Optional[int] = None,
    dry_run: bool = False,
) -> Path:
    """Loop all candidate files for a condition; write manifest; return out dir."""
    ...
