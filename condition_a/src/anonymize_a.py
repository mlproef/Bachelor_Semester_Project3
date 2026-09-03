"""Condition A — spaCy NER anonymization on query / candidate profiles.

Reads shared ``data/splits/user_*_{side}.jsonl`` and writes
``anonymized_a/query/`` or ``anonymized_a/candidates/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import spacy
from tqdm import tqdm

from src.paths import (
    ANONYMIZED_A,
    ANONYMIZED_A_CANDIDATES,
    ANONYMIZED_A_QUERY,
    DATA_SPLITS,
)
from src.reddit_jsonl import comment_body

_OUT_BY_SIDE = {
    "query": ANONYMIZED_A_QUERY,
    "candidate": ANONYMIZED_A_CANDIDATES,
}


def condition_a_ner(text: str, nlp) -> str:
    """Replace NER spans with tags (same logic as the original step4 A path)."""
    doc = nlp(text)
    spans = sorted(doc.ents, key=lambda s: s.start_char, reverse=True)
    out = text
    label_map = {
        "PERSON": "[PERSON]",
        "GPE": "[LOCATION]",
        "LOC": "[LOCATION]",
        "ORG": "[ORGANIZATION]",
        "DATE": "[DATE]",
    }
    for sp in spans:
        replacement = label_map.get(sp.label_, f"[{sp.label_}]")
        out = out[: sp.start_char] + replacement + out[sp.end_char :]
    return out


def _iter_src_objects(
    src_file: Path,
    *,
    limit_lines_per_file: Optional[int] = None,
) -> List[dict]:
    objs: List[dict] = []
    for line_idx, line in enumerate(
        src_file.read_text(encoding="utf-8", errors="ignore").splitlines()
    ):
        if limit_lines_per_file is not None and line_idx >= limit_lines_per_file:
            break
        if not line.strip():
            continue
        try:
            objs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return objs


def _write_out_lines(dst_file: Path, out_lines: List[str]) -> None:
    dst_file.write_text(
        "\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8"
    )


def _apply_anonymizer_to_obj(obj: dict, anonymizer: Callable[[str], str]) -> dict:
    text = comment_body(obj)
    if not text:
        return obj
    new_text = anonymizer(text)
    if "body" in obj:
        obj["body"] = new_text
    if "b" in obj:
        obj["b"] = new_text
    if "body" not in obj and "b" not in obj:
        obj["body"] = new_text
    return obj


def _process_profile(
    src_file: Path,
    dst_file: Path,
    anonymizer: Callable[[str], str],
    *,
    force: bool,
    limit_lines_per_file: Optional[int] = None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "lines": 0,
        "skipped_file": 0,
        "resumed_file": 0,
    }
    src_objs = _iter_src_objects(src_file, limit_lines_per_file=limit_lines_per_file)
    if not src_objs:
        return stats

    out_lines: List[str] = []
    start_idx = 0
    if dst_file.is_file() and not force:
        existing = [
            ln
            for ln in dst_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        ]
        if len(existing) >= len(src_objs):
            stats["skipped_file"] = 1
            return stats
        if existing:
            out_lines = existing
            start_idx = len(existing)
            stats["resumed_file"] = 1

    for obj in src_objs[start_idx:]:
        stats["lines"] += 1
        obj = _apply_anonymizer_to_obj(obj, anonymizer)
        out_lines.append(json.dumps(obj, ensure_ascii=True))
        _write_out_lines(dst_file, out_lines)

    return stats


def run_condition_a(
    *,
    side: str = "query",
    force: bool = False,
    dry_run: bool = False,
    limit_files: Optional[int] = None,
    limit_lines_per_file: Optional[int] = None,
    spacy_model: str = "en_core_web_lg",
) -> Path:
    if side not in _OUT_BY_SIDE:
        raise ValueError(f"side must be 'query' or 'candidate', got {side!r}")

    out_dir = _OUT_BY_SIDE[side]
    out_dir.mkdir(parents=True, exist_ok=True)

    src_files = sorted(DATA_SPLITS.glob(f"user_*_{side}.jsonl"))
    if not src_files:
        raise FileNotFoundError(
            f"No user_*_{side}.jsonl in {DATA_SPLITS}. "
            "Expected shared POOL-EN data at repo data/splits/."
        )
    if limit_files is not None:
        src_files = src_files[:limit_files]

    man_suffix = "" if side == "query" else f"_{side}"

    if dry_run:
        est_lines = 0
        for sf in src_files:
            for line in sf.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.strip():
                    try:
                        json.loads(line)
                        est_lines += 1
                    except json.JSONDecodeError:
                        pass
        manifest = {
            "dry_run": True,
            "condition": "A",
            "side": side,
            f"n_{side}_files": len(src_files),
            "estimated_comment_lines": est_lines,
            "spacy_model": spacy_model,
        }
        man_path = ANONYMIZED_A / f"step4_A{man_suffix}_dry_run.json"
        man_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        return out_dir

    nlp = spacy.load(spacy_model)
    anonymizer: Callable[[str], str] = lambda txt: condition_a_ner(txt, nlp)

    total_lines = 0
    skipped_files = 0
    for sf in tqdm(src_files, desc=f"Anonymize A ({side})"):
        stats = _process_profile(
            sf,
            out_dir / sf.name,
            anonymizer,
            force=force,
            limit_lines_per_file=limit_lines_per_file,
        )
        total_lines += stats["lines"]
        skipped_files += stats["skipped_file"]

    manifest = {
        "dry_run": False,
        "condition": "A",
        "side": side,
        f"n_{side}_files": len(src_files),
        "processed_comment_lines": total_lines,
        "skipped_files": skipped_files,
        "model": f"spacy/{spacy_model}",
        "output_dir": str(out_dir.resolve()),
    }
    man_path = ANONYMIZED_A / f"step4_A{man_suffix}_manifest.json"
    man_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return out_dir
