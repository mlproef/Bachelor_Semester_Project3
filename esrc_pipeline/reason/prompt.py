"""Reason-stage prompt building (Lermen record selection) + JSON parse helpers.

Ported from esrc_stas/reason_prompt.py (no pool hardcoding).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional


JSON_OUTPUT_INSTRUCTIONS = """

Return ONLY valid JSON with this schema:
{
  "selected_candidate_number": 1,
  "confidence": 0.0,
  "reasoning_short": "brief explanation"
}

Rules:
- selected_candidate_number must be an integer from 1 to the number of candidates listed above.
- Choose by the [number] label only; do not invent candidate_user_id values.
- confidence must be between 0 and 1.
"""


def load_prompt_template(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_candidate_block(candidates: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for i, c in enumerate(candidates, start=1):
        cid = str(c["candidate_user_id"])
        rank = int(c["rank"])
        score = float(c["score"])
        summary = str(c["summary"])
        parts.append(
            f"[{i}] candidate_user_id: {cid}\n"
            f"    rank: {rank}\n"
            f"    score: {score:.6f}\n"
            f"    summary: {summary}\n"
        )
    return "\n".join(parts).strip()


def build_user_prompt(template: str, query_summary: str, candidate_block: str) -> str:
    prompt = template
    if "{query_summary}" in prompt:
        prompt = prompt.replace("{query_summary}", query_summary)
    else:
        prompt = prompt.rstrip() + "\n\nQUERY:\n" + query_summary

    if "{candidate_block}" in prompt:
        prompt = prompt.replace("{candidate_block}", candidate_block)
    else:
        prompt = prompt.rstrip() + "\n\nCANDIDATES:\n" + candidate_block

    return prompt.rstrip() + JSON_OUTPUT_INSTRUCTIONS


def _salvage_json_fields(text: str) -> Optional[dict[str, Any]]:
    """Recover the decision fields from JSON the parser cannot load.

    Covers two real failure modes seen on qwen3.6-35b: output truncated
    mid-``reasoning_short`` (no closing brace), and raw newlines inside a
    string. The verdict itself precedes the prose, so it survives both.
    """
    num = re.search(r'"selected_candidate_number"\s*:\s*"?(\d+)"?', text)
    if not num:
        return None
    obj: dict[str, Any] = {"selected_candidate_number": int(num.group(1))}
    conf = re.search(r'"confidence"\s*:\s*([0-9]*\.?[0-9]+)', text)
    if conf:
        obj["confidence"] = float(conf.group(1))
    reason = re.search(r'"reasoning_short"\s*:\s*"(.*)', text, re.S)
    if reason:
        obj["reasoning_short"] = reason.group(1).rstrip('"} \n')
    obj["salvaged_from_malformed_json"] = True
    return obj


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if not text:
        raise ValueError("Empty model content (possible thinking-only response).")
    for candidate in (text, None):
        if candidate is None:
            m = re.search(r"\{[\s\S]*\}", text)
            if not m:
                break
            candidate = m.group(0)
        for strict in (True, False):
            try:
                obj = json.loads(candidate, strict=strict)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
            raise ValueError("Model returned non-object JSON.")

    salvaged = _salvage_json_fields(text)
    if salvaged is not None:
        return salvaged
    raise ValueError(f"No JSON object found in model output: {text[:200]!r}")


def parse_candidate_number(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    s = str(value).strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    m = re.match(r"^(?:candidate\s*)?(\d+)$", s, re.I)
    if m:
        return int(m.group(1))
    return None


def resolve_predicted_candidate_id(
    out_obj: dict[str, Any],
    candidate_ids: list[str],
) -> tuple[str, int, str]:
    """Returns (candidate_user_id, 1-based number, error)."""
    n = len(candidate_ids)
    if n == 0:
        return "", -1, "no candidates provided"
    num = parse_candidate_number(out_obj.get("selected_candidate_number"))
    if num is None:
        return "", -1, "missing selected_candidate_number"
    if not (1 <= num <= n):
        return "", num, f"selected_candidate_number out of range: {num}"
    return candidate_ids[num - 1], num, ""


def clamp01(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, v))
