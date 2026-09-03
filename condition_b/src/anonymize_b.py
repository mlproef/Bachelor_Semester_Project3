"""Condition B — LLM generalization anonymization on query / candidate profiles.

Reads shared ``data/splits/user_*_{side}.jsonl`` and writes
``anonymized_b/query/`` or ``anonymized_b/candidates/``.

Uses the system prompt in ``prompts/condition_b_generalize.txt`` and an
OpenAI-compatible chat API (historically ``gpt-4o-mini``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from tqdm import tqdm

try:
    import httpx

    _HTTPX_NETWORK_ERRORS = (
        httpx.ConnectError,
        httpx.ConnectTimeout,
        httpx.ReadTimeout,
        httpx.RemoteProtocolError,
    )
except ImportError:  # pragma: no cover
    _HTTPX_NETWORK_ERRORS = ()

_OPENAI_REQUEST_TIMEOUT_S = 600.0
_OPENAI_MAX_RETRIES = 6

from src.paths import (
    ANONYMIZED_B,
    ANONYMIZED_B_CANDIDATES,
    ANONYMIZED_B_QUERY,
    DATA_SPLITS,
    PROMPTS_DIR,
)
from src.reddit_jsonl import comment_body

CONDITION_B_PROMPT_FILE = PROMPTS_DIR / "condition_b_generalize.txt"

_OUT_BY_SIDE = {
    "query": ANONYMIZED_B_QUERY,
    "candidate": ANONYMIZED_B_CANDIDATES,
}

# Model sometimes returns assistant meta/refusal instead of rewritten comment text.
_REFUSAL_MARKERS = (
    "i cannot",
    "as an ai",
    "as a language model",
    "i am designed",
    "i'm designed",
    "please provide",
    "does not contain any identifying",
    "cannot process",
    "cannot fulfill",
    "cannot engage",
    "cannot experience",
    "cannot access",
    "cannot rewrite",
    "violates safety",
    "safety guidelines",
    "i am here to help",
    "i'm here to help",
    "i can't process",
    "i can't fulfill",
    "i can't engage",
    "i can't access",
    "i can't rewrite",
)


# Reject assistant over-answers that balloon a short comment into an essay.
_OVEREXPAND_MIN_ABS = 400
_OVEREXPAND_RATIO = 3.0


def _looks_like_refusal(original: str, rewritten: str) -> bool:
    """True if model output is meta/refusal rather than a comment rewrite."""
    out = (rewritten or "").strip()
    if not out:
        return True
    src = (original or "").strip()
    if out == src:
        return False
    low = out.lower()
    if any(m in low for m in _REFUSAL_MARKERS):
        return True
    return False


def _looks_like_overexpand(original: str, rewritten: str) -> bool:
    """True if rewritten text is far longer than the original comment."""
    out = (rewritten or "").strip()
    src = (original or "").strip()
    if not out or out == src:
        return False
    limit = max(_OVEREXPAND_MIN_ABS, int(_OVEREXPAND_RATIO * len(src)))
    return len(out) > limit


def _client_settings(*, model: Optional[str] = None) -> Dict[str, str]:
    """Read OpenAI-compatible endpoint from the environment (no hardcoded secrets).

    Required: OLLAMA_URL or OPENAI_API_BASE
              OLLAMA_API_KEY or OPENAI_API_KEY
    Optional: pass ``model=``, or CONDITION_B_MODEL. Default: qwen3.5-4b
    (does not use OLLAMA_MODEL — that is often a larger Reason model).
    """
    base_url = (os.getenv("OLLAMA_URL") or os.getenv("OPENAI_API_BASE") or "").rstrip(
        "/"
    )
    api_key = os.getenv("OLLAMA_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
    model_name = model or os.getenv("CONDITION_B_MODEL") or "qwen3.5-4b"
    missing = []
    if not base_url:
        missing.append("OLLAMA_URL (or OPENAI_API_BASE)")
    if not api_key:
        missing.append("OLLAMA_API_KEY (or OPENAI_API_KEY)")
    if not model_name:
        missing.append("model (pass model= / --model or CONDITION_B_MODEL)")
    if missing:
        raise RuntimeError(
            "Missing API settings for Condition B: "
            + ", ".join(missing)
            + ". Set them in the environment (or a local .env; do not commit secrets)."
        )
    return {"base_url": base_url, "api_key": api_key, "model": model_name}


def _load_prompt(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {path}")
    return text


def _call_openai_rewrite(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    text: str,
    max_retries: int = _OPENAI_MAX_RETRIES,
) -> str:
    if not text.strip():
        return text

    retryable = (
        RateLimitError,
        APIError,
        APIConnectionError,
        APITimeoutError,
        *_HTTPX_NETWORK_ERRORS,
    )
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0.0,
                max_tokens=1200,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
            )
            choice = resp.choices[0].message
            out = (choice.content or "").strip()
            if out:
                return out
            refusal = getattr(choice, "refusal", None)
            last_err = RuntimeError(
                f"Empty OpenAI response (refusal={refusal!r}, "
                f"finish_reason={resp.choices[0].finish_reason!r})"
            )
        except retryable as e:
            last_err = e
            time.sleep(min(120.0, 2.0 ** (attempt + 1)))
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(30.0, 2.0 * (attempt + 1)))

    raise RuntimeError(
        f"OpenAI anonymization failed after {max_retries} tries: {last_err}"
    )


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


def _make_resilient_openai_anonymizer(
    client: OpenAI,
    *,
    model: str,
    system_prompt: str,
    errors: List[Dict[str, Any]],
    source_file: str,
    fallback_counter: List[int],
    refusal_counter: List[int],
    overexpand_counter: List[int],
) -> Callable[[str], str]:
    """Retry API calls; on failure / refusal / over-expand keep original text."""

    def anonymize(txt: str) -> str:
        try:
            out = _call_openai_rewrite(
                client, model=model, system_prompt=system_prompt, text=txt
            )
        except Exception as e:  # noqa: BLE001
            fallback_counter[0] += 1
            errors.append(
                {
                    "file": source_file,
                    "error": repr(e),
                    "text_preview": txt[:300],
                }
            )
            return txt

        if _looks_like_refusal(txt, out):
            refusal_counter[0] += 1
            errors.append(
                {
                    "file": source_file,
                    "error": "refusal_fallback",
                    "text_preview": txt[:300],
                    "model_preview": out[:300],
                }
            )
            return txt
        if _looks_like_overexpand(txt, out):
            overexpand_counter[0] += 1
            errors.append(
                {
                    "file": source_file,
                    "error": "overexpand_fallback",
                    "text_preview": txt[:300],
                    "model_preview": out[:300],
                    "src_len": len(txt),
                    "out_len": len(out),
                }
            )
            return txt
        return out

    return anonymize


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
        "api_calls": 0,
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
        text = comment_body(obj)
        if text:
            stats["api_calls"] += 1
        obj = _apply_anonymizer_to_obj(obj, anonymizer)
        out_lines.append(json.dumps(obj, ensure_ascii=True))
        _write_out_lines(dst_file, out_lines)

    return stats


def run_condition_b(
    *,
    side: str = "query",
    force: bool = False,
    dry_run: bool = False,
    limit_files: Optional[int] = None,
    limit_lines_per_file: Optional[int] = None,
    model: Optional[str] = None,
    sleep_s: float = 0.0,
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
            "condition": "B",
            "side": side,
            f"n_{side}_files": len(src_files),
            "estimated_comment_lines": est_lines,
            "model": model,
            "prompt_file": str(CONDITION_B_PROMPT_FILE),
            "output_dir": str(out_dir.resolve()),
        }
        man_path = ANONYMIZED_B / f"step4_B{man_suffix}_dry_run.json"
        man_path.write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
        )
        return out_dir

    settings = _client_settings(model=model)
    model_name = settings["model"]
    print(
        f"[B/{side}] API base_url={settings['base_url']} model={model_name} "
        f"out={out_dir}",
        flush=True,
    )
    client = OpenAI(
        api_key=settings["api_key"],
        base_url=settings["base_url"],
        timeout=_OPENAI_REQUEST_TIMEOUT_S,
        max_retries=2,
    )
    system_prompt = _load_prompt(CONDITION_B_PROMPT_FILE)
    api_errors: List[Dict[str, Any]] = []
    fallback_original = 0
    fallback_refusal = 0
    fallback_overexpand = 0
    total_lines = 0
    total_api = 0
    skipped_files = 0

    for sf in tqdm(src_files, desc=f"Anonymize B ({side})"):
        file_fallback = [0]
        file_refusal = [0]
        file_overexpand = [0]
        anonymizer = _make_resilient_openai_anonymizer(
            client,
            model=model_name,
            system_prompt=system_prompt,
            errors=api_errors,
            source_file=sf.name,
            fallback_counter=file_fallback,
            refusal_counter=file_refusal,
            overexpand_counter=file_overexpand,
        )
        stats = _process_profile(
            sf,
            out_dir / sf.name,
            anonymizer,
            force=force,
            limit_lines_per_file=limit_lines_per_file,
        )
        fallback_original += file_fallback[0]
        fallback_refusal += file_refusal[0]
        fallback_overexpand += file_overexpand[0]
        total_lines += stats["lines"]
        total_api += stats["api_calls"]
        skipped_files += stats["skipped_file"]
        if sleep_s > 0 and stats["api_calls"] > 0:
            time.sleep(sleep_s)

    manifest: Dict[str, Any] = {
        "dry_run": False,
        "condition": "B",
        "side": side,
        f"n_{side}_files": len(src_files),
        "processed_comment_lines": total_lines,
        "api_calls": total_api,
        "skipped_files": skipped_files,
        "fallback_original": fallback_original,
        "fallback_refusal": fallback_refusal,
        "fallback_overexpand": fallback_overexpand,
        "model": model_name,
        "base_url": settings["base_url"],
        "prompt_file": str(CONDITION_B_PROMPT_FILE.name),
        "output_dir": str(out_dir.resolve()),
    }
    if api_errors:
        manifest["api_errors_count"] = len(api_errors)
        err_path = ANONYMIZED_B / f"step4_B{man_suffix}_errors.json"
        err_path.write_text(
            json.dumps(api_errors[:500], ensure_ascii=True, indent=2), encoding="utf-8"
        )
        manifest["errors_file"] = str(err_path.resolve())
    man_path = ANONYMIZED_B / f"step4_B{man_suffix}_manifest.json"
    man_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8"
    )
    return out_dir
