"""Batch Extract driver — generic over condition and side query|candidate.

Not hardcoded to one method: scripts/run_extract_{a,b,c,so2}.py only pass condition=.

Output layout:
  results/{baseline|a|b|c|so2_a|so2_b|so2_c}/extract/{query|candidates}/{user_id}.txt
  results/{cond}/extract/{side}_meta.jsonl
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from extract.paths import SIDE_SUBDIRS, Condition, Side


def meta_path(out_dir: Path, side: Side) -> Path:
    """Path to append-only extract meta jsonl for this side.

    ``out_dir`` is ``…/extract/{query|candidates}/``;
    meta sits beside it: ``…/extract/{query|candidates}_meta.jsonl``.
    """
    if side not in SIDE_SUBDIRS:
        raise ValueError(f"Unknown side {side!r}; expected query|candidate")
    return out_dir.parent / f"{SIDE_SUBDIRS[side]}_meta.jsonl"


def load_prompts() -> tuple[str, str]:
    """Load (summarization_template, merge_template) from prompts/."""
    from extract.paths import extract_merge_prompt, summarization_prompt

    sum_path = summarization_prompt()
    merge_path = extract_merge_prompt()
    if not sum_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {sum_path}")
    if not merge_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {merge_path}")
    return (
        sum_path.read_text(encoding="utf-8"),
        merge_path.read_text(encoding="utf-8"),
    )


def pending_profile_files(
    condition: Condition,
    side: Side,
    *,
    meta: Path,
    resume: bool,
    force: bool,
    limit_files: Optional[int],
) -> List[Path]:
    """List anonymized JSONL still to process (apply resume / force / limit)."""
    from extract.profile_io import list_profile_files, user_id_from_path
    from extract.resume import load_resume_skip_user_ids

    files = list_profile_files(condition, side, limit_files=limit_files)
    if force or not resume:
        return files

    skip = load_resume_skip_user_ids(meta)
    if not skip:
        return files
    return [p for p in files if user_id_from_path(p) not in skip]


def append_meta_row(path: Path, row: dict, lock: threading.Lock) -> None:
    """Thread-safe append one JSON object as a line."""
    import json
    import os

    line = json.dumps(row, ensure_ascii=True) + "\n"
    with lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def process_one_profile(
    src_file: Path,
    *,
    out_dir: Path,
    condition: Condition,
    side: Side,
    model: str,
    extract_tpl: str,
    merge_tpl: str,
    max_tokens: int,
    timeout: float,
    write_lock: threading.Lock,
    meta: Path,
) -> Dict[str, Any]:
    """
    One user: load text → extract_profile → write {user_id}.txt + meta row.

    Catch permanent LLM errors as status=permanent_error; other errors as error.
    """
    import time
    import traceback

    from extract.extract_profile import extract_profile
    from extract.generate import (
        ContextLengthExceededError,
        PermanentRequestError,
        get_client,
    )
    from extract.profile_io import load_profile_text, user_id_from_path

    uid = user_id_from_path(src_file)
    t0 = time.perf_counter()
    row: Dict[str, Any] = {
        "user_id": uid,
        "query_user_id": uid,
        "condition": condition,
        "side": side,
        "model": model,
        "status": "error",
        "error": "",
        "error_type": "",
    }
    try:
        text = load_profile_text(src_file)
        row["n_words"] = len(text.split())
        client = get_client(timeout=timeout, max_retries=0)
        result = extract_profile(
            text,
            client=client,
            model=model,
            extract_template=extract_tpl,
            merge_template=merge_tpl,
            max_tokens=max_tokens,
            enable_thinking=False,
            seed=None,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{uid}.txt").write_text(result.summary + "\n", encoding="utf-8")
        row.update(
            {
                "status": "ok",
                "chunked": result.chunked,
                "n_chunks": result.n_chunks,
                "method": result.method,
                "chunk_sizes": list(result.chunk_sizes),
                "probe_tokens_full": result.probe_tokens_full,
                "probe_tokens_per_chunk": list(result.probe_tokens_per_chunk),
                "summary_chars": len(result.summary),
            }
        )
    except (ContextLengthExceededError, PermanentRequestError) as exc:
        row["status"] = "permanent_error"
        row["error_type"] = type(exc).__name__
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback_tail"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()
    except Exception as exc:  # noqa: BLE001
        row["error_type"] = type(exc).__name__
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback_tail"] = "".join(
            traceback.format_exception_only(type(exc), exc)
        ).strip()

    row["latency_s"] = round(time.perf_counter() - t0, 3)
    append_meta_row(meta, row, write_lock)
    print(
        f"[{condition}/{side}] extract {row['status']} {uid} "
        f"chunked={row.get('chunked')} n_chunks={row.get('n_chunks')} "
        f"{row['latency_s']:.1f}s",
        flush=True,
    )
    return row


def run_extract(
    *,
    condition: Condition,
    side: Side = "query",
    model: str = "qwen3.5-4b",
    force: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    limit_files: Optional[int] = None,
    concurrency: int = 2,
    max_tokens: int = 1024,
    timeout: float = 600.0,
) -> Path:
    """
    Drive Extract for one condition + side. Return out_dir.

    Steps:
      1. extract_out_dir + mkdir
      2. load_prompts
      3. pending_profile_files
      4. dry_run → print counts and return
      5. process_one_profile for each (sequential or ThreadPool)
      6. return out_dir
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from extract.paths import extract_out_dir
    from extract.resume import load_ok_user_ids

    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")

    out_dir = extract_out_dir(condition, side)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = meta_path(out_dir, side)

    extract_tpl, merge_tpl = load_prompts()
    pending = pending_profile_files(
        condition,
        side,
        meta=meta,
        resume=resume,
        force=force,
        limit_files=limit_files,
    )
    n_ok = len(load_ok_user_ids(meta)) if resume and meta.exists() else 0
    print(
        f"[{condition}/{side}] extract: {n_ok} ok cached, "
        f"{len(pending)} pending (K={concurrency}) out={out_dir}",
        flush=True,
    )

    if dry_run:
        print(f"[{condition}/{side}] dry_run — not calling LLM", flush=True)
        return out_dir

    if not pending:
        return out_dir

    write_lock = threading.Lock()

    def _one(src: Path) -> Dict[str, Any]:
        return process_one_profile(
            src,
            out_dir=out_dir,
            condition=condition,
            side=side,
            model=model,
            extract_tpl=extract_tpl,
            merge_tpl=merge_tpl,
            max_tokens=max_tokens,
            timeout=timeout,
            write_lock=write_lock,
            meta=meta,
        )

    if concurrency <= 1:
        for src in pending:
            _one(src)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futs = [pool.submit(_one, src) for src in pending]
            for fut in as_completed(futs):
                fut.result()

    return out_dir
