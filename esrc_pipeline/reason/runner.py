"""Orchestrate ESRC Reason for one condition (baseline|A|B|C).

Flow (from esrc_stas phase_reason, adapted to github-summer)::

  1. load search_top15.csv → hits per query
  2. for each pending query (resume via reason_predictions.jsonl):
       - load query extract summary
       - load top-k candidate extract summaries
       - build_user_prompt(record_selection template, …)
       - generate(LLM) → JSON → resolve_predicted_candidate_id
       - correct = (pred_id == query_uid); true_in_top15 = …
  3. write reason_predictions.jsonl
  4. evaluate → reason_metrics.json
"""
from __future__ import annotations

import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from extract.paths import Condition
from reason.paths import (
    DEFAULT_REASON_MODEL,
    extract_candidate_dir,
    extract_query_dir,
    reason_metrics_json,
    reason_out_dir,
    reason_predictions_jsonl,
    record_selection_prompt,
    search_csv,
)
from search.paths import DEFAULT_TOP_K


def process_one_query(
    uid: str,
    *,
    condition: Condition,
    hits: list[dict[str, str]],
    query_dir: Path,
    cand_dir: Path,
    reason_tpl: str,
    model: str,
    k: int,
    max_tokens: int,
    timeout: float,
    seed: int,
    write_lock: threading.Lock,
    out_jsonl: Path,
) -> dict[str, Any]:
    """One query: Search hits → LLM pick → append JSONL row."""
    from extract.generate import (
        ContextLengthExceededError,
        PermanentRequestError,
        generate,
        get_client,
    )
    from reason.io import append_reason_row, load_summary_text
    from reason.prompt import (
        build_candidate_block,
        build_user_prompt,
        clamp01,
        extract_json_object,
        resolve_predicted_candidate_id,
    )

    t0 = time.perf_counter()
    rows = sorted(hits, key=lambda r: int(r["rank"]))
    row: dict[str, Any] = {
        "condition": condition,
        "query_user_id": uid,
        "user_id": uid,
        "status": "error",
        "error": "",
        "error_type": "",
        "model": model,
        "max_tokens": max_tokens,
        "k": k,
    }
    try:
        if len(rows) < k:
            print(
                f"[{condition}/reason] WARN {uid}: only {len(rows)} search hits "
                f"(expected {k})",
                flush=True,
            )
        q_text = load_summary_text(query_dir, uid)
        cands: list[dict[str, Any]] = []
        for r in rows[:k]:
            cid = r["candidate_user_id"]
            cands.append(
                {
                    "candidate_user_id": cid,
                    "rank": int(r["rank"]),
                    "score": float(r["score"]),
                    "summary": load_summary_text(cand_dir, cid),
                }
            )
        block = build_candidate_block(cands)
        user_prompt = build_user_prompt(reason_tpl, q_text, block)
        client = get_client(timeout=timeout, max_retries=0)
        result = generate(
            [{"role": "user", "content": user_prompt}],
            model=model,
            client=client,
            temperature=0.0,
            max_tokens=max_tokens,
            seed=seed,
            enable_thinking=False,
        )
        obj = extract_json_object(result.text)
        cand_ids = [c["candidate_user_id"] for c in cands]
        pred_id, pred_num, err = resolve_predicted_candidate_id(obj, cand_ids)
        if err:
            raise ValueError(err)
        row.update(
            {
                "status": "ok",
                "selected_candidate_user_id": pred_id,
                "selected_candidate_number": pred_num,
                "confidence": clamp01(obj.get("confidence")),
                "reasoning_short": str(obj.get("reasoning_short") or "")[:500],
                "correct": pred_id == uid,
                "true_in_top15": any(c["candidate_user_id"] == uid for c in cands),
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
    append_reason_row(out_jsonl, row, write_lock)
    print(
        f"[{condition}/reason] {row['status']} {uid} "
        f"pick={row.get('selected_candidate_number')} "
        f"correct={row.get('correct')} {row['latency_s']:.1f}s",
        flush=True,
    )
    return row


def run_reason(
    condition: Condition,
    *,
    model: str = DEFAULT_REASON_MODEL,
    k: int = DEFAULT_TOP_K,
    force: bool = False,
    resume: bool = True,
    dry_run: bool = False,
    limit_queries: int | None = None,
    concurrency: int = 2,
    max_tokens: int = 512,
    timeout: float = 600.0,
    seed: int = 0,
    gallery_condition: Condition | None = None,
) -> Path:
    """Run Reason for ``condition``; return path to reason_predictions.jsonl.

    ``gallery_condition`` selects whose Extract candidate summaries the model
    picks from (default: same as ``condition``). One-sided A vs raw::
      run_reason("A", gallery_condition="baseline")
    """
    from extract.resume import load_ok_user_ids, load_resume_skip_user_ids
    from reason.io import load_search_hits_by_query
    from reason.metrics import evaluate_reason, write_reason_metrics
    from reason.prompt import load_prompt_template

    if concurrency < 1:
        raise ValueError(f"concurrency must be >= 1, got {concurrency}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")

    gallery_condition = gallery_condition or condition

    out_dir = reason_out_dir(condition, gallery_condition=gallery_condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = reason_predictions_jsonl(
        condition, gallery_condition=gallery_condition
    )
    metrics_path = reason_metrics_json(
        condition, gallery_condition=gallery_condition
    )

    prompt_path = record_selection_prompt()
    reason_tpl = load_prompt_template(prompt_path)

    hits_path = search_csv(condition, k=k, gallery_condition=gallery_condition)
    by_query = load_search_hits_by_query(hits_path)
    users = sorted(by_query.keys())
    if limit_queries is not None:
        users = users[:limit_queries]

    if force or not resume:
        pending = list(users)
    else:
        skip = load_resume_skip_user_ids(out_jsonl)
        pending = [u for u in users if u not in skip]

    n_ok = len(load_ok_user_ids(out_jsonl)) if resume and out_jsonl.exists() else 0
    print(
        f"[{condition}/reason] {n_ok} ok cached, {len(pending)} pending "
        f"(K={concurrency}, model={model}, gallery={gallery_condition}) "
        f"out={out_jsonl}",
        flush=True,
    )

    if dry_run:
        print(f"[{condition}/reason] dry_run — not calling LLM", flush=True)
        return out_jsonl

    if pending:
        query_dir = extract_query_dir(condition)
        cand_dir = extract_candidate_dir(gallery_condition)
        write_lock = threading.Lock()

        def _one(uid: str) -> dict[str, Any]:
            return process_one_query(
                uid,
                condition=condition,
                hits=by_query[uid],
                query_dir=query_dir,
                cand_dir=cand_dir,
                reason_tpl=reason_tpl,
                model=model,
                k=k,
                max_tokens=max_tokens,
                timeout=timeout,
                seed=seed,
                write_lock=write_lock,
                out_jsonl=out_jsonl,
            )

        if concurrency <= 1:
            for uid in pending:
                _one(uid)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futs = [pool.submit(_one, uid) for uid in pending]
                for fut in as_completed(futs):
                    fut.result()

    if out_jsonl.exists():
        metrics = evaluate_reason(out_jsonl)
        write_reason_metrics(metrics_path, metrics)
        print(
            f"[{condition}/reason] metrics top1="
            f"{metrics['top1_correct']}/{metrics['n_reason_ok']} "
            f"({metrics['top1_accuracy']:.3f}) "
            f"hit@15={metrics['hit_at_15_count']}/{metrics['n_reason_ok']} "
            f"({metrics['hit_at_15']:.3f}) → {metrics_path}",
            flush=True,
        )

    return out_jsonl
