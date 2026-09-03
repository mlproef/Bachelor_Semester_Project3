#!/usr/bin/env python3
"""Batch chunked anonymize over many users with resume.

Loops profiles in a jsonl file. For each user writes to
  <out-root>/<username>/
Skips users that are already complete. Incomplete users resume from the
last saved chunk. Safe to re-run after crashes / API outages.


"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_chunked_anonymize import (  # noqa: E402
    anonymize_one_user,
    user_is_complete,
)


def iter_profile_records(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)


def load_batch_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "users": {},
        "order": [],
    }


def save_batch_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_batch_log(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profiles",
        type=Path,
        default=REPO_ROOT / "data" / "profiles" / "profiles_query.jsonl",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "results_condition_c",
    )
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--model", type=str, default="qwen3.6-35b-a3b-nvfp4")
    parser.add_argument(
        "--limit-users",
        type=int,
        default=None,
        help="Process at most N users from the profiles file (in order).",
    )
    parser.add_argument(
        "--offset-users",
        type=int,
        default=0,
        help="Skip the first N users in the profiles file.",
    )
    parser.add_argument(
        "--usernames",
        type=str,
        default=None,
        help="Comma-separated usernames to run (optional filter).",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Recompute chunks even if present (still skips complete users unless set).",
    )
    parser.add_argument(
        "--redo-complete",
        action="store_true",
        help="Do not skip users marked complete.",
    )
    parser.add_argument("--limit-chunks", type=int, default=None)
    args = parser.parse_args()

    from src.anonymized.anonymizers.llm_anonymizers import LLMFullAnonymizer
    from src.configs import AnonymizerConfig, Config, ModelConfig, REDDITConfig, Task
    from src.configs.config import AnonymizationConfig
    from src.models.model_factory import get_model
    from src.prompts import Prompt
    from src.reddit.reddit import create_prompts, parse_answer
    from src.reddit.reddit_types import AnnotatedComments, Comment, Profile
    from src.utils.initialization import set_credentials

    args.out_root.mkdir(parents=True, exist_ok=True)
    state_path = args.out_root / "batch_state.json"
    log_path = args.out_root / "batch_log.jsonl"
    state = load_batch_state(state_path)

    wanted: Optional[set[str]] = None
    if args.usernames:
        wanted = {u.strip() for u in args.usernames.split(",") if u.strip()}

    records = list(iter_profile_records(args.profiles))
    if args.offset_users:
        records = records[args.offset_users :]
    if args.limit_users is not None:
        records = records[: args.limit_users]
    if wanted is not None:
        records = [r for r in records if r.get("username") in wanted]

    print(
        f"batch: {len(records)} users from {args.profiles} → {args.out_root} "
        f"(chunk={args.chunk_size}, iters={args.max_iterations})",
        flush=True,
    )

    cfg = Config(
        output_dir="results",
        seed=10,
        task=Task.ANONYMIZED,
        task_config=AnonymizationConfig(
            profile_path=str(args.profiles),
            outpath=str(args.out_root),
        ),
        gen_model=ModelConfig(
            name=args.model,
            provider="openai",
            args={
                "temperature": 0.1,
                "max_tokens": 8000,
                "request_timeout": 600,
                "max_retries": 8,
                "retry_base_delay": 5.0,
            },
        ),
    )
    set_credentials(cfg)
    model = get_model(
        ModelConfig(
            name=args.model,
            provider="openai",
            args={
                "temperature": 0.1,
                "max_tokens": 8000,
                "request_timeout": 600,
                "max_retries": 8,
                "retry_base_delay": 5.0,
            },
        )
    )
    anonymizer = LLMFullAnonymizer(
        AnonymizerConfig(
            anon_type="llm", prompt_level=3, target_mode="single", max_workers=1
        ),
        model,
    )
    reddit_cfg = REDDITConfig(
        path="unused",
        outpath="unused",
        profile_filter={"hardness": 1, "certainty": 1},
    )

    t_batch = time.time()
    n_done = 0
    n_skip = 0
    n_fail = 0

    for i, rec in enumerate(records):
        username = rec["username"]
        n_comments = len(rec.get("comments") or [])
        user_dir = args.out_root / username

        if username not in state["order"]:
            state["order"].append(username)

        if (not args.redo_complete) and user_is_complete(
            user_dir, n_comments, args.chunk_size
        ):
            print(
                f"\n>>> [{i+1}/{len(records)}] SKIP complete {username}",
                flush=True,
            )
            state["users"][username] = {
                "status": "complete",
                "out_dir": str(user_dir),
                "n_comments": n_comments,
                "skipped": True,
            }
            save_batch_state(state_path, state)
            n_skip += 1
            continue

        print(
            f"\n>>> [{i+1}/{len(records)}] START {username} "
            f"(comments={n_comments}) → {user_dir}",
            flush=True,
        )
        state["users"][username] = {
            "status": "running",
            "out_dir": str(user_dir),
            "n_comments": n_comments,
            "started_at": time.time(),
        }
        save_batch_state(state_path, state)

        t0 = time.time()
        try:
            manifest = anonymize_one_user(
                rec=rec,
                out_dir=user_dir,
                model=model,
                model_name=args.model,
                anonymizer=anonymizer,
                create_prompts=create_prompts,
                parse_answer=parse_answer,
                AnnotatedComments=AnnotatedComments,
                Comment=Comment,
                Profile=Profile,
                Prompt=Prompt,
                reddit_cfg=reddit_cfg,
                chunk_size=args.chunk_size,
                max_iterations=args.max_iterations,
                limit_chunks=args.limit_chunks,
                no_resume=args.no_resume,
            )
            elapsed = round(time.time() - t0, 3)
            row = {
                "username": username,
                "status": "complete",
                "elapsed_s": elapsed,
                "n_comments_out": manifest.get("n_comments_out"),
                "n_comments_changed": manifest.get("n_comments_changed"),
                "complete": manifest.get("complete"),
            }
            state["users"][username] = {
                **row,
                "out_dir": str(user_dir),
                "finished_at": time.time(),
            }
            append_batch_log(log_path, row)
            save_batch_state(state_path, state)
            n_done += 1
            print(
                f">>> [{i+1}/{len(records)}] OK {username} in {elapsed}s",
                flush=True,
            )
        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            err = f"{type(e).__name__}: {e}"
            print(
                f">>> [{i+1}/{len(records)}] FAIL {username} after {elapsed}s: {err}",
                flush=True,
            )
            traceback.print_exc()
            row = {
                "username": username,
                "status": "failed",
                "elapsed_s": elapsed,
                "error": err[:500],
            }
            state["users"][username] = {
                **row,
                "out_dir": str(user_dir),
                "failed_at": time.time(),
            }
            append_batch_log(log_path, row)
            save_batch_state(state_path, state)
            n_fail += 1
            # Continue to next user; re-run later will resume this user's chunks.
            continue

    summary = {
        "n_users_planned": len(records),
        "n_done": n_done,
        "n_skipped_complete": n_skip,
        "n_failed": n_fail,
        "total_elapsed_s": round(time.time() - t_batch, 3),
        "out_root": str(args.out_root),
        "profiles": str(args.profiles),
        "chunk_size": args.chunk_size,
        "max_iterations": args.max_iterations,
        "model": args.model,
    }
    (args.out_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("\n=== BATCH DONE ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()