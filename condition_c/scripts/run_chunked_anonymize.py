#!/usr/bin/env python3
"""Chunked Staab infer→anonymize without modifying core anonymized.py.

Splits a Profile into comment chunks (default 20). For each chunk runs up to
max_iterations of: inference → anonymize. Saves each chunk to disk immediately
and can resume skipped completed chunks.

Anonymize keeps the original Staab prompt style (# after a brief explanation)
but numbers comments (0:, 1:, ...) so pairs can be realigned by id if the model
reorders lines. Falls back to Staab filter_and_align_comments if numbering fails.

Example (from condition_c/):
  python scripts/run_chunked_anonymize.py \\
    --chunk-size 20 --max-iterations 3
  # defaults: ../data/profiles/profiles_query_one_full.jsonl
  #         → results_condition_c/user_smoke/
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent

STAAB_PII_TYPES = (
    "age",
    "gender",
    "location",
    "pobp",
    "education",
    "occupation",
    "married",
    "income",
)


def staab_reviews() -> dict[str, Any]:
    return {
        "pool_en": {
            pii: {
                "estimate": "unknown",
                "detect_from_subreddit": False,
                "hardness": 1,
                "certainty": 5,
            }
            for pii in STAAB_PII_TYPES
        }
    }


def chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield i, items[i : i + size]


def load_profile_record(path: Path, username: Optional[str] = None) -> dict:
    chosen = None
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if username is None:
                return obj
            if obj.get("username") == username:
                return obj
            if chosen is None:
                chosen = obj
    if username:
        raise SystemExit(f"username {username!r} not found in {path}")
    if chosen is None:
        raise SystemExit(f"no profiles in {path}")
    return chosen


def build_chunk_profile(
    *,
    username: str,
    comment_dicts: list[dict],
    Comment,
    AnnotatedComments,
    Profile,
):
    comments = [
        Comment(
            c["text"],
            c.get("subreddit", "reddit"),
            c.get("user", username),
            str(c.get("timestamp", "0")),
        )
        for c in comment_dicts
    ]
    reviews = staab_reviews()
    annotated = AnnotatedComments(comments=comments, review_pii=reviews, predictions={})
    return Profile(
        username=username,
        annotated_comments=[annotated],
        review_pii=reviews,
    )


def chunk_path(out_dir: Path, chunk_idx: int) -> Path:
    return out_dir / "chunks" / f"chunk_{chunk_idx:04d}.json"


def save_chunk(out_dir: Path, payload: dict) -> Path:
    path = chunk_path(out_dir, int(payload["chunk_idx"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    # Append/update meta index
    meta_path = out_dir / "chunks.jsonl"
    meta = {
        "chunk_idx": payload["chunk_idx"],
        "offset": payload["offset"],
        "n_comments": payload["n_comments"],
        "rounds": payload["rounds"],
        "elapsed_s": payload["elapsed_s"],
        "n_changed": payload.get("n_changed", 0),
        "path": str(path.relative_to(out_dir)),
    }
    with meta_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return path


def load_saved_chunk(out_dir: Path, chunk_idx: int) -> Optional[dict]:
    path = chunk_path(out_dir, chunk_idx)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


_BRACKET_TAG_RE = re.compile(r"\[([^\]\n]{1,40})\]")


def has_ner_style_tags(text: str) -> bool:
    """True if text contains bracket placeholders like [Location], [City]."""
    for m in _BRACKET_TAG_RE.finditer(text or ""):
        inner = m.group(1).strip()
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9 _/.-]{0,38}", inner):
            return True
    return False


def postfilter_remove_bracket_tags(
    *,
    model: Any,
    Prompt: Any,
    original: str,
    anonymized: str,
) -> str:
    """Extra LLM call: rewrite without [tags], keep natural generalizations."""
    system = (
        "You are an expert anonymizer. Rewrite text so it stays anonymous and readable. "
        "Replace any bracket placeholders such as [Location], [City], [Name], [Parent], "
        "[PERSON], [OCCUPATION], [Series Name], [Country] with natural-language "
        "generalizations. Do not invent new facts. Keep the rest of the wording as close "
        "as possible. Never output bracket tags."
    )
    user = (
        "The anonymized text below still contains bracket placeholders. "
        "Rewrite it without any [tags]. Use plain generalizations "
        "(e.g. '[City]' → 'a large city', '[Name]' → 'a person', "
        "'[Country]' → 'a country').\n\n"
        f"Original text:\n{original}\n\n"
        f"Anonymized text with tags:\n{anonymized}\n\n"
        "In a first step briefly explain. Then on a new line write a single # "
        "and return only the cleaned anonymized text."
    )
    prompt = Prompt(
        system_prompt=system,
        header="",
        intermediate=user,
        footer="",
        target="",
        original_point=None,
        gt=[],
        answer="",
        shots=[],
        id="postfilter",
    )
    raw = model.predict(prompt)
    if "\n#" in raw:
        cleaned = raw.split("\n#", 1)[1].strip()
    elif raw.strip().startswith("#"):
        cleaned = raw.strip().lstrip("#").strip()
    else:
        cleaned = raw.strip()
    if has_ner_style_tags(cleaned) and not has_ner_style_tags(anonymized):
        return anonymized
    return cleaned if cleaned else anonymized


def create_numbered_anon_prompt(*, profile, Prompt, model_name: str):
    """Staab prompt_level=3 style, but comments numbered 0:, 1:, ..."""
    comments = profile.get_latest_comments().comments
    n = len(comments)
    # Numbered plain texts (same content as Staab, plus id prefix).
    comment_string = "\n".join(f"{i}: {c.text}" for i, c in enumerate(comments))

    previous_inferences = profile.get_latest_comments().predictions[model_name]
    inference_string = ""
    try:
        for key, inf in previous_inferences.items():
            if key == "full_answer":
                continue
            if "guess" not in inf:
                continue
            inference_string += f"Type: {key}\n"
            inference_string += f"Inference: {inf['inference']}\n"
            inference_string += f"Guess: {inf['guess']}\n"
    except Exception:
        inference_string = previous_inferences.get("full_answer", "")

    # Same wording as LLMFullAnonymizer prompt_level=3, plus keep numbers.
    system_prompt = (
        "You are an expert anonymizer that carefully adapts small parts of the text "
        "to make it anonymous. You follow the instructions and format precisely and "
        "you try to change as little as possible, keeping the original text in tact "
        "as much as possible."
    )
    header = (
        "Below I give you a set of comments from a user as well as personal inferences "
        "made about the user that can be made from the comments. Your task is to "
        "anonymize the comments to prevent these inferences, while keeping the rest "
        "of the text intact."
    )
    footer = (
        "In a first step briefly explain the changes you will make. "
        "In a new line write a single # and then return the anonymized text. "
        "Only change relevant parts and keep everything else as is. "
        "Make sure you keep the original meaning, i.e. do not invent new information. "
        "Never use bracket placeholders like [City], [Location], [Person], [Parent], "
        "[Occupation], or similar tags. "
        "Always use natural-language generalizations "
        '(e.g. "Sydney" → "a large city", not "[City]"). '
        f"Keep the same numbering: each anonymized line must start with its id "
        f"(0: ... through {n - 1}: ...), one comment per line."
    )
    intermediate = f"\n\n {comment_string}\n\nInferences:\n\n{inference_string}"

    return Prompt(
        system_prompt=system_prompt,
        header=header,
        intermediate=intermediate,
        footer=footer,
        target=(
            profile.get_relevant_pii()[0] if len(profile.get_relevant_pii()) > 0 else ""
        ),
        original_point=profile,
        gt=profile.get_relevant_pii(),
        answer="",
        shots=[],
        id=profile.username,
    )


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.\)]\s*(.*)$")


def parse_numbered_anon_texts(answer: str, n: int) -> Optional[list[str]]:
    """Parse '#\\n0: ...\\n1: ...' (order-independent) → list length n by id."""
    raw = (answer or "").strip()
    if not raw:
        return None

    if "\n#" in raw:
        body = raw.split("\n#", 1)[1].strip()
    elif raw.startswith("#"):
        body = raw.lstrip("#").strip()
    else:
        body = raw

    by_id: dict[int, str] = {}
    for line in body.split("\n"):
        line = line.strip()
        if not line:
            continue
        m = _NUMBERED_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        if re.search(r"\d{4}-\d{2}-\d{2}:", text[:11]) is not None:
            text = text[11:].strip()
        if 0 <= idx < n:
            by_id[idx] = text

    if not all(i in by_id for i in range(n)):
        return None
    return [by_id[i] for i in range(n)]


def texts_to_comments(texts: list[str], old_comments, Comment) -> list:
    typed = []
    for i, text in enumerate(texts):
        text = strip_numbered_prefix(text)
        if re.search(r"\d{4}-\d{2}-\d{2}:", text[:11]) is not None:
            text = text[11:].strip()
        old = old_comments[i]
        typed.append(Comment(text, old.subreddit, old.user, old.timestamp))
    return typed


def run_chunk(
    *,
    profile,
    model,
    model_name: str,
    anonymizer,
    create_prompts,
    parse_answer,
    AnnotatedComments,
    Comment,
    Prompt,
    reddit_cfg,
    max_iterations: int,
    chunk_idx: int,
) -> dict[str, Any]:
    t0 = time.time()
    originals = [c.text for c in profile.get_latest_comments().comments]
    rounds = 0
    postfilter_n = 0
    numbered_ok_n = 0
    numbered_fallback_n = 0
    for round_idx in range(1, max_iterations + 1):
        rounds = round_idx
        prompts = create_prompts(profile, reddit_cfg)
        if not prompts:
            raise RuntimeError(f"chunk {chunk_idx}: create_prompts returned empty")
        answer = model.predict(prompts[0])
        parsed = parse_answer(answer, prompts[0].gt)
        parsed["full_answer"] = answer
        profile.get_latest_comments().predictions[model_name] = parsed

        before_texts = [c.text for c in profile.get_latest_comments().comments]
        anon_prompt = create_numbered_anon_prompt(
            profile=profile, Prompt=Prompt, model_name=model_name
        )
        anon_answer = model.predict(anon_prompt)
        n = len(before_texts)
        numbered_texts = parse_numbered_anon_texts(anon_answer, n)
        if numbered_texts is not None:
            typed = texts_to_comments(
                numbered_texts, profile.get_latest_comments().comments, Comment
            )
            numbered_ok_n += 1
        else:
            print(
                f"chunk {chunk_idx} round {round_idx}: numbered parse failed, "
                "fallback to filter_and_align_comments",
                flush=True,
            )
            typed = anonymizer.filter_and_align_comments(anon_answer, profile)
            # Fallback may keep '0: text' in the body — strip ids.
            for com in typed:
                com.text = strip_numbered_prefix(com.text)
            numbered_fallback_n += 1
        profile.comments.append(AnnotatedComments(typed, profile.review_pii, {}, {}))

        # Post-filter: remove NER-style [tags] if the model emitted them.
        latest = profile.get_latest_comments().comments
        for i, com in enumerate(latest):
            if not has_ner_style_tags(com.text):
                continue
            orig = before_texts[i] if i < len(before_texts) else originals[i]
            cleaned = postfilter_remove_bracket_tags(
                model=model,
                Prompt=Prompt,
                original=orig,
                anonymized=com.text,
            )
            postfilter_n += 1
            if cleaned.strip() and cleaned.strip() != com.text.strip():
                com.text = cleaned

    finals = [c.text for c in profile.get_latest_comments().comments]
    pairs = [
        {"i": i, "before": a, "after": b, "changed": a.strip() != b.strip()}
        for i, (a, b) in enumerate(zip(originals, finals))
    ]
    return {
        "chunk_idx": chunk_idx,
        "rounds": rounds,
        "n_comments": len(finals),
        "elapsed_s": round(time.time() - t0, 3),
        "final_texts": finals,
        "pairs": pairs,
        "n_changed": sum(1 for p in pairs if p["changed"]),
        "postfilter_n": postfilter_n,
        "n_brackets_left": sum(1 for t in finals if has_ner_style_tags(t)),
        "numbered_ok_n": numbered_ok_n,
        "numbered_fallback_n": numbered_fallback_n,
    }


_NUMBERED_PREFIX_RE = re.compile(r"^\s*\d+\s*[:.\)]\s*")


def strip_numbered_prefix(text: str) -> str:
    """Remove leftover '0: ' / '12: ' prefixes from anonymized text."""
    prev = None
    out = text or ""
    # Peel repeated layers in case a fallback left numbers and a later round
    # only stripped the outer id.
    while prev != out:
        prev = out
        out = _NUMBERED_PREFIX_RE.sub("", out, count=1).strip()
    return out


def canonicalize_chunk_alignment(
    part_texts: list[str],
    pairs: list[dict],
    fallback_anonymized: Optional[list[str]] = None,
) -> tuple[list[str], list[dict], int]:
    """Align anonymized texts to source order using pairs.before → pairs.after.

    `originals[i]` / `anonymized[i]` arrays can drift; `pairs` holds the true
    before/after mapping. Returns (anonymized_in_part_order, pairs_in_part_order,
    n_unmatched).
    """
    after_by_before: dict[str, str] = {}
    for p in pairs or []:
        before = p.get("before")
        after = p.get("after")
        if before is None or after is None:
            continue
        after_by_before[before] = strip_numbered_prefix(str(after))

    anonymized: list[str] = []
    aligned_pairs: list[dict] = []
    unmatched = 0
    for i, original in enumerate(part_texts):
        if original in after_by_before:
            after = after_by_before[original]
        elif fallback_anonymized is not None and i < len(fallback_anonymized):
            after = strip_numbered_prefix(fallback_anonymized[i])
            unmatched += 1
        else:
            after = original
            unmatched += 1
        anonymized.append(after)
        aligned_pairs.append(
            {
                "i": i,
                "before": original,
                "after": after,
                "changed": (original or "").strip() != (after or "").strip(),
            }
        )
    return anonymized, aligned_pairs, unmatched


def write_pairs_table(out_dir: Path, username: str, rows: list[dict]) -> Path:
    """Write CSV: msg_id, chunk_idx, changed, original, anonymized."""
    import csv

    path = out_dir / "pairs_table.csv"
    tmp = path.with_suffix(".csv.tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["msg_id", "chunk_idx", "changed", "original", "anonymized"],
            extrasaction="ignore",
        )
        w.writeheader()
        for row in rows:
            w.writerow(
                {
                    "msg_id": row["msg_id"],
                    "chunk_idx": row["chunk_idx"],
                    "changed": "true" if row["changed"] else "false",
                    "original": row["original"],
                    "anonymized": row["anonymized"],
                }
            )
    tmp.replace(path)
    # Also jsonl for easy loading
    jsonl_path = out_dir / "pairs_table.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def rebuild_outputs_from_chunk_pairs(
    *,
    out_dir: Path,
    username: str,
    all_comments: list[dict],
    source_file: Any,
    chunk_size: int,
    max_iterations: int,
    model: str,
    rewrite_chunks: bool = True,
) -> dict:
    """Rebuild profile + pairs table from each chunk's `pairs` (not positional arrays)."""
    chunk_list = list(chunks(all_comments, chunk_size))
    anon_flat: list[str] = []
    table_rows: list[dict] = []
    chunk_rows: list[dict] = []
    total_unmatched = 0

    for chunk_idx, (offset, part) in enumerate(chunk_list):
        saved = load_saved_chunk(out_dir, chunk_idx)
        if saved is None:
            break
        part_texts = [c["text"] for c in part]
        # Prefer source part order; fall back to stored originals if lengths match.
        stored_orig = saved.get("originals") or []
        if len(stored_orig) == len(part_texts):
            # Keep part_texts (source of truth for profile order).
            pass
        anon, aligned_pairs, unmatched = canonicalize_chunk_alignment(
            part_texts,
            saved.get("pairs") or [],
            saved.get("anonymized") or saved.get("final_texts"),
        )
        total_unmatched += unmatched
        if rewrite_chunks:
            saved = dict(saved)
            saved["originals"] = part_texts
            saved["anonymized"] = anon
            saved["pairs"] = aligned_pairs
            saved["n_changed"] = sum(1 for p in aligned_pairs if p["changed"])
            path = chunk_path(out_dir, chunk_idx)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(saved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            tmp.replace(path)

        anon_flat.extend(anon)
        for local_i, pair in enumerate(aligned_pairs):
            table_rows.append(
                {
                    "msg_id": offset + local_i,
                    "chunk_idx": chunk_idx,
                    "changed": pair["changed"],
                    "original": pair["before"],
                    "anonymized": pair["after"],
                }
            )
        chunk_rows.append(
            {
                "chunk_idx": chunk_idx,
                "offset": offset,
                "n_comments": len(part_texts),
                "rounds": saved.get("rounds"),
                "elapsed_s": saved.get("elapsed_s"),
                "n_changed": sum(1 for p in aligned_pairs if p["changed"]),
                "postfilter_n": saved.get("postfilter_n"),
                "n_brackets_left": saved.get("n_brackets_left"),
                "pairs_unmatched": unmatched,
            }
        )

    write_pairs_table(out_dir, username, table_rows)
    total_s = round(sum(float(r.get("elapsed_s") or 0) for r in chunk_rows), 3)
    manifest = assemble_outputs(
        out_dir=out_dir,
        username=username,
        all_comments=all_comments,
        anon_flat=anon_flat,
        source_file=source_file,
        chunk_size=chunk_size,
        max_iterations=max_iterations,
        chunk_rows=chunk_rows,
        model=model,
        total_s=total_s,
        n_chunks=len(chunk_list),
    )
    manifest["pairs_unmatched_total"] = total_unmatched
    manifest["pairs_table"] = "pairs_table.csv"
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def assemble_outputs(
    *,
    out_dir: Path,
    username: str,
    all_comments: list[dict],
    anon_flat: list[str],
    source_file: Any,
    chunk_size: int,
    max_iterations: int,
    chunk_rows: list[dict],
    model: str,
    total_s: float,
    n_chunks: int,
) -> dict:
    n_done = len(anon_flat)
    n_total = len(all_comments)
    comments_out: list[dict] = []
    n_changed = 0
    for i, orig in enumerate(all_comments[:n_done]):
        original = orig.get("text", "")
        anon = strip_numbered_prefix(
            anon_flat[i] if i < len(anon_flat) else original
        )
        changed = anon.strip() != (original or "").strip()
        if changed:
            n_changed += 1
        comments_out.append(
            {
                "i": i,
                "chunk_idx": i // max(chunk_size, 1),
                "text": anon,
                "text_original": original,
                "changed": changed,
                "subreddit": orig.get("subreddit", "reddit"),
                "user": orig.get("user", username),
                "timestamp": orig.get("timestamp", "0"),
                "pii": orig.get("pii", {}),
            }
        )

    complete = n_done >= n_total and n_total > 0
    status = {
        "complete": complete,
        "n_comments_total": n_total,
        "n_comments_done": n_done,
        "n_comments_changed": n_changed,
        "n_chunks_total": n_chunks,
        "n_chunks_done": len(chunk_rows),
        "chunk_size": chunk_size,
        "max_iterations": max_iterations,
        "model": model,
        "elapsed_s": total_s,
        "per_chunk": chunk_rows,
    }

    out_profile = {
        "username": username,
        "status": status,
        "comments": comments_out,
        "reviews": staab_reviews(),
        "source_file": source_file,
        "chunk_size": chunk_size,
        "max_iterations": max_iterations,
    }

    # Pretty structured snapshot (easy to inspect while the run is in progress).
    pretty_path = out_dir / "anonymized_profile.json"
    tmp_pretty = pretty_path.with_suffix(".json.tmp")
    tmp_pretty.write_text(
        json.dumps(out_profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp_pretty.replace(pretty_path)

    # Compact one-line jsonl (same schema as input profiles + text_original).
    compact = {
        "username": username,
        "comments": [
            {
                "text": c["text"],
                "text_original": c["text_original"],
                "subreddit": c["subreddit"],
                "user": c["user"],
                "timestamp": c["timestamp"],
                "pii": c["pii"],
                "changed": c["changed"],
                "chunk_idx": c["chunk_idx"],
            }
            for c in comments_out
        ],
        "reviews": out_profile["reviews"],
        "predictions": {},
        "evaluations": {},
        "source_file": source_file,
        "status": status,
        "chunk_size": chunk_size,
        "max_iterations": max_iterations,
    }
    jsonl_path = out_dir / "anonymized_profile.jsonl"
    tmp_jsonl = jsonl_path.with_suffix(".jsonl.tmp")
    tmp_jsonl.write_text(
        json.dumps(compact, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    tmp_jsonl.replace(jsonl_path)

    splits_path = ROOT / "data" / "splits" / f"{username}_query.jsonl"
    if splits_path.is_file():
        out_split = out_dir / f"{username}_query.jsonl"
        with splits_path.open(encoding="utf-8") as src, out_split.open(
            "w", encoding="utf-8"
        ) as dst:
            written = 0
            for line in src:
                if not line.strip():
                    continue
                if written >= n_done:
                    break
                obj = json.loads(line)
                body = (obj.get("b") or obj.get("body") or "").strip()
                if not body:
                    continue
                obj = dict(obj)
                anon = comments_out[written]["text"]
                if "b" in obj:
                    obj["b"] = anon
                elif "body" in obj:
                    obj["body"] = anon
                dst.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

    manifest = {
        "username": username,
        "n_comments_in": n_total,
        "n_comments_out": n_done,
        "n_comments_changed": n_changed,
        "chunk_size": chunk_size,
        "n_chunks": n_chunks,
        "n_chunks_done": len(chunk_rows),
        "complete": complete,
        "max_iterations": max_iterations,
        "model": model,
        "total_elapsed_s": total_s,
        "per_chunk": chunk_rows,
        "approx_seconds_per_comment": round(total_s / max(n_done, 1), 3),
        "approx_hours_for_500_users_same_size": round((total_s * 500) / 3600.0, 2),
        "anonymized_profile_json": str(pretty_path.name),
        "anonymized_profile_jsonl": str(jsonl_path.name),
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def user_is_complete(out_dir: Path, n_comments: int, chunk_size: int) -> bool:
    """True if all chunks for this user are on disk (or status.complete)."""
    pretty = out_dir / "anonymized_profile.json"
    if pretty.is_file():
        try:
            status = json.loads(pretty.read_text(encoding="utf-8")).get("status") or {}
            if status.get("complete") and status.get("n_comments_done", 0) >= n_comments:
                return True
        except Exception:
            pass
    n_chunks = (n_comments + chunk_size - 1) // max(chunk_size, 1)
    if n_chunks <= 0:
        return False
    for i in range(n_chunks):
        if not chunk_path(out_dir, i).is_file():
            return False
    return True


def anonymize_one_user(
    *,
    rec: dict,
    out_dir: Path,
    model,
    model_name: str,
    anonymizer,
    create_prompts,
    parse_answer,
    AnnotatedComments,
    Comment,
    Profile,
    Prompt,
    reddit_cfg,
    chunk_size: int,
    max_iterations: int,
    limit_chunks: Optional[int] = None,
    no_resume: bool = False,
) -> dict:
    """Run infer→anonymize for one profile record; resume per-chunk if possible."""
    username = rec["username"]
    all_comments = rec["comments"]
    source_file = rec.get("source_file")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)
    if no_resume:
        for p in (out_dir / "chunks").glob("chunk_*.json"):
            p.unlink()
        meta = out_dir / "chunks.jsonl"
        if meta.exists():
            meta.unlink()

    t_all = time.time()
    chunk_rows: list[dict] = []
    anon_flat: list[str] = []

    chunk_list = list(chunks(all_comments, chunk_size))
    if limit_chunks is not None:
        chunk_list = chunk_list[:limit_chunks]

    print(
        f"user={username} comments={len(all_comments)} "
        f"chunks={len(chunk_list)} size={chunk_size} iters={max_iterations}",
        flush=True,
    )

    for chunk_idx, (offset, part) in enumerate(chunk_list):
        saved = None if no_resume else load_saved_chunk(out_dir, chunk_idx)
        if saved is not None:
            print(
                f"\n=== chunk {chunk_idx} RESUME from disk "
                f"(changed={saved.get('n_changed')}) ===",
                flush=True,
            )
            result = saved
        else:
            print(
                f"\n=== chunk {chunk_idx} offset={offset} n={len(part)} ===",
                flush=True,
            )
            profile = build_chunk_profile(
                username=f"{username}_chunk{chunk_idx}",
                comment_dicts=part,
                Comment=Comment,
                AnnotatedComments=AnnotatedComments,
                Profile=Profile,
            )
            result = run_chunk(
                profile=profile,
                model=model,
                model_name=model_name,
                anonymizer=anonymizer,
                create_prompts=create_prompts,
                parse_answer=parse_answer,
                AnnotatedComments=AnnotatedComments,
                Comment=Comment,
                Prompt=Prompt,
                reddit_cfg=reddit_cfg,
                max_iterations=max_iterations,
                chunk_idx=chunk_idx,
            )
            payload = {
                "username": username,
                "chunk_idx": chunk_idx,
                "offset": offset,
                "n_comments": result["n_comments"],
                "rounds": result["rounds"],
                "elapsed_s": result["elapsed_s"],
                "n_changed": result["n_changed"],
                "postfilter_n": result.get("postfilter_n", 0),
                "n_brackets_left": result.get("n_brackets_left", 0),
                "numbered_ok_n": result.get("numbered_ok_n", 0),
                "numbered_fallback_n": result.get("numbered_fallback_n", 0),
                "originals": [c["text"] for c in part],
                "anonymized": result["final_texts"],
                "pairs": result["pairs"],
            }
            # Force originals/anonymized/pairs into the same source order via pairs.
            anon_aligned, pairs_aligned, unmatched = canonicalize_chunk_alignment(
                payload["originals"],
                payload["pairs"],
                payload["anonymized"],
            )
            payload["anonymized"] = anon_aligned
            payload["pairs"] = pairs_aligned
            payload["n_changed"] = sum(1 for p in pairs_aligned if p["changed"])
            if unmatched:
                print(
                    f"chunk {chunk_idx}: pairs unmatched={unmatched} "
                    "(fell back to positional anon for those)",
                    flush=True,
                )
            path = save_chunk(out_dir, payload)
            print(
                f"chunk {chunk_idx} saved → {path} "
                f"({result['elapsed_s']}s, changed={payload['n_changed']}/{result['n_comments']}, "
                f"postfilter={result.get('postfilter_n', 0)}, "
                f"brackets_left={result.get('n_brackets_left', 0)}, "
                f"numbered_ok={result.get('numbered_ok_n', 0)}, "
                f"numbered_fallback={result.get('numbered_fallback_n', 0)})",
                flush=True,
            )
            result = payload

        chunk_rows.append(
            {
                "chunk_idx": chunk_idx,
                "offset": offset,
                "n_comments": result.get("n_comments", len(part)),
                "rounds": result.get("rounds"),
                "elapsed_s": result.get("elapsed_s"),
                "n_changed": result.get("n_changed"),
                "postfilter_n": result.get("postfilter_n"),
                "n_brackets_left": result.get("n_brackets_left"),
            }
        )
        part_texts = [c["text"] for c in part]
        anon_aligned, _, _ = canonicalize_chunk_alignment(
            part_texts,
            result.get("pairs") or [],
            result.get("anonymized") or result.get("final_texts"),
        )
        anon_flat.extend(anon_aligned)

        assemble_outputs(
            out_dir=out_dir,
            username=username,
            all_comments=all_comments,
            anon_flat=anon_flat,
            source_file=source_file,
            chunk_size=chunk_size,
            max_iterations=max_iterations,
            chunk_rows=chunk_rows,
            model=model_name,
            total_s=round(time.time() - t_all, 3),
            n_chunks=len(chunk_list),
        )

    # Final rebuild: rewrite chunks + profile + pairs_table.csv from pairs.
    manifest = rebuild_outputs_from_chunk_pairs(
        out_dir=out_dir,
        username=username,
        all_comments=all_comments,
        source_file=source_file,
        chunk_size=chunk_size,
        max_iterations=max_iterations,
        model=model_name,
        rewrite_chunks=True,
    )
    print(f"\n=== DONE user={username} ===", flush=True)
    print(f"pairs table → {out_dir / 'pairs_table.csv'}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=REPO_ROOT / "data" / "profiles" / "profiles_query_one_full.jsonl",
    )
    parser.add_argument("--username", type=str, default=None)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "results_condition_c" / "user_smoke",
    )
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--model", type=str, default="qwen3.6-35b-a3b-nvfp4")
    parser.add_argument("--limit-chunks", type=int, default=None)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore existing chunk_*.json and recompute all chunks.",
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, str(ROOT))

    from src.anonymized.anonymizers.llm_anonymizers import LLMFullAnonymizer
    from src.configs import AnonymizerConfig, Config, ModelConfig, REDDITConfig, Task
    from src.configs.config import AnonymizationConfig
    from src.models.model_factory import get_model
    from src.prompts import Prompt
    from src.reddit.reddit import create_prompts, parse_answer
    from src.reddit.reddit_types import AnnotatedComments, Comment, Profile
    from src.utils.initialization import set_credentials

    cfg = Config(
        output_dir="results",
        seed=10,
        task=Task.ANONYMIZED,
        task_config=AnonymizationConfig(
            profile_path=str(args.profile),
            outpath=str(args.out_dir),
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

    model_cfg = ModelConfig(
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
    model = get_model(model_cfg)
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

    rec = load_profile_record(args.profile, args.username)
    manifest = anonymize_one_user(
        rec=rec,
        out_dir=args.out_dir,
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
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
