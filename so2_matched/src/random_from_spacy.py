"""
So2 — matched POS degradation (fake defence).

For each query profile, randomly replace a fraction of words with other
same-POS words from pos_lexicon.json, at rates matched to So1 A / B / C.
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from shared.lexicon import LexiconEntry, PosBucket, PosLexicon
from shared.paths import DATA_SPLITS
from shared.profiles import QueryCorpus, UserProfile

from so1_bge.src.reddit_jsonl import comment_body
from so2_matched.src.paths import (
    DEFAULT_LEXICON,
    OUT_DIRS,
    RATES,
    SPACY_MODEL,
)

SKIP_POS = {"SPACE", "PUNCT", "X"}


# ---------------------------------------------------------------------------
# 1) Lexicon
# ---------------------------------------------------------------------------

def load_lexicon(path: Path) -> PosLexicon:
    """Load pos_lexicon.json from count_spacy into shared.lexicon.PosLexicon."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Lexicon not found: {path}\n"
            "Build it first: python so2_matched/lexicon/src/count_spacy.py --sides both"
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    by_pos_raw = raw.get("by_pos")
    if not isinstance(by_pos_raw, dict) or not by_pos_raw:
        raise ValueError(f"Invalid lexicon (missing by_pos): {path}")

    by_pos: Dict[str, PosBucket] = {}
    for pos, block in by_pos_raw.items():
        entries_raw = block.get("words") or []
        entries: List[LexiconEntry] = []
        for item in entries_raw:
            word = item.get("word")
            count = item.get("count")
            if not isinstance(word, str) or not word:
                continue
            if not isinstance(count, int) or count <= 0:
                continue
            entries.append(LexiconEntry(id=len(entries), word=word, count=count))
        if not entries:
            continue
        by_pos[pos] = PosBucket(pos=pos, entries=tuple(entries))

    if not by_pos:
        raise ValueError(f"Lexicon has no usable POS buckets: {path}")

    n_tokens = int(raw.get("n_tokens") or sum(b.n_tokens for b in by_pos.values()))
    return PosLexicon(
        by_pos=by_pos,
        path=path.resolve(),
        spacy_model=str(raw.get("spacy_model") or ""),
        sides=str(raw.get("sides") or ""),
        n_tokens=n_tokens,
    )


# ---------------------------------------------------------------------------
# 2) Query profiles / corpus
# ---------------------------------------------------------------------------

def _user_id_from_query_filename(name: str) -> str:
    stem = Path(name).stem
    if stem.endswith("_query"):
        return stem[: -len("_query")]
    return stem


def load_query_profile(path: Path) -> UserProfile:
    """Read one user_*_query.jsonl into a UserProfile (raw comment dicts)."""
    comments: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            comments.append(obj)
    return UserProfile(
        user_id=_user_id_from_query_filename(path.name),
        path=path.resolve(),
        comments=comments,
    )


def load_query_corpus(
    *,
    splits_dir: Path = DATA_SPLITS,
    limit_users: Optional[int] = None,
    name: str = "original",
) -> QueryCorpus:
    """Load query-side profiles from data/splits/."""
    paths = sorted(splits_dir.glob("user_*_query.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No user_*_query.jsonl in {splits_dir}")
    if limit_users is not None:
        paths = paths[:limit_users]
    profiles = [load_query_profile(p) for p in paths]
    return QueryCorpus(name=name, profiles=profiles)


def iter_query_profiles(
    *,
    limit_users: Optional[int] = None,
) -> List[UserProfile]:
    return list(load_query_corpus(limit_users=limit_users).profiles)


# ---------------------------------------------------------------------------
# 3–4) Core degradation
# ---------------------------------------------------------------------------

def _match_case(original: str, replacement: str) -> str:
    if not original or not replacement:
        return replacement
    if len(original) > 1 and original.isupper():
        return replacement.upper()
    if original[0].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def sample_replacement(
    lexicon: PosLexicon,
    pos: str,
    *,
    exclude: str,
    rng: random.Random,
) -> str:
    """Draw one word for this POS from the lexicon, ∝ corpus frequency."""
    bucket = lexicon.get(pos)
    if bucket is None or bucket.n_types == 0:
        return exclude

    exclude_l = exclude.lower()
    words = list(bucket.words)
    counts = list(bucket.counts)

    alt_words: List[str] = []
    alt_counts: List[int] = []
    for w, c in zip(words, counts):
        if w != exclude_l:
            alt_words.append(w)
            alt_counts.append(c)

    if alt_words:
        return rng.choices(alt_words, weights=alt_counts, k=1)[0]
    return rng.choices(words, weights=counts, k=1)[0]


def degrade_text(
    text: str,
    lexicon: PosLexicon,
    nlp: Any,
    *,
    rate: float,
    rng: random.Random,
) -> str:
    """Randomly replace ~rate of eligible tokens with same-POS lexicon words."""
    if not text or rate <= 0:
        return text

    doc = nlp(text)
    eligible_idxs: List[int] = []
    for i, tok in enumerate(doc):
        if tok.is_space or tok.is_punct or not tok.text.strip():
            continue
        if tok.pos_ in SKIP_POS:
            continue
        if lexicon.get(tok.pos_) is None:
            continue
        eligible_idxs.append(i)

    n = len(eligible_idxs)
    if n == 0:
        return text

    k = int(round(rate * n))
    k = max(0, min(n, k))
    if k == 0:
        return text

    replace_idxs = set(rng.sample(eligible_idxs, k=k))
    parts: List[str] = []
    for i, tok in enumerate(doc):
        if i in replace_idxs:
            new_word = sample_replacement(
                lexicon,
                tok.pos_,
                exclude=tok.text,
                rng=rng,
            )
            parts.append(_match_case(tok.text, new_word))
        else:
            parts.append(tok.text)
        parts.append(tok.whitespace_)
    return "".join(parts)


def degrade_profile(
    profile: UserProfile,
    lexicon: PosLexicon,
    nlp: Any,
    *,
    rate: float,
    rng: random.Random,
) -> UserProfile:
    """Run degrade_text on every comment body; does not mutate input."""
    new_comments: List[Dict[str, Any]] = []
    for obj in profile.comments:
        text = comment_body(obj)
        if not text:
            new_comments.append(dict(obj))
            continue

        new_text = degrade_text(text, lexicon, nlp, rate=rate, rng=rng)
        new_obj = dict(obj)
        if "b" in new_obj:
            new_obj["b"] = new_text
        if "body" in new_obj:
            new_obj["body"] = new_text
        if "b" not in new_obj and "body" not in new_obj:
            new_obj["b"] = new_text
        new_comments.append(new_obj)

    return UserProfile(
        user_id=profile.user_id,
        path=profile.path,
        comments=new_comments,
    )


# ---------------------------------------------------------------------------
# 5) Output
# ---------------------------------------------------------------------------

def write_profile(profile: UserProfile, out_dir: Path, *, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile.user_id}_query.jsonl"
    if out_path.is_file() and not force:
        return out_path
    lines = profile.to_jsonl_lines()
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_path


def create_json(corpus: QueryCorpus, out_dir: Path, *, force: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    for profile in corpus.profiles:
        write_profile(profile, out_dir, force=force)
    return out_dir


# ---------------------------------------------------------------------------
# 6) Driver
# ---------------------------------------------------------------------------

def run_matched(
    *,
    match: str,
    lexicon_path: Path = DEFAULT_LEXICON,
    limit_users: Optional[int] = None,
    seed: int = 0,
    force: bool = False,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    match in {"A","B","C"} → So1 rate for that condition.
    Writes user_*_query.jsonl under so2_matched/results/{a|b|c}/.
    """
    match = match.upper()
    if match not in RATES:
        raise ValueError(f"match must be A, B, or C, got {match!r}")

    rate = RATES[match]
    target = out_dir if out_dir is not None else OUT_DIRS[match]
    rng = random.Random(seed)

    lexicon = load_lexicon(lexicon_path)
    try:
        import spacy
    except ImportError as e:
        raise SystemExit(
            "spaCy missing — pip install spacy && python -m spacy download en_core_web_lg"
        ) from e

    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser", "lemmatizer"])
    original = load_query_corpus(limit_users=limit_users, name="original")

    degraded_profiles: List[UserProfile] = []
    try:
        from tqdm import tqdm

        profiles_iter = tqdm(original.profiles, desc=f"matched_{match}", unit="user")
    except ImportError:
        profiles_iter = original.profiles

    for profile in profiles_iter:
        out_path = target / f"{profile.user_id}_query.jsonl"
        if out_path.is_file() and not force:
            degraded_profiles.append(load_query_profile(out_path))
            continue
        degraded = degrade_profile(profile, lexicon, nlp, rate=rate, rng=rng)
        write_profile(degraded, target, force=True)
        degraded_profiles.append(degraded)

    corpus = QueryCorpus(name=f"matched_{match}", profiles=degraded_profiles)

    manifest = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "condition": f"matched_{match}",
        "match": match,
        "rate": rate,
        "seed": seed,
        "spacy_model": SPACY_MODEL,
        "lexicon": str(lexicon_path.resolve()),
        "lexicon_sides": lexicon.sides,
        "n_users": corpus.n_users,
        "n_comments": corpus.n_comments,
        "out_dir": str(target.resolve()),
        "limit_users": limit_users,
        "force": force,
    }
    man_path = target / f"manifest_matched_{match}.json"
    target.mkdir(parents=True, exist_ok=True)
    man_path.write_text(json.dumps(manifest, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"wrote {corpus.n_users} users → {target}")
    print(f"manifest → {man_path}")
    return target
