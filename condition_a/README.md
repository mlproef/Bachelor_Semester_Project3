# Condition A — spaCy NER anonymization

Only the code that runs **Condition A**. Shared POOL-EN data is at the repo root.

## Method

**Rule-based NER masking** (no LLM).

spaCy `en_core_web_lg` scans each comment, finds named entities, and replaces them with type tags such as `[PERSON]`, `[LOCATION]`, `[ORGANIZATION]`, `[DATE]`, …. Everything else is left unchanged. One pass per profile — no inference step and no iterative rewrite.

## Layout

```text
condition_a/
├── anonymized_a/
│   ├── query/                         # 500 anonymized query profiles
│   └── candidates/                    # 1000 anonymized candidates
├── scripts/
│   ├── run_condition_a.py             # --side query (default)
│   └── run_condition_a_candidates.py  # side candidate
└── src/
    ├── anonymize_a.py
    ├── reddit_jsonl.py
    └── paths.py                       # → ../data/splits
```

## Shared input

`../data/splits/user_*_query.jsonl` (500 users)
`../data/splits/user_*_candidate.jsonl` (1000 candidates)

## Existing output

`anonymized_a/query/user_*_query.jsonl` — 500 profiles after spaCy NER.
`anonymized_a/candidates/user_*_candidate.jsonl` — 1000 candidate profiles.

Writes go **directly** to those folders. Finished files are skipped unless `--force`.

## Run

```bash
pip install spacy tqdm
python -m spacy download en_core_web_lg

# from repo root
python condition_a/scripts/run_condition_a.py --limit-files 1
python condition_a/scripts/run_condition_a.py
# → condition_a/anonymized_a/query/

python condition_a/scripts/run_condition_a_candidates.py --limit-files 1
python condition_a/scripts/run_condition_a_candidates.py
# → condition_a/anonymized_a/candidates/
```
