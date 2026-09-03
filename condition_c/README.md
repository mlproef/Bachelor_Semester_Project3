# Condition C — Staab-style LLM anonymization

Only the code that runs **Condition C**. Shared Staab-format profiles live at the repo root.

## Method

Condition C follows the **Staab et al.** idea: before rewriting text, the model first **guesses** what personal attributes can be inferred from the comments, then **anonymizes** to block those guesses. This is different from A (NER tags) and B (one-shot generalization).

How one user is processed:

1. **Chunking** — comments are split into chunks of 20 (long profiles would overflow context).
2. **Inference** — for the current chunk texts, the LLM predicts personal attributes (age, sex, location, occupation, …) in Staab’s structured format.
3. **Anonymize** — the same LLM gets the comments **plus** those inferences and rewrites the comments so the attributes become hard to recover. Prompt style ≈ Staab `prompt_level=3`, with numbered lines (`0:`, `1:`, …) so each comment can be mapped back reliably.
4. **Iterate** — steps 2→3 repeat up to **3** times on the already-anonymized texts (each round tries to remove leftover leakage).
5. **Postfilter** — if the model left NER-style `[PERSON]` / `[LOCATION]` tags, a short cleanup pass rewrites them into natural wording.
6. **Save & resume** — each chunk is written to disk immediately; finished users/chunks are skipped on rerun.

Batch mode walks all users in `profiles_query.jsonl` and calls this loop per user.

## Layout

```text
condition_c/
├── anonymized_c/
│   └── query/                         # 500 committed anonymized query profiles
├── scripts/
│   ├── run_condition_c.py              # entrypoint (like A/B)
│   ├── run_batch_chunked_anonymize.py  # multi-user loop
│   └── run_chunked_anonymize.py        # core per-user logic
├── src/                                # Staab pieces used by C
└── credentials_clean.py
```

There is no `anonymized_c/candidates/` yet.

## Shared input

`../data/profiles/profiles_query.jsonl` (500 users)

## Run

The CLI default `--out-root` is `condition_c/results_condition_c/` (local rerun, gitignored). The committed experiment output is `anonymized_c/query/`.

```bash
cp condition_c/credentials_clean.py condition_c/credentials.py   # optional; or use .env
# preferred: fill repo-root .env (OLLAMA_URL / OLLAMA_API_KEY / OLLAMA_MODEL)

# from repo root
python condition_c/scripts/run_condition_c.py
# defaults: data/profiles/profiles_query.jsonl
#           chunk=20, iters=3, model from OLLAMA_MODEL or qwen3.6-35b-a3b-nvfp4
#           → condition_c/results_condition_c/

# committed query output:
#   condition_c/anonymized_c/query/user_*_query.jsonl

# smoke
python condition_c/scripts/run_condition_c.py --limit-users 1 --limit-chunks 1 --max-iterations 1
```

API via `credentials.py` or `OLLAMA_*` / `OPENAI_*` (do not commit secrets).
