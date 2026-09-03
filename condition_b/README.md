# Condition B — LLM generalization anonymization

Only the code that runs **Condition B**. Shared POOL-EN data is at the repo root.

## Method

**One-shot LLM generalization** (no separate inference step).

Each comment is sent once to an LLM with `prompts/condition_b_generalize.txt`: rewrite so identifying details become more generic, while meaning and style stay close to the original. Unlike Condition C, there is no “guess attributes → anonymize → repeat” loop — just a single rewrite per comment.

The committed query run in `anonymized_b/query/` used **`qwen3.6-35b-a3b-nvfp4`** (OpenAI-compatible API). An older gpt-4o-mini So1 snapshot is kept under `so1_bge/results/b_gpt4o_mini_backup/`, not in `anonymized_b/`.

## Layout

```text
condition_b/
├── anonymized_b/
│   ├── query/                         # 500 anonymized query profiles
│   └── candidates/                    # 837/1000 complete; resume in place
├── prompts/condition_b_generalize.txt
├── scripts/run_condition_b.py
└── src/
    ├── anonymize_b.py
    ├── reddit_jsonl.py
    └── paths.py                       # → ../data/splits
```

## Shared input

`../data/splits/user_*_query.jsonl` (500 users)
`../data/splits/user_*_candidate.jsonl` (1000 candidates)

## Existing output

`anonymized_b/query/user_*_query.jsonl` — 500 completed query profiles.
`anonymized_b/candidates/` — **837/1000** complete profiles (500 comments each). Incomplete files resume without `--force`.

## Run

Set API settings in the environment (do not commit secrets):

- `OLLAMA_URL` or `OPENAI_API_BASE`
- `OLLAMA_API_KEY` or `OPENAI_API_KEY`

Chat model: pass `--model`, or set `CONDITION_B_MODEL`. If both are unset, the CLI default is `qwen3.5-4b`. The experiment query/candidate runs used `--model qwen3.6-35b-a3b-nvfp4`.

```bash
pip install openai tqdm python-dotenv

# from repo root (loads condition_b/.env or repo-root .env)
python condition_b/scripts/run_condition_b.py --side query --limit-files 1
python condition_b/scripts/run_condition_b.py --side query --model qwen3.6-35b-a3b-nvfp4
# → condition_b/anonymized_b/query/

python condition_b/scripts/run_condition_b.py --side candidate --model qwen3.6-35b-a3b-nvfp4
# → condition_b/anonymized_b/candidates/
```
