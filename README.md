# Summer anonymization experiment

Conditions **A / B / C**, **So1** utility metrics, **So2** matched POS control, and the **ESRC** attack pipeline (Extract → Search → Reason). Shared POOL-EN inputs live in `data/`.

```text
github-summer/
├── data/
│   ├── splits/                 # POOL-EN query (500) + candidate (1000) jsonl
│   └── profiles/               # Staab profiles_query.jsonl (C input)
├── condition_a/                # spaCy NER → anonymized_a/{query,candidates}/
├── condition_b/                # LLM generalize → anonymized_b/{query,candidates}/
├── condition_c/                # Staab-style chunked LLM → anonymized_c/query/
├── so1_bge/                    # So1: local BGE cosine + token-change
├── so1_server/                 # So1 fork: remote embeddings (jina-v3)
├── so2_matched/                # So2: POS word swaps at So1 token rates
├── esrc_pipeline/              # attack: extract / search / reason
├── so3/                        # two-sided stubs (not a full runner yet)
├── requirements.txt
└── README.md
```

## What is already in the repo

| Artifact | Path | Notes |
|----------|------|--------|
| Original queries + candidates | `data/splits/` | Input for A / B / So1 / So2 |
| Profiles (Staab) | `data/profiles/` | Input for C |
| Condition A | `condition_a/anonymized_a/` | 500 query + 1000 candidate, spaCy NER |
| Condition B query | `condition_b/anonymized_b/query/` | 500 users, `qwen3.6-35b-a3b-nvfp4` |
| Condition B candidates | `condition_b/anonymized_b/candidates/` | **837/1000** complete; resume without `--force` |
| Condition C query | `condition_c/anonymized_c/query/` | 500 users, Staab chunked LLM |
| So1 BGE + token + hand ratings | `so1_bge/results/{a,b,c}/` | 500 users; C is present |
| So1 bootstrap | `so1_bge/results/so1_stats/` | percentile 10k, seed 0 |
| So2 fake queries | `so2_matched/results/{a,b,c}/` | 500 users, rates matched to So1 |
| ESRC one-sided attack | `esrc_pipeline/results/` | baseline, A/B/C, so2_a/b/c vs raw gallery |

### So1 headline numbers (BGE `BAAI/bge-base-en-v1.5`, 500 users)

| Condition | Mean cosine similarity | Mean cosine distance | Token change |
|-----------|------------------------|----------------------|--------------|
| A | 0.932 | 0.068 | 8.5% |
| B | 0.947 | 0.053 | 10.9% |
| C | 0.928 | 0.072 | 13.8% |

Older gpt-4o-mini B metrics are in `so1_bge/results/b_gpt4o_mini_backup/` (cosine 0.893). Current B in `anonymized_b/query/` is the qwen run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_lg   # Condition A and So2 lexicon

# fill OLLAMA_URL / OLLAMA_API_KEY / OLLAMA_MODEL (gitignored .env)
```

Condition C historically used `openai==0.28.1`. If B and C conflict in one venv, use a second venv for C (see `requirements.txt` comments).

## Run (from repo root)

Paths are resolved from each script’s location. **A and B write into `anonymized_*/`** (resume skips finished files unless `--force`). C defaults to `condition_c/results_condition_c/` (gitignored); committed query output is `anonymized_c/query/`.

```bash
# Anonymization
python condition_a/scripts/run_condition_a.py --limit-files 1
python condition_a/scripts/run_condition_a_candidates.py --limit-files 1
python condition_b/scripts/run_condition_b.py --side query --limit-files 1
python condition_b/scripts/run_condition_b.py --side candidate --limit-files 1
python condition_c/scripts/run_condition_c.py --limit-users 1 --limit-chunks 1 --max-iterations 1

# So1 utility
python so1_bge/scripts/run_A_bge.py --limit-users 2
python so1_bge/scripts/run_B_bge.py --limit-users 2
python so1_bge/scripts/run_C_bge.py --limit-users 2
python so1_bge/scripts/fraction_of_tokens.py --conditions A,B,C --limit-users 2

# So2 / ESRC / So3 — see so2_matched/README.md, esrc_pipeline/README.md, so3/README.md
```

| Condition | Method | Status |
|-----------|--------|--------|
| A | spaCy NER (`en_core_web_lg`) | query + candidates done |
| B | LLM rewrite (`qwen3.6-35b-a3b-nvfp4`) | query done; candidates **837/1000** |
| C | Staab infer→anonymize (chunked) | query done; no candidates yet |

Details: `data/README.md`, `condition_*/README.md`, `so1_bge/README.md`, `so1_bge/results/README.md`, `so2_matched/README.md`, `esrc_pipeline/README.md`, `so3/README.md`.

## Secrets

- Use `.env` (gitignored) or env vars: `OLLAMA_URL`, `OLLAMA_API_KEY`, `OLLAMA_MODEL`
- Condition C reads gitignored `condition_c/credentials.py` (loads the same `.env`)
- Never commit `.env` or `credentials.py`
