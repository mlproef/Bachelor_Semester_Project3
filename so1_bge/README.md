# So1 — utility metrics (BGE similarity + token change)

Compares **original** `data/splits/` query profiles with anonymized A/B/C. Local embeddings: `BAAI/bge-base-en-v1.5`.

```text
so1_bge/
├── objects/          # data classes
├── src/              # I/O, embeddings, cosine, metrics
├── scripts/          # CLI + bootstrap_utility.py
├── results/
│   ├── a/            # Condition A
│   ├── b/            # Condition B (current qwen run)
│   ├── c/            # Condition C
│   ├── b_gpt4o_mini_backup/
│   ├── so1_stats/    # bootstrap JSON
│   └── hand_ratings_summary.json
└── README.md
```

| Condition | Anonymized input | Metrics output |
|-----------|------------------|----------------|
| A | `condition_a/anonymized_a/query/` | `so1_bge/results/a/` |
| B | `condition_b/anonymized_b/query/` | `so1_bge/results/b/` |
| C | `condition_c/anonymized_c/query/` | `so1_bge/results/c/` |

## Setup

```bash
pip install sentence-transformers tqdm numpy
```

## Run (from repo root)

```bash
python so1_bge/scripts/run_A_bge.py --limit-users 2
python so1_bge/scripts/run_B_bge.py --limit-users 2
python so1_bge/scripts/run_C_bge.py --limit-users 2
python so1_bge/scripts/run.py --conditions A,B,C --limit-users 2
python so1_bge/scripts/fraction_of_tokens.py --conditions A,B,C --limit-users 2
python so1_bge/scripts/bootstrap_utility.py
```

Outputs go to `so1_bge/results/{a,b,c}/`. Bootstrap: `so1_bge/results/so1_stats/utility_bootstrap.json`.
