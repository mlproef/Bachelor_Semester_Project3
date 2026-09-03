# So1 results

Per-condition metric tables for the current BGE run (`BAAI/bge-base-en-v1.5`, 500 users).

| Folder | Contents |
|--------|----------|
| `a/` | BGE + token-change + hand-rating CSVs for A |
| `b/` | BGE + token-change for B (qwen anonymization) |
| `c/` | BGE + token-change for C |
| `b_gpt4o_mini_backup/` | older B metrics (gpt-4o-mini; cosine 0.893) |
| `so1_stats/` | `utility_bootstrap.json` (10k percentile CI) |
| `hand_ratings_summary.json` | 50-pair meaning + fluency summary |

Typical files inside `a/`, `b/`, or `c/`:

- `bge_condition_X.json` / `*_per_user.jsonl` — BGE cosine similarity
- `token_change_condition_X.json` / `*_per_user.jsonl` — token edit fraction

Headline means: A cosine 0.932 / token 8.5%; B 0.947 / 10.9%; C 0.928 / 13.8%.
