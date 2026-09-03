# So2 — matched POS control

Query-only **text damage** at the same token-change rates as So1, without the real A/B/C anonymizers. spaCy tags tokens, then random same-POS swaps from `lexicon/result/pos_lexicon.json`.

Rates (from So1 `token_change_*.json`, current qwen B, not the old gpt-4o-mini run):

| Match | Token-change rate |
|-------|-------------------|
| A | 8.5% |
| B | 10.9% |
| C | 13.8% |

```text
so2_matched/
├── lexicon/result/pos_lexicon.json
├── scripts/run_A.py   # also run_B.py, run_C.py
├── src/random_from_spacy.py
└── results/{a,b,c}/   # 500 fake query jsonl each
```

Needs `en_core_web_lg` (same as Condition A).

```bash
# from repo root
python so2_matched/scripts/run_A.py --limit-users 5
python so2_matched/scripts/run_B.py
python so2_matched/scripts/run_C.py
```

ESRC then attacks these queries against the **raw** baseline gallery (`esrc_pipeline/scripts_so2/`).
