# ESRC attack pipeline

Extract → Search → Reason. One-sided runs use anonymized **queries** against the **raw** (non-anonymized) candidate gallery.

```text
esrc_pipeline/
├── extract/ search/ reason/
├── scripts/          # baseline + real A/B/C
├── scripts_so2/      # So2 control queries (does not overwrite A/B/C)
└── results/
    ├── baseline/ a/ b/ c/
    ├── so2_a/ so2_b/ so2_c/
    └── so2_stats/    # reason bootstrap + McNemar
```

Needs repo-root `.env` (`OLLAMA_URL`, `OLLAMA_API_KEY`). Extract default model in these CLIs is `qwen3.5-4b`; Reason for the experiment used `qwen3.6-35b-a3b-nvfp4`.

```bash
# from repo root — smoke
python esrc_pipeline/scripts/run_extract_baseline.py --side query --limit-files 1 --dry-run
python esrc_pipeline/scripts/run_extract_a.py --side query --limit-files 1
python esrc_pipeline/scripts/run_search_a.py
python esrc_pipeline/scripts/run_reason_a.py

python esrc_pipeline/scripts_so2/run_extract_a.py --limit-files 1
python esrc_pipeline/scripts_so2/bootstrap_reason.py
```

Search/Reason for A/B/C and so2_* compare against the raw baseline gallery. Two-sided (anonymized gallery) is **So3** and is not a full runner yet.
