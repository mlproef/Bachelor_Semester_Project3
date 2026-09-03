from pathlib import Path

# condition_a/src/paths.py → repo root is parents[2]
CONDITION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

# Shared POOL-EN inputs (A / B / C)
DATA_SPLITS = REPO_ROOT / "data" / "splits"

# Committed / run outputs for condition A
ANONYMIZED_A = CONDITION_ROOT / "anonymized_a"
ANONYMIZED_A_QUERY = ANONYMIZED_A / "query"
ANONYMIZED_A_CANDIDATES = ANONYMIZED_A / "candidates"

# Local recompute scratch (optional)
EXPERIMENTS = CONDITION_ROOT / "experiments"


def ensure_project_dirs() -> None:
    for p in (DATA_SPLITS, ANONYMIZED_A_QUERY, ANONYMIZED_A_CANDIDATES):
        p.mkdir(parents=True, exist_ok=True)
