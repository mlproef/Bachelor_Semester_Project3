from pathlib import Path

CONDITION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_SPLITS = REPO_ROOT / "data" / "splits"
PROMPTS_DIR = CONDITION_ROOT / "prompts"
EXPERIMENTS = CONDITION_ROOT / "experiments"

# Committed anonymized outputs (same layout as condition A)
ANONYMIZED_B = CONDITION_ROOT / "anonymized_b"
ANONYMIZED_B_QUERY = ANONYMIZED_B / "query"
ANONYMIZED_B_CANDIDATES = ANONYMIZED_B / "candidates"


def ensure_project_dirs() -> None:
    for p in (
        DATA_SPLITS,
        PROMPTS_DIR,
        EXPERIMENTS / "condition_B",
        ANONYMIZED_B_QUERY,
        ANONYMIZED_B_CANDIDATES,
    ):
        p.mkdir(parents=True, exist_ok=True)
