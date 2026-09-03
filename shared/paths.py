"""Common project paths for github-summer (shared by So1 / So2)."""
from __future__ import annotations

from pathlib import Path

# Repo root (github-summer/)
REPO_ROOT = Path(__file__).resolve().parents[1]

# Aliases kept for bsp3-summer-style imports in So2 scripts
BSP3_SUMMER = REPO_ROOT
PROJECT = REPO_ROOT

DATA_SPLITS = REPO_ROOT / "data" / "splits"
EXPERIMENTS = REPO_ROOT / "experiments"
RESULTS_TABLES = REPO_ROOT / "results" / "tables"
