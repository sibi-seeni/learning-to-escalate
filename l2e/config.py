"""Paths and project-wide constants. Everything else imports from here."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw AgentDrift clone (scripts/get_data.sh puts it here). Override with L2E_RAW=/some/path.
RAW = Path(os.environ.get("L2E_RAW", ROOT / "data" / "raw" / "AgentDrift"))
PROCESSED = ROOT / "data" / "processed"
ANON = PROCESSED / "anon"            # anonymized + masked trajectories
SPLITS_DIR = PROCESSED / "splits"     # tool-coverage and leave-one-domain-out id lists
MODELS = ROOT / "models"
REPORTS = ROOT / "reports" / "week1"

# Pin the dataset version so results are reproducible.
AGENTDRIFT_URL = "https://github.com/Asif-0209/AgentDrift.git"
AGENTDRIFT_COMMIT = "014a514fa998b4ac4519579fceb8a5884b379bda"

SPLIT_DIRS = {"stratified": "data", "taskdisjoint": "data_taskdisjoint"}
PARTS = ("train", "val", "test")
EXPECTED_SIZES = {
    "stratified": {"train": 8775, "val": 1880, "test": 1881},
    "taskdisjoint": {"train": 9081, "val": 1733, "test": 1722},
}
DOMAINS = ("banking", "coding", "email", "medical", "web")

SEED = 42

# Error labels: a missed hijack (dangerous) counts this many times more than a false alarm.
DANGEROUS_WEIGHT = 5.0

# Tool-shift protocol: share of each domain's tools held out from training, and how many random draws.
TOOL_HOLDOUT_FRAC = 0.2
TOOL_HOLDOUT_SEEDS = (0, 1, 2)
COVERAGE_LEVELS = (1.0, 0.75, 0.5, 0.25)


def ensure_dirs():
    for d in (PROCESSED, ANON, SPLITS_DIR, MODELS, REPORTS):
        d.mkdir(parents=True, exist_ok=True)
