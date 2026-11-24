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
REPORTS2 = ROOT / "reports" / "week2"
REPORTS3 = ROOT / "reports" / "week3"
FIGURES = ROOT / "results" / "figures"

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

# Week 2: folds for out-of-fold monitor predictions (the deferrer's training labels).
OOF_FOLDS = 5

# The main (edge) monitor, fixed after the week 2 domain-shift comparison (docs/WEEK2.md section 7).
MAIN_ARCH, MAIN_FEATURES = "transformer", "hash-roles"
MAIN_MONITOR = f"monitor_{MAIN_ARCH}_{MAIN_FEATURES}.pt"

# Week 3: out-of-fold labels can be grouped by task (week 2, the default) or by domain, so the deferrer learns
# from errors the monitor made on a domain it never trained on. Extra monitor seeds get their own folders.
# A "full-only" run trains just the full model: an extra seed for the seed ensemble / disagreement policies.
FOLD_BY = ("task", "domain")
MONITOR_SEEDS = (SEED, 1, 2)


def oof_tag(arch=MAIN_ARCH, features=MAIN_FEATURES, seed=SEED, fold_by="task"):
    """Folder name for one out-of-fold run: transformer_hash-roles, transformer_hash-roles_s1_domfold, ..."""
    tag = f"{arch}_{features}"
    if seed != SEED:
        tag += f"_s{seed}"
    if fold_by == "domain":
        tag += "_domfold"
    elif fold_by == "full":  # only the full model (val, test, context): an ensemble member, no deferrer labels
        tag += "_fullonly"
    return tag


def ensure_dirs():
    for d in (PROCESSED, ANON, SPLITS_DIR, MODELS, REPORTS, REPORTS2, REPORTS3, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
