"""Load AgentDrift splits and print a short summary.

Run:  python -m l2e.data
"""
import json
from collections import Counter
from pathlib import Path

from . import config


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_part(split="taskdisjoint", part="train", root=None):
    """Raw trajectories for one part of a split ('stratified' or 'taskdisjoint')."""
    root = Path(root or config.RAW)
    path = root / config.SPLIT_DIRS[split] / f"{part}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run scripts/get_data.sh first.")
    return read_jsonl(path)


def load_split(split="taskdisjoint", root=None):
    return {p: load_part(split, p, root) for p in config.PARTS}


def load_anon(split="taskdisjoint", part="train"):
    """Anonymized + masked trajectories written by `python -m l2e.anonymize`."""
    path = config.ANON / split / f"{part}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m l2e.anonymize` first.")
    return read_jsonl(path)


def summarize(split):
    data = load_split(split)
    lines = [f"### {split} split", "", "| part | trajectories | steps | attacked | benign | failed_attack | hard_negative |",
             "|---|---|---|---|---|---|---|"]
    for part, recs in data.items():
        exp = config.EXPECTED_SIZES[split][part]
        assert len(recs) == exp, f"{split}/{part}: expected {exp} trajectories, got {len(recs)}"
        cats = Counter(r["category"] for r in recs)
        steps = sum(len(r["steps"]) for r in recs)
        lines.append(f"| {part} | {len(recs)} | {steps} | {cats['attacked']} | {cats['benign']} | "
                     f"{cats['failed_attack']} | {cats['hard_negative']} |")
    test = data["test"]
    lines += ["", "Test part by domain: " + ", ".join(f"{k} {v}" for k, v in sorted(Counter(r['agent'] for r in test).items())),
              "Test attacked by compliance: " + ", ".join(
                  f"{k} {v}" for k, v in sorted(Counter(r['compliance'] for r in test if r['category'] == 'attacked').items())), ""]
    return lines


def main():
    config.ensure_dirs()
    out = ["# Week 1 · Data summary", "", f"AgentDrift commit `{config.AGENTDRIFT_COMMIT}`", ""]
    for split in config.SPLIT_DIRS:
        out += summarize(split)
    text = "\n".join(out)
    print(text)
    (config.REPORTS / "01_data_summary.md").write_text(text + "\n")


if __name__ == "__main__":
    main()
