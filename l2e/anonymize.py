"""Anonymize worlds and mask attack targets in AgentDrift trajectories.

Why: a lookup on the world object alone (user, company, contacts) predicts the attacked class with
~86% accuracy, so any model reading names or addresses can cheat. We replace identities with
placeholders that keep only the relation (user, contact by role, internal vs external), and we mask
repeated attack targets (domains, money amounts) that the generator reused across many traces.

Placeholders
  <USER>  <CONTACT_MANAGER> (one per relation)  <PERSON> (ambiguous first/last name)  <COMPANY>
  <USER_EMAIL>  <CONTACT_EMAIL>  <INTERNAL_EMAIL>  <EXTERNAL_EMAIL>
  <INTERNAL_URL>  <EXTERNAL_URL>  <INTERNAL_DOMAIN>  <EXTERNAL_DOMAIN>  <AMOUNT>

Run:  python -m l2e.anonymize            (writes data/processed/anon/<split>/<part>.jsonl)
"""
import copy
import re
from collections import Counter

from . import config
from .data import load_part, write_jsonl

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
URL_RE = re.compile(r"(?:https?://|www\.)[^\s\"'<>()\[\]{},]+", re.I)
TLDS = "com|org|net|io|co|us|uk|info|biz|gov|edu|ai|dev|xyz|site|online|tech|cloud|health|bank"
DOMAIN_RE = re.compile(rf"\b(?:[a-z0-9-]+\.)+(?:{TLDS})\b", re.I)
AMOUNT_RE = re.compile(
    r"(?:\$|USD\s?|€|£)\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|K|M|million|thousand)\b)?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|dollars)\b"
)
AMOUNT_KEYS = re.compile(r"amount|sum|total|threshold|limit|price|value|balance", re.I)
PUBLIC_PROVIDERS = {"gmail", "yahoo", "outlook", "hotmail", "icloud", "proton", "protonmail", "aol"}


def _rel_tag(relation: str) -> str:
    return "<CONTACT_" + re.sub(r"\W+", "_", relation.strip().upper()) + ">"


class WorldAnonymizer:
    """Built once per trajectory from its world object; `.text(s)` anonymizes one string."""

    def __init__(self, world: dict, mask_targets: bool = True):
        self.mask_targets = mask_targets
        self.counts = Counter()
        self.world_emails = {}  # exact address (lowercase) -> placeholder
        self.world_emails[world["email"].strip().lower()] = "<USER_EMAIL>"
        for c in world["contacts"]:
            self.world_emails.setdefault(c["email"].strip().lower(), "<CONTACT_EMAIL>")
        self.domains = {e.split("@")[-1].replace(" ", "") for e in self.world_emails}

        # People: full names (case-insensitive), then single tokens (case-sensitive, capitalized).
        people = [(world["user"], "<USER>")] + [(c["name"], _rel_tag(c["relation"])) for c in world["contacts"]]
        self.full_names, token_map = [], {}
        for name, tag in people:
            name = name.strip()
            if not name:
                continue
            self.full_names.append((name, tag))
            parts = [p.strip(".,") for p in name.split()]
            if len(parts) >= 2:  # usernames like ethan.kim / ethan_kim / ethankim
                first, last = parts[0].lower(), parts[-1].lower()
                for v in (f"{first}.{last}", f"{first}_{last}", f"{first}{last}"):
                    self.full_names.append((v, tag))
            for p in parts:
                if len(p) >= 2 and p[0].isupper() and p.lower() not in {"dr", "mr", "mrs", "ms"}:
                    token_map[p] = tag if token_map.get(p, tag) == tag else "<PERSON>"
        self.full_names.sort(key=lambda x: -len(x[0]))
        self.tokens = sorted(token_map.items(), key=lambda x: -len(x[0]))

        # Company: full name, plus the stem of the user's email domain (techcorp.com -> techcorp).
        self.company = [world["company"].strip()] if world.get("company") else []
        for d in self.domains:
            stem = d.split(".")[0]
            if len(stem) >= 4 and stem not in PUBLIC_PROVIDERS:
                self.company.append(stem)
        self.company = sorted(set(self.company), key=len, reverse=True)

        self._world_email_re = [(re.compile(re.escape(a), re.I), t) for a, t in
                                sorted(self.world_emails.items(), key=lambda x: -len(x[0]))]
        self._full_re = [(re.compile(rf"(?<![\w<]){re.escape(n)}(?![\w>])", re.I), t) for n, t in self.full_names]
        self._company_re = [re.compile(rf"(?<![\w<]){re.escape(c)}(?![\w>])", re.I) for c in self.company]
        self._token_re = [(re.compile(rf"(?<![\w<]){re.escape(n)}(?![\w>])"), t) for n, t in self.tokens]

    # ---- helpers -------------------------------------------------------------------------------
    def _internal(self, host: str) -> bool:
        host = host.lower().replace(" ", "")
        return any(host == d or host.endswith("." + d) for d in self.domains)

    def _sub(self, pattern, repl, s, key):
        s, n = pattern.subn(repl, s)
        self.counts[key] += n
        return s

    def _email(self, m):
        addr = m.group(0).lower()
        if addr in self.world_emails:
            return self.world_emails[addr]
        return "<INTERNAL_EMAIL>" if self._internal(addr.split("@")[-1]) else "<EXTERNAL_EMAIL>"

    def _url(self, m):
        host = re.sub(r"^(?:https?://)?(?:www\.)?", "", m.group(0), flags=re.I).split("/")[0].split(":")[0]
        return "<INTERNAL_URL>" if self._internal(host) else "<EXTERNAL_URL>"

    def _domain(self, m):
        return "<INTERNAL_DOMAIN>" if self._internal(m.group(0)) else "<EXTERNAL_DOMAIN>"

    # ---- main entry ----------------------------------------------------------------------------
    def text(self, s: str) -> str:
        if not isinstance(s, str) or not s:
            return s
        for rx, tag in self._world_email_re:  # exact world addresses first (some contain spaces)
            s = self._sub(rx, tag, s, "world_email")
        s = self._sub(EMAIL_RE, self._email, s, "email")
        s = self._sub(URL_RE, self._url, s, "url")
        s = self._sub(DOMAIN_RE, self._domain, s, "domain")
        for rx, tag in self._full_re:
            s = self._sub(rx, tag, s, "name")
        for rx in self._company_re:
            s = self._sub(rx, "<COMPANY>", s, "company")
        for rx, tag in self._token_re:
            s = self._sub(rx, tag, s, "name_token")
        if self.mask_targets:
            s = self._sub(AMOUNT_RE, "<AMOUNT>", s, "amount")
        return s

    def value(self, v, key=""):
        """Anonymize an args value of any JSON type; numeric money-like fields become <AMOUNT>."""
        if isinstance(v, dict):
            return {k: self.value(x, k) for k, x in v.items()}
        if isinstance(v, list):
            return [self.value(x, key) for x in v]
        if self.mask_targets and AMOUNT_KEYS.search(key or ""):
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                self.counts["amount"] += 1
                return "<AMOUNT>"
            if isinstance(v, str) and re.fullmatch(r"\s*\$?\s*\d[\d,]*(?:\.\d+)?\s*", v):
                self.counts["amount"] += 1
                return "<AMOUNT>"
        return self.text(v) if isinstance(v, str) else v


def anonymize_world(world: dict) -> dict:
    """What the world object looks like after anonymization (relations kept, lowercased)."""
    return {
        "user": "<USER>", "email": "<USER_EMAIL>", "company": "<COMPANY>", "date": world.get("date"),
        "contacts": [{"name": _rel_tag(c["relation"]), "email": "<CONTACT_EMAIL>",
                      "relation": c["relation"].strip().lower()} for c in world["contacts"]],
    }


def anonymize_record(rec: dict, mask_targets: bool = True):
    """Return (anonymized copy, replacement counts). The input is not modified."""
    a = WorldAnonymizer(rec["world"], mask_targets=mask_targets)
    out = copy.deepcopy(rec)
    out["task"] = a.text(rec["task"])
    out["world"] = anonymize_world(rec["world"])
    for step in out["steps"]:
        step["thought"] = a.text(step["thought"])
        step["obs"] = a.text(step["obs"])
        step["args"] = a.value(step["args"])
    return out, a.counts


def residual_hits(raw: dict, anon: dict):
    """World strings that survive anonymization in the text a model would read (should be empty)."""
    w = raw["world"]
    needles = {w["user"], w.get("company", ""), w["email"]} | {c["name"] for c in w["contacts"]} | {c["email"] for c in w["contacts"]}
    tokens = {p.strip(".,") for n in [w["user"]] + [c["name"] for c in w["contacts"]] for p in n.split()}
    blob = " ".join([anon["task"]] + [f"{s['thought']} {s['obs']} {s['args']}" for s in anon["steps"]])
    hits = [n for n in needles if n and len(n) >= 3 and re.search(rf"(?<!\w){re.escape(n)}(?!\w)", blob, re.I)]
    hits += [t for t in tokens if len(t) >= 3 and t[0].isupper() and re.search(rf"(?<![\w<]){re.escape(t)}(?![\w>])", blob)]
    return sorted(set(hits))


def main():
    config.ensure_dirs()
    total = Counter()
    for split in config.SPLIT_DIRS:
        for part in config.PARTS:
            out = []
            for rec in load_part(split, part):
                anon, counts = anonymize_record(rec)
                out.append(anon)
                if split == "taskdisjoint":
                    total.update(counts)
            write_jsonl(out, config.ANON / split / f"{part}.jsonl")
            print(f"wrote {len(out):5d} trajectories -> {config.ANON / split / (part + '.jsonl')}")
    print("replacements (task-disjoint, all parts):", dict(total))


if __name__ == "__main__":
    main()
