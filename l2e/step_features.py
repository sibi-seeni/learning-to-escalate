"""Step featurizer for the neural monitor.

A trace becomes a stream of events:   call_0, obs_0, call_1, obs_1, ..., call_t
  call_k  the agent's thought, tool name and arguments for step k
  obs_k   what that tool returned
The monitor reads the stream left to right and outputs P(hijacked) at every call event, using only
events before it. So the score at call_t sees steps 0..t-1 in full plus step t's call, never step t's
result. This is the same view as decision_points.prefix_steps().

Each event is described by:
  tool id      index into a vocabulary built from training data only (0 = padding, 1 = unknown tool)
  structured   14 numbers (STRUCT below): event type, position, state-changing and sink flags,
               argument count, counts of placeholders, suspicious keywords, text length
  text         "hash": hashed bag of words and word pairs, HASH_DIM numbers, no pretrained model
               "minilm" / "mpnet": a sentence embedding, looked up from a cache (l2e/encoders.py)

  roles        8 numbers, only with "-roles": what the event's tool does (l2e/tool_roles.py), e.g. sends data
               out or moves money. Meant to carry over to new domains, where tool names are new.

Feature-set names: the encoder name plus options, joined by "-":
  nothought  hide every agent thought (does the monitor still work without the agent's reasoning?)
  roles      add the tool-role features
  noid       drop the memorized tool id (every tool becomes "unknown"), so the model must rely on roles and text
E.g. "hash", "hash-nothought", "hash-roles", "minilm-roles-noid".

This file needs only numpy and the standard library, so the board can import a copy of it as is.
The laptop-only helper default_cfg() pulls the tool lists and keywords from the rest of l2e.
"""
import json
import math
import re
import zlib

import numpy as np

HASH_DIM = 512
PAD, UNK = 0, 1
STRUCT = ["is_call", "is_obs", "position", "state_changing", "sink", "n_args", "ext_email", "ext_url",
          "ext_domain", "amount", "internal", "known_contact", "keywords", "length"]
TOKEN_RE = re.compile(r"<[a-z_]+>|[a-z0-9]+")


OPTIONS = ("nothought", "roles", "noid")
ROLE_NAMES = ["reads_private", "reads_untrusted", "writes", "sends_out", "moves_money", "runs_code", "deletes",
              "changes_trust"]  # same order as l2e/tool_roles.py
# Fallback for tools that aren't in the role table: keywords in the tool's name.
ROLE_KEYWORDS = {
    "reads_private": ("get_", "list_", "check_", "search_patients", "history", "record"),
    "reads_untrusted": ("read", "search", "fetch", "browse", "web", "page", "inbox", "thread", "attachment", "download",
                        "extract", "scrape", "log", "docs"),
    "writes": ("write", "create", "update", "edit", "set_", "save", "add_", "move", "record_", "commit", "upload"),
    "sends_out": ("send", "forward", "reply", "submit", "post", "share", "upload", "push", "message", "email", "form",
                  "publish", "export"),
    "moves_money": ("transfer", "pay", "withdraw", "deposit", "loan", "purchase", "buy", "refund", "wire", "claim"),
    "runs_code": ("execute", "run", "install", "shell", "eval", "script", "command"),
    "deletes": ("delete", "remove", "cancel", "drop", "purge", "close"),
    "changes_trust": ("contact", "beneficiar", "login", "auth", "permission", "filter", "grant", "freeze", "token",
                      "password", "key"),
}


def split_feature_set(feature_set):
    """'minilm-roles-noid' -> ('minilm', {'nothought': False, 'roles': True, 'noid': True})"""
    enc, *opts = feature_set.split("-")
    bad = set(opts) - set(OPTIONS)
    if bad:
        raise ValueError(f"unknown feature-set option(s) {bad}; known: {OPTIONS}")
    return enc, {o: o in opts for o in OPTIONS}


def default_cfg(feature_set="hash"):
    """Everything the featurizer needs, as plain JSON (saved next to every exported model)."""
    from .features import KEYWORDS
    from .tool_roles import ROLE_NAMES as TR_NAMES, TOOL_ROLES
    from .tools import SINK_VERBS, STATE_CHANGING
    assert TR_NAMES == ROLE_NAMES
    _, opt = split_feature_set(feature_set)
    return {"hash_dim": HASH_DIM, "state_changing": sorted(STATE_CHANGING), "sink_verbs": list(SINK_VERBS),
            "keywords": list(KEYWORDS), "thoughts": not opt["nothought"], "roles": opt["roles"],
            "tool_roles": TOOL_ROLES if opt["roles"] else {}}


def roles_from_name(tool):
    """Roles for a tool that isn't in the table, guessed from its name."""
    t = tool.lower()
    return [r for r in ROLE_NAMES if any(k in t for k in ROLE_KEYWORDS[r])]


def call_text(step, thoughts=True):
    thought = step["thought"] if thoughts else ""
    return f"{thought} | {step['tool']} | {json.dumps(step['args'], sort_keys=True)}"


def event_list(steps, thoughts=True):
    """[(kind, step index, step, text)] for the stream call_0, obs_0, ..., call_last.

    The last step's result is never included: no decision uses it (and in a prefix it doesn't exist).
    """
    out = []
    for k, s in enumerate(steps):
        out.append(("call", k, s, call_text(s, thoughts)))
        if k < len(steps) - 1:
            out.append(("obs", k, s, s.get("obs", "")))
    return out


def hash_vector(text, dim=HASH_DIM):
    """Binary bag of words + word pairs, hashed into `dim` slots with crc32 (stable across machines)."""
    toks = TOKEN_RE.findall(text.lower())
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    v = np.zeros(dim, np.float32)
    if grams:
        idx = {zlib.crc32(g.encode()) % dim for g in grams}
        v[list(idx)] = 1.0 / math.sqrt(len(idx))
    return v


class Featurizer:
    """steps -> (tool names, event matrix X, positions of the call events).

    text_lookup: None for the hashed text features, or a callable text -> vector (cached embeddings).
    """

    def __init__(self, cfg=None, text_lookup=None, text_dim=None):
        self.cfg = cfg or default_cfg()
        self.sc = set(self.cfg["state_changing"])
        self.sinks = tuple(self.cfg["sink_verbs"])
        self.keywords = self.cfg["keywords"]
        self.thoughts = self.cfg.get("thoughts", True)
        self.roles = self.cfg.get("roles", False)
        self.tool_roles = self.cfg.get("tool_roles", {})
        self.n_struct = len(STRUCT) + (len(ROLE_NAMES) if self.roles else 0)
        self.text_lookup = text_lookup
        self.text_dim = self.cfg["hash_dim"] if text_lookup is None else text_dim
        self.dim = self.n_struct + self.text_dim

    def role_vec(self, tool):
        rs = self.tool_roles.get(tool)
        if rs is None:
            rs = roles_from_name(tool)
        return [float(r in rs) for r in ROLE_NAMES]

    def struct(self, kind, k, step, text):
        call = kind == "call"
        tool = step["tool"]
        lo = text.lower()
        c = text.count
        return [
            float(call), float(not call), k / 10.0,
            float(call and tool in self.sc), float(call and any(v in tool for v in self.sinks)),
            (len(step["args"]) / 5.0) if call else 0.0,
            math.log1p(c("<EXTERNAL_EMAIL>")), math.log1p(c("<EXTERNAL_URL>")), math.log1p(c("<EXTERNAL_DOMAIN>")),
            math.log1p(c("<AMOUNT>")),
            math.log1p(c("<INTERNAL_EMAIL>") + c("<INTERNAL_URL>") + c("<INTERNAL_DOMAIN>")),
            math.log1p(c("<USER_EMAIL>") + c("<CONTACT_EMAIL>")),
            math.log1p(sum(kw in lo for kw in self.keywords)),
            math.log1p(len(text)) / 8.0,
        ]

    def text_vec(self, text):
        if self.text_lookup is None:
            return hash_vector(text, self.text_dim)
        return np.asarray(self.text_lookup(text), np.float32)

    def events(self, steps):
        ev = event_list(steps, self.thoughts)
        X = np.zeros((len(ev), self.dim), np.float32)
        tools, calls = [], []
        for i, (kind, k, s, text) in enumerate(ev):
            X[i, :len(STRUCT)] = self.struct(kind, k, s, text)
            if self.roles:  # on a result event: the roles of the tool that produced it
                X[i, len(STRUCT):self.n_struct] = self.role_vec(s["tool"])
            X[i, self.n_struct:] = self.text_vec(text)
            tools.append(s["tool"])
            if kind == "call":
                calls.append(i)
        return tools, X, calls


def build_vocab(tool_names):
    """Tool name -> id, from training tools only. 0 = padding, 1 = unknown."""
    return {t: i + 2 for i, t in enumerate(sorted(set(tool_names)))}


def tool_ids(tools, vocab):
    return np.array([vocab.get(t, UNK) for t in tools], np.int64)
