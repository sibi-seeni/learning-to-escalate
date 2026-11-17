"""Sentence-encoder text features for the monitor (MiniLM, mpnet), and MiniLM for the board.

The hashed text features need no pretrained model. These are the two alternatives in the plan:
  minilm  sentence-transformers/all-MiniLM-L6-v2   22.7M params, 384-dim   (edge candidate)
  mpnet   sentence-transformers/all-mpnet-base-v2  ~110M params, 768-dim   (DriftNet's encoder; laptop only,
                                                                            for the "better monitor" baseline)

Every event text of every task-disjoint trace (with and without thoughts) is embedded once and cached.
Embeddings are L2-normalized, as in sentence-transformers' usual setup.

Run (needs internet once, to download the model):
  python -m l2e.encoders embed minilm          -> data/processed/emb/minilm.npy + minilm_texts.json
  python -m l2e.encoders embed mpnet
  python -m l2e.encoders export-minilm         -> board/models/minilm.onnx, minilm_int8.onnx, minilm_tokenizer.json
Then train on them:  python -m l2e.train_monitor --encoder minilm
"""
import argparse
import json
import shutil
import time

import numpy as np

from . import config
from .data import load_anon
from .step_features import event_list

MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "mpnet": "sentence-transformers/all-mpnet-base-v2"}
EMB = config.PROCESSED / "emb"
BOARD_MODELS = config.ROOT / "board" / "models"
MINILM_MAX_TOKENS = 256  # all-MiniLM-L6-v2's max_seq_length in sentence-transformers


def device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def all_texts():
    """Unique event texts over all task-disjoint traces, with and without the agent's thoughts."""
    seen = {}
    for part in config.PARTS:
        for tr in load_anon("taskdisjoint", part):
            for thoughts in (True, False):
                for _, _, _, text in event_list(tr["steps"], thoughts):
                    seen.setdefault(text, None)
    return list(seen)


def embed(name, batch_size=64):
    from sentence_transformers import SentenceTransformer
    EMB.mkdir(parents=True, exist_ok=True)
    texts = all_texts()
    dev = device()
    model = SentenceTransformer(MODELS[name], device=dev)
    print(f"embedding {len(texts)} unique event texts with {MODELS[name]} on {dev} "
          f"(max {model.max_seq_length} tokens) ...", flush=True)
    t0 = time.time()
    E = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
                     normalize_embeddings=True).astype(np.float32)
    np.save(EMB / f"{name}.npy", E)
    (EMB / f"{name}_texts.json").write_text(json.dumps(texts))
    meta = {"model": MODELS[name], "texts": len(texts), "dim": int(E.shape[1]), "device": dev,
            "max_seq_length": model.max_seq_length, "seconds": round(time.time() - t0)}
    (EMB / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(meta)


class EmbeddingLookup:
    """text -> cached embedding (built by `embed`)."""

    def __init__(self, name):
        path = EMB / f"{name}.npy"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run `python -m l2e.encoders embed {name}` first.")
        self.E = np.load(path)
        self.index = {t: i for i, t in enumerate(json.loads((EMB / f"{name}_texts.json").read_text()))}
        self.dim = self.E.shape[1]
        self.name = name

    def __call__(self, text):
        i = self.index.get(text)
        if i is None:
            raise KeyError(f"text not in the {self.name} cache; rerun `python -m l2e.encoders embed {self.name}`")
        return self.E[i]


def export_minilm():
    """MiniLM -> ONNX (mean pooling + L2 norm inside the graph), int8 copy, and the tokenizer file."""
    import torch
    import torch.nn.functional as F
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel

    from .export_onnx import session

    repo = MODELS["minilm"]
    BOARD_MODELS.mkdir(parents=True, exist_ok=True)
    bert = AutoModel.from_pretrained(repo, attn_implementation="eager").eval()

    class Pooled(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            w = attention_mask.unsqueeze(-1).to(h.dtype)
            return F.normalize((h * w).sum(1) / w.sum(1).clamp(min=1e-9), dim=-1)

    tok_path = BOARD_MODELS / "minilm_tokenizer.json"
    shutil.copy(hf_hub_download(repo, "tokenizer.json"), tok_path)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(tok_path))
    tok.enable_truncation(MINILM_MAX_TOKENS)
    tok.no_padding()

    enc = tok.encode("Forwarding the attachment to <EXTERNAL_EMAIL> for verification")
    ids = torch.tensor([enc.ids])
    mask = torch.tensor([enc.attention_mask])
    f_path, q_path = BOARD_MODELS / "minilm.onnx", BOARD_MODELS / "minilm_int8.onnx"
    kw = dict(input_names=["input_ids", "attention_mask"], output_names=["emb"], opset_version=17,
              dynamic_axes={"input_ids": {0: "b", 1: "n"}, "attention_mask": {0: "b", 1: "n"}})
    torch.onnx.export(Pooled(bert).eval(), (ids, mask), str(f_path), dynamo=False, **kw)
    quantize_dynamic(str(f_path), str(q_path), weight_type=QuantType.QInt8)

    # parity against sentence-transformers on a sample of real event texts
    from sentence_transformers import SentenceTransformer
    texts = all_texts()[:: max(1, len(all_texts()) // 300)][:300]
    ref = SentenceTransformer(repo, device="cpu").encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    out = {}
    for k, p in (("float", f_path), ("int8", q_path)):
        s = session(p)
        cos = []
        for t, r in zip(texts, ref):
            e = tok.encode(t)
            v = s.run(None, {"input_ids": np.array([e.ids], np.int64), "attention_mask": np.array([e.attention_mask], np.int64)})[0][0]
            cos.append(float(v @ r))
        out[k] = {"min_cosine_vs_sentence_transformers": round(min(cos), 4), "mean": round(float(np.mean(cos)), 4),
                  "size_mb": round(p.stat().st_size / 1e6, 1)}
    (config.REPORTS2 / "02_export_minilm_encoder.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "export-minilm"])
    ap.add_argument("name", nargs="?", default="minilm", choices=list(MODELS))
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    config.ensure_dirs()
    if args.cmd == "embed":
        embed(args.name, args.batch_size)
    else:
        export_minilm()


if __name__ == "__main__":
    main()
