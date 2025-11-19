"""
Element Language Model (ELM)
---------------------------------------

A compact Transformer-based language model built entirely from
Hilbert informational elements.

This file provides:
- Vocabulary construction from hilbert_elements.csv
- Sequence building from span_element_fusion.csv
- Transformer model (PyTorch, optional)
- Training loop
- Inference helpers (completion and scoring)
- Pipeline stage wrapper

If PyTorch is not available, all LM functions will safely no-op
and log a warning instead of crashing the API.
"""

import os
import json
from collections import Counter

import pandas as pd

# Try to import PyTorch, but do not crash if it is missing
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object  # dummy base
    DataLoader = object
    TORCH_AVAILABLE = False


# ======================================================================
# 1. VOCAB BUILDER
# ======================================================================

def build_element_vocab(results_dir: str, min_freq: int = 1, emit=lambda *_: None):
    """
    Build an element vocabulary from hilbert_elements.csv.

    Returns a dict mapping element string -> integer id.
    """
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        emit("warn", {"stage": "element_lm", "message": "hilbert_elements.csv missing"})
        return None

    df = pd.read_csv(path)
    if "element" not in df.columns:
        emit("warn", {"stage": "element_lm", "message": "hilbert_elements.csv missing 'element' column"})
        return None

    counts = Counter(df["element"].astype(str))

    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
    }
    for el, c in counts.items():
        if c >= min_freq:
            vocab[el] = len(vocab)

    emit("log", {"stage": "element_lm", "message": f"Vocab size: {len(vocab)}"})
    return vocab


# ======================================================================
# 2. SEQUENCE BUILDER
# ======================================================================

def build_sequences(results_dir: str, vocab: dict, max_len: int = 128, emit=lambda *_: None):
    """
    Build element sequences per document from span_element_fusion.csv.
    Each sequence is a list of vocab ids with BOS and EOS markers.
    """
    fusion_path = os.path.join(results_dir, "span_element_fusion.csv")
    if not os.path.exists(fusion_path):
        emit("warn", {"stage": "element_lm", "message": "span_element_fusion.csv missing"})
        return []

    fdf = pd.read_csv(fusion_path)
    required_cols = {"doc", "span_id", "element"}
    if not required_cols.issubset(set(fdf.columns)):
        emit(
            "warn",
            {
                "stage": "element_lm",
                "message": f"span_element_fusion.csv missing columns: {required_cols - set(fdf.columns)}",
            },
        )
        return []

    fdf["element"] = fdf["element"].astype(str)
    fdf = fdf.sort_values(["doc", "span_id"])

    sequences = []
    for doc, group in fdf.groupby("doc"):
        els = [vocab.get(e, vocab["<unk>"]) for e in group["element"].tolist()]
        if not els:
            continue

        for i in range(0, len(els), max_len):
            chunk = els[i : i + max_len]
            if len(chunk) < 4:
                continue
            seq = [vocab["<bos>"]] + chunk + [vocab["<eos>"]]
            sequences.append({"doc": doc, "tokens": seq})

    emit("log", {"stage": "element_lm", "message": f"Built {len(sequences)} sequences"})
    return sequences


# ======================================================================
# 3. TRANSFORMER DEFINITION (only if PyTorch is available)
# ======================================================================

if TORCH_AVAILABLE:

    class ElementTransformerConfig:
        def __init__(
            self,
            vocab_size: int,
            d_model: int = 256,
            n_heads: int = 4,
            n_layers: int = 4,
            d_ff: int = 1024,
            max_len: int = 256,
            dropout: float = 0.1,
        ):
            self.vocab_size = vocab_size
            self.d_model = d_model
            self.n_heads = n_heads
            self.n_layers = n_layers
            self.d_ff = d_ff
            self.max_len = max_len
            self.dropout = dropout


    class ElementTransformer(nn.Module):
        def __init__(self, cfg: "ElementTransformerConfig"):
            super().__init__()
            self.cfg = cfg

            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=cfg.n_layers
            )

            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            # input_ids: [B, T]
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            x = self.token_emb(input_ids) + self.pos_emb(pos)

            # causal mask (no looking ahead)
            mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))

            x = self.transformer(x, mask)
            logits = self.lm_head(x)
            return logits


    # ==================================================================
    # 4. DATASET + TRAIN LOOP
    # ==================================================================

    class ElementSequenceDataset(Dataset):
        def __init__(self, sequences, max_len: int, pad_id: int):
            self.sequences = sequences
            self.max_len = max_len
            self.pad_id = pad_id

        def __len__(self) -> int:
            return len(self.sequences)

        def __getitem__(self, idx: int):
            seq = self.sequences[idx]["tokens"]
            if len(seq) > self.max_len:
                seq = seq[: self.max_len]

            input_ids = seq[:-1]       # BOS + tokens
            target_ids = seq[1:]       # tokens + EOS

            pad_len = self.max_len - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [self.pad_id] * pad_len
                target_ids = target_ids + [-100] * pad_len  # ignore index

            return torch.tensor(input_ids), torch.tensor(target_ids)


def train_element_transformer(
    results_dir: str,
    emit=lambda *_: None,
    epochs: int = 3,
    batch_size: int = 32,
):
    """
    Train the element language model on the given results directory.

    If PyTorch is not available this logs a warning and returns immediately.
    """
    if not TORCH_AVAILABLE:
        emit(
            "warn",
            {
                "stage": "element_lm",
                "message": "PyTorch not available - skipping element LM training.",
            },
        )
        return

    vocab = build_element_vocab(results_dir, emit=emit)
    if vocab is None:
        return

    sequences = build_sequences(results_dir, vocab, emit=emit)
    if not sequences:
        emit("warn", {"stage": "element_lm", "message": "No sequences to train on"})
        return

    cfg = ElementTransformerConfig(vocab_size=len(vocab), max_len=256)
    model = ElementTransformer(cfg)

    dataset = ElementSequenceDataset(
        sequences, max_len=cfg.max_len, pad_id=vocab["<pad>"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if len(loader) == 0:
        emit("warn", {"stage": "element_lm", "message": "Empty DataLoader - nothing to train"})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        total = 0.0

        for input_ids, target_ids in loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            optim.zero_grad()
            logits = model(input_ids)
            loss = criterion(
                logits.view(-1, cfg.vocab_size),
                target_ids.view(-1),
            )
            loss.backward()
            optim.step()

            total += loss.item()

        emit(
            "log",
            {
                "stage": "element_lm",
                "epoch": epoch,
                "loss": total / max(1, len(loader)),
            },
        )

    # Save model and vocab
    model_path = os.path.join(results_dir, "element_lm.pt")
    vocab_path = os.path.join(results_dir, "element_vocab.json")

    torch.save(model.state_dict(), model_path)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, indent=2)

    emit("log", {"stage": "element_lm", "message": "Training complete."})


# ======================================================================
# 5. INFERENCE HELPERS
# ======================================================================

def load_element_lm(results_dir: str, emit=lambda *_: None):
    """
    Load the trained element language model and vocabulary.

    Returns (model, vocab, inverse_vocab) or (None, None, None) if missing
    or if PyTorch is not available.
    """
    if not TORCH_AVAILABLE:
        emit(
            "warn",
            {
                "stage": "element_lm",
                "message": "PyTorch not available - cannot load element LM.",
            },
        )
        return None, None, None

    vocab_path = os.path.join(results_dir, "element_vocab.json")
    model_path = os.path.join(results_dir, "element_lm.pt")

    if not (os.path.exists(vocab_path) and os.path.exists(model_path)):
        emit(
            "warn",
            {
                "stage": "element_lm",
                "message": "element_lm.pt or element_vocab.json not found.",
            },
        )
        return None, None, None

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    ivocab = {i: tok for tok, i in vocab.items()}

    cfg = ElementTransformerConfig(vocab_size=len(vocab))
    model = ElementTransformer(cfg)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return model, vocab, ivocab


def suggest_next_elements(prefix, results_dir: str, k: int = 5, emit=lambda *_: None):
    """
    Given a prefix of element tokens (strings), suggest the next k elements.

    Returns a list of (element, probability) pairs.
    """
    model, vocab, ivocab = load_element_lm(results_dir, emit)
    if model is None or vocab is None or ivocab is None:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids = [vocab.get(tok, vocab.get("<unk>", 0)) for tok in prefix]
    if not ids:
        ids = [vocab.get("<bos>", 1)]

    inp = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)[0, -1]  # last token logits

    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=min(k, probs.shape[0]))

    results = []
    for idx, prob in zip(topk.indices, topk.values):
        token_id = int(idx.item())
        token = ivocab.get(token_id, "<unk>")
        results.append((token, float(prob.item())))
    return results


def score_element_sequence(seq, results_dir: str, emit=lambda *_: None) -> float:
    """
    Compute the average log probability of a sequence of element tokens.

    Higher values indicate a more typical sequence under the element LM.
    """
    model, vocab, ivocab = load_element_lm(results_dir, emit)
    if model is None or vocab is None:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids = [vocab.get(tok, vocab.get("<unk>", 0)) for tok in seq]
    if len(ids) < 2:
        return 0.0

    inp = torch.tensor(ids[:-1], dtype=torch.long).unsqueeze(0).to(device)
    tgt = torch.tensor(ids[1:], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        log_probs = torch.log_softmax(logits, dim=-1)

    total = 0.0
    count = 0
    for t, target in enumerate(tgt[0]):
        lp = log_probs[0, t, target].item()
        total += lp
        count += 1

    if count == 0:
        return 0.0
    return total / count


# ======================================================================
# 6. PIPELINE STAGE WRAPPER
# ======================================================================

def run_element_lm_stage(results_dir: str, emit=lambda *_: None):
    """
    Pipeline stage entry point.

    Safe to call even if PyTorch is not installed. In that case it will
    log a warning and return without failing the run.
    """
    emit("log", {"stage": "element_lm", "message": "Starting element LM training..."})
    train_element_transformer(results_dir, emit=emit)
    emit("log", {"stage": "element_lm", "message": "Element LM stage complete."})
