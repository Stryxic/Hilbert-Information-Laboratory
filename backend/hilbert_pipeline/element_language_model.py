# =============================================================================
# hilbert_pipeline/element_language_models.py — Element Language Model (v3.1)
# =============================================================================
"""
Element Language Model (ELM)
============================

This module defines a **compact Transformer-based language model** whose
vocabulary is composed entirely of *Hilbert informational elements*.  
It supports:

- Vocabulary construction from :file:`hilbert_elements.csv`
- Sequence building from :file:`span_element_fusion.csv`
- Optional PyTorch-based Transformer LM
- Training and checkpointing
- Completion (next-element prediction)
- Sequence scoring (log-likelihood)
- Orchestrator stage wrapper

The ELM is conceptually orthogonal to the LSA/molecule/roots pipeline,
but can be used for:

- detecting **semantically atypical sequences** of elements,
- ranking candidate informational pathways,
- generating cluster-consistent element completions,
- supporting the epistemic signatures system.

If PyTorch is not installed (or GPU unavailable), this module degrades
**gracefully** to a fully safe no-op layer that logs warnings but never
breaks the pipeline.

Public API
----------

Vocabulary and sequences:
    - :func:`build_element_vocab`  
    - :func:`build_sequences`

Training/inference:
    - :func:`train_element_transformer`  
    - :func:`load_element_lm`  
    - :func:`suggest_next_elements`  
    - :func:`score_element_sequence`

Pipeline wrapper:
    - :func:`run_element_lm_stage`
"""

from __future__ import annotations

import os
import json
from collections import Counter
from typing import Dict, List, Any, Tuple

import pandas as pd

# -----------------------------------------------------------------------------
# PyTorch optional dependency
# -----------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    Dataset = object
    DataLoader = object
    TORCH_AVAILABLE = False


# =============================================================================
# 1. Vocabulary Builder
# =============================================================================

def build_element_vocab(
    results_dir: str,
    min_freq: int = 1,
    emit=lambda *_: None,
) -> Dict[str, int] | None:
    """
    Construct a token→ID vocabulary from :file:`hilbert_elements.csv`.

    Parameters
    ----------
    results_dir : str
        Directory containing Hilbert run outputs.
    min_freq : int
        Only elements with frequency ≥ min_freq are included.
    emit : callable
        Orchestrator-compatible logger.

    Returns
    -------
    dict or None
        Mapping ``element -> int`` including special tokens
        (``<pad>``, ``<bos>``, ``<eos>``, ``<unk>``),  
        or None if elements cannot be loaded.

    Notes
    -----
    This vocabulary becomes the core alphabet for the transformer LM.
    """
    path = os.path.join(results_dir, "hilbert_elements.csv")
    if not os.path.exists(path):
        emit("warn", {"stage": "element_lm", "message": "hilbert_elements.csv missing"})
        return None

    df = pd.read_csv(path)
    if "element" not in df.columns:
        emit("warn", {"stage": "element_lm", "message": "'element' column missing"})
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


# =============================================================================
# 2. Sequence Builder
# =============================================================================

def build_sequences(
    results_dir: str,
    vocab: Dict[str, int],
    max_len: int = 128,
    emit=lambda *_: None,
) -> List[Dict[str, Any]]:
    """
    Build per-document element sequences using :file:`span_element_fusion.csv`.

    Each sequence is tokenized using the ELM vocabulary and wrapped in
    ``<bos> ... <eos>`` markers.

    Parameters
    ----------
    results_dir : str
        Hilbert results directory.
    vocab : dict
        Mapping produced by :func:`build_element_vocab`.
    max_len : int
        Maximum unpadded sequence length (after BOS/EOS insertion).
    emit : callable
        Logging hook.

    Returns
    -------
    list of dict
        One record per sequence:

        ``{"doc": str, "tokens": List[int]}``

    Notes
    -----
    - Sequences are sorted by (doc, span_id).
    - Very short sequences (< 4 tokens) are dropped.
    """
    fusion_path = os.path.join(results_dir, "span_element_fusion.csv")
    if not os.path.exists(fusion_path):
        emit("warn", {"stage": "element_lm", "message": "span_element_fusion.csv missing"})
        return []

    fdf = pd.read_csv(fusion_path)
    if not {"doc", "span_id", "element"}.issubset(fdf.columns):
        emit("warn", {"stage": "element_lm",
                      "message": "fusion CSV missing required columns"})
        return []

    fdf = fdf.sort_values(["doc", "span_id"])
    sequences = []

    for doc, group in fdf.groupby("doc"):
        ids = [vocab.get(e, vocab["<unk>"]) for e in group["element"].astype(str)]
        if not ids:
            continue

        # Chunk into max_len windows
        for i in range(0, len(ids), max_len):
            chunk = ids[i:i + max_len]
            if len(chunk) < 4:
                continue
            seq = [vocab["<bos>"]] + chunk + [vocab["<eos>"]]
            sequences.append({"doc": doc, "tokens": seq})

    emit("log", {"stage": "element_lm", "message": f"Built {len(sequences)} sequences"})
    return sequences


# =============================================================================
# 3. Transformer Definition (PyTorch-only)
# =============================================================================

if TORCH_AVAILABLE:

    class ElementTransformerConfig:
        """
        Lightweight configuration object for :class:`ElementTransformer`.
        """

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
        """
        Minimal Transformer encoder used as an element language model.
        Causal masking is applied to enforce left-to-right structure.
        """

        def __init__(self, cfg: "ElementTransformerConfig"):
            super().__init__()
            self.cfg = cfg

            self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
            self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)

            layer = nn.TransformerEncoderLayer(
                d_model=cfg.d_model,
                nhead=cfg.n_heads,
                dim_feedforward=cfg.d_ff,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            input_ids : Tensor [B, T]
                Batch of token sequences.

            Returns
            -------
            Tensor [B, T, vocab_size]
                Next-token logits.
            """
            B, T = input_ids.shape
            pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
            x = self.token_emb(input_ids) + self.pos_emb(pos)

            # Causal mask
            mask = torch.triu(torch.ones(T, T, device=input_ids.device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float("-inf"))

            x = self.transformer(x, mask)
            return self.lm_head(x)


    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    class ElementSequenceDataset(Dataset):
        """Simple LM dataset of input→target token sequences."""

        def __init__(self, sequences, max_len: int, pad_id: int):
            self.sequences = sequences
            self.max_len = max_len
            self.pad_id = pad_id

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            seq = list(self.sequences[idx]["tokens"])
            seq = seq[:self.max_len]

            inp = seq[:-1]
            tgt = seq[1:]

            pad = self.max_len - len(inp)
            if pad > 0:
                inp += [self.pad_id] * pad
                tgt += [-100] * pad

            return torch.tensor(inp), torch.tensor(tgt)


# =============================================================================
# 4. Training
# =============================================================================

def train_element_transformer(
    results_dir: str,
    emit=lambda *_: None,
    epochs: int = 3,
    batch_size: int = 32,
):
    """
    Train the element Transformer LM.

    If PyTorch is unavailable, this function logs a warning and returns
    without error.

    Parameters
    ----------
    results_dir : str
        Hilbert results directory.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for DataLoader.

    Returns
    -------
    None
    """
    if not TORCH_AVAILABLE:
        emit("warn", {"stage": "element_lm",
                      "message": "PyTorch not installed – skipping LM training"})
        return

    vocab = build_element_vocab(results_dir, emit=emit)
    if not vocab:
        return

    sequences = build_sequences(results_dir, vocab, emit=emit)
    if not sequences:
        emit("warn", {"stage": "element_lm", "message": "No sequences found"})
        return

    cfg = ElementTransformerConfig(vocab_size=len(vocab))
    model = ElementTransformer(cfg)

    dataset = ElementSequenceDataset(
        sequences, max_len=cfg.max_len, pad_id=vocab["<pad>"]
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if len(loader) == 0:
        emit("warn", {"stage": "element_lm", "message": "Empty DataLoader"})
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for inp, tgt in loader:
            inp = inp.to(device)
            tgt = tgt.to(device)

            optim.zero_grad()
            logits = model(inp)
            loss = loss_fn(logits.view(-1, cfg.vocab_size), tgt.view(-1))
            loss.backward()
            optim.step()
            epoch_loss += loss.item()

        emit("log", {
            "stage": "element_lm",
            "epoch": epoch,
            "loss": epoch_loss / len(loader),
        })

    # Save model
    torch.save(model.state_dict(), os.path.join(results_dir, "element_lm.pt"))
    with open(os.path.join(results_dir, "element_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    emit("log", {"stage": "element_lm", "message": "Training complete"})


# =============================================================================
# 5. Inference Helpers
# =============================================================================

def load_element_lm(
    results_dir: str,
    emit=lambda *_: None,
) -> Tuple[Any, Dict[str, int], Dict[int, str]]:
    """
    Load the trained ELM model and vocabulary.

    Returns
    -------
    (model, vocab, inverse_vocab)
        model : nn.Module or None
        vocab : dict or None
        inverse_vocab : dict or None
    """
    if not TORCH_AVAILABLE:
        emit("warn", {"stage": "element_lm",
                      "message": "PyTorch unavailable – cannot load LM"})
        return None, None, None

    model_path = os.path.join(results_dir, "element_lm.pt")
    vocab_path = os.path.join(results_dir, "element_vocab.json")

    if not (os.path.exists(model_path) and os.path.exists(vocab_path)):
        emit("warn", {"stage": "element_lm",
                      "message": "Missing LM checkpoint or vocab"})
        return None, None, None

    with open(vocab_path) as f:
        vocab = json.load(f)
    ivocab = {i: tok for tok, i in vocab.items()}

    cfg = ElementTransformerConfig(vocab_size=len(vocab))
    model = ElementTransformer(cfg)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    return model, vocab, ivocab


def suggest_next_elements(
    prefix: List[str],
    results_dir: str,
    k: int = 5,
    emit=lambda *_: None,
) -> List[Tuple[str, float]]:
    """
    Suggest the top-k next elements given a prefix.

    Parameters
    ----------
    prefix : list of str
        Sequence of element tokens.
    results_dir : str
        Directory containing the trained model.
    k : int
        Number of suggestions.

    Returns
    -------
    list of (token, probability)
    """
    model, vocab, ivocab = load_element_lm(results_dir, emit)
    if model is None:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ids = [vocab.get(tok, vocab["<unk>"]) for tok in prefix] or [vocab["<bos>"]]
    inp = torch.tensor(ids).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = torch.topk(probs, k=min(k, probs.numel()))

    out = []
    for idx, prob in zip(top.indices, top.values):
        token = ivocab.get(int(idx), "<unk>")
        out.append((token, float(prob)))
    return out


def score_element_sequence(
    seq: List[str],
    results_dir: str,
    emit=lambda *_: None,
) -> float:
    """
    Compute mean log-probability of a token sequence.

    Parameters
    ----------
    seq : list of str
        Element token sequence.
    results_dir : str
        Directory containing trained ELM weights.

    Returns
    -------
    float
        Average log probability under the LM.
    """
    model, vocab, ivocab = load_element_lm(results_dir, emit)
    if model is None:
        return 0.0

    ids = [vocab.get(tok, vocab["<unk>"]) for tok in seq]
    if len(ids) < 2:
        return 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inp = torch.tensor(ids[:-1]).unsqueeze(0).to(device)
    tgt = torch.tensor(ids[1:]).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(inp)
        logp = torch.log_softmax(logits, dim=-1)

    total = 0.0
    count = 0
    for i, target in enumerate(tgt[0]):
        lp = logp[0, i, target].item()
        total += lp
        count += 1

    return total / max(1, count)


# =============================================================================
# 6. Pipeline Wrapper
# =============================================================================

def run_element_lm_stage(results_dir: str, emit=lambda *_: None) -> None:
    """
    Orchestrator entry point for the Element Language Model stage.

    - Safe if dependencies missing  
    - Emits structured pipeline logs  
    - Writes ``element_lm.pt`` and ``element_vocab.json`` if training succeeds
    """
    emit("log", {"stage": "element_lm", "event": "start"})
    train_element_transformer(results_dir, emit=emit)
    emit("log", {"stage": "element_lm", "event": "end"})
