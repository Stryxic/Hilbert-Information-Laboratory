# =============================================================================
# graph_export.py — Hilbert Graph Snapshot Exporter
# =============================================================================
"""
Generates static PNG exports of the informational graph at multiple stages:

Outputs to results/hilbert_run/figures/:
    graph_full.png
    graph_100.png
    graph_200.png
    graph_500.png
    graph_<N>.png (adaptive depending on total elements)
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional, List

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

DEFAULT_EMIT = lambda *_: None


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def _log(msg, emit):
    print(msg)
    emit("log", {"message": msg})


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


# -------------------------------------------------------------------------
# Graph construction
# -------------------------------------------------------------------------

def _load_graph(results_dir: str, emit=DEFAULT_EMIT):
    """Load elements.csv + edges.csv and assemble a NetworkX graph."""
    el_path = os.path.join(results_dir, "hilbert_elements.csv")
    edges_path = os.path.join(results_dir, "edges.csv")

    if not os.path.exists(el_path) or not os.path.exists(edges_path):
        _log("[graph] Missing hilbert_elements.csv or edges.csv", emit)
        return None, None, None

    elements = pd.read_csv(el_path)
    edges = pd.read_csv(edges_path)

    if elements.empty or edges.empty:
        _log("[graph] Empty elements or edges. Skipping graph export.", emit)
        return None, None, None

    # Normalisation
    elements["element"] = elements["element"].astype(str)
    edges["source"] = edges["source"].astype(str)
    edges["target"] = edges["target"].astype(str)

    # Construct graph
    G = nx.Graph()
    for _, row in elements.iterrows():
        G.add_node(
            row["element"],
            entropy=_safe_float(row.get("mean_entropy", 0)),
            coherence=_safe_float(row.get("mean_coherence", 0)),
            tf=_safe_float(row.get("tf", 0)),
            doc=row.get("doc", "")
        )

    for _, row in edges.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            weight=_safe_float(row.get("weight", 0.0))
        )

    return G, elements, edges


# -------------------------------------------------------------------------
# Graph rendering
# -------------------------------------------------------------------------

def _render_graph(G, elements, outfile: str, emit=DEFAULT_EMIT):
    """Render the graph as a static PNG using a force layout."""

    if G is None or G.number_of_nodes() == 0:
        return

    _log(f"[graph] Rendering {outfile}", emit)

    # Node positions: stable spectral layout
    pos = nx.spring_layout(G, k=0.25, iterations=50)

    # Node styling: coherence = green intensity, entropy = red intensity
    ent = np.array([G.nodes[n].get("entropy", 0.0) for n in G.nodes])
    coh = np.array([G.nodes[n].get("coherence", 0.0) for n in G.nodes])

    ent_norm = (ent - ent.min()) / (ent.max() - ent.min() + 1e-9)
    coh_norm = (coh - coh.min()) / (coh.max() - coh.min() + 1e-9)

    node_colors = [(coh_norm[i], 0.2, ent_norm[i]) for i in range(len(ent))]

    # Figure
    plt.figure(figsize=(12, 10))
    nx.draw_networkx_edges(G, pos, width=0.4, alpha=0.3)
    nx.draw_networkx_nodes(
        G, pos,
        node_size=18,
        node_color=node_colors,
        linewidths=0
    )

    # Hide labels for performance
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outfile, dpi=180)
    plt.close()


# -------------------------------------------------------------------------
# Stage snapshots
# -------------------------------------------------------------------------

def _stage_cut_sequence(n_total: int) -> List[int]:
    """Generate stage cutoffs dynamically based on element count."""
    cuts = []

    # First 100
    if n_total > 50:
        cuts.append(100)

    # 200, 300, 500 if available
    for c in [200, 300, 500]:
        if n_total > c:
            cuts.append(c)

    # Final 1000-tier
    if n_total > 1000:
        cuts.extend([1000, 1500, 2000])

    return cuts


def export_graph_snapshots(results_dir: str, emit=DEFAULT_EMIT):
    """Entry point for orchestrator."""

    out_fig = Path(results_dir) / "figures"
    out_fig.mkdir(exist_ok=True)

    G, elements, edges = _load_graph(results_dir, emit)
    if G is None:
        return

    n_total = G.number_of_nodes()
    _log(f"[graph] Total nodes: {n_total}", emit)

    # -----------------------------------------------------------------
    # Full graph export
    # -----------------------------------------------------------------
    full_png = out_fig / "graph_full.png"
    _render_graph(G, elements, str(full_png), emit)

    # -----------------------------------------------------------------
    # Stage snapshots (increasing number of elements)
    # -----------------------------------------------------------------
    cuts = _stage_cut_sequence(n_total)

    ordered_nodes = list(G.nodes)

    for cutoff in cuts:
        sub_nodes = ordered_nodes[:cutoff]
        H = G.subgraph(sub_nodes)
        out = out_fig / f"graph_{cutoff}.png"
        _render_graph(H, elements, str(out), emit)

    emit("artifact", {"path": str(full_png), "kind": "graph_full"})
    for cutoff in cuts:
        emit("artifact", {
            "path": str(out_fig / f"graph_{cutoff}.png"),
            "kind": "graph_stage"
        })
    _log("[graph] Graph snapshots complete.", emit)
