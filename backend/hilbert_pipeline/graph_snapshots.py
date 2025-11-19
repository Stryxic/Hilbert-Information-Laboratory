# =============================================================================
# graph_snapshots.py — Hilbert Graph Snapshot Generator (compound-aware, v2)
# =============================================================================
#
# Generates progressive graph visualisations during the Hilbert pipeline:
#
#   graph_5.png, graph_10.png, ..., graph_full.png
#
# Design goals (aggressive / research-grade):
#   - Focus on the largest, most connected clusters / molecules
#   - Compact, stable layout (spring + Kamada–Kawai hybrid)
#   - Node colour:
#         1) compound_id from informational_compounds.json (if present)
#         2) root_element from element_roots.csv (if present)
#         3) connected component id (fallback)
#   - Node size:
#         mix of normalised mean_coherence, tf and degree
#   - Labels:
#         two-tier hierarchy; bigger nodes always labelled
#         outlined text for legibility on dark background
#   - Layout:
#         single layout on filtered graph, reused for all snapshots
#   - Visual extras:
#         - halo / glow behind important nodes
#         - edge widths ∝ weight, alpha tuned to density
#         - convex hulls around large compounds (if SciPy available)
#         - camera framing using percentile cropping
#
# Public API (used by orchestrator):
#
#     from hilbert_pipeline.graph_snapshots import generate_graph_snapshots
#     generate_graph_snapshots(str(out_path), emit=ctx.emit)
#
# =============================================================================

from __future__ import annotations

import json
import os
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# optional: convex hulls for compound envelopes
try:
    from scipy.spatial import ConvexHull  # type: ignore

    _HAS_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SCIPY = False

# optional: text outline for legibility
try:
    import matplotlib.patheffects as patheffects  # type: ignore

    _HAS_PATH_EFFECTS = True
except Exception:  # pragma: no cover
    _HAS_PATH_EFFECTS = False


# -----------------------------------------------------------------------------#
# Global defaults
# -----------------------------------------------------------------------------#

DEFAULT_EMIT = lambda *_: None  # noqa: E731

# Minimum connected-component size to be considered "interesting".
# Nodes in smaller components are excluded from the progressive snapshots.
MIN_COMPONENT_SIZE = 4

# Minimum number of nodes for which to draw a convex hull for a compound
MIN_HULL_SIZE = 6


# -----------------------------------------------------------------------------#
# Logging helpers
# -----------------------------------------------------------------------------#

def _log(msg: str, emit=DEFAULT_EMIT) -> None:
    print(msg)
    emit("log", {"message": msg})


def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


# -----------------------------------------------------------------------------#
# Loading helpers
# -----------------------------------------------------------------------------#

def _load_elements(elements_csv: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    """
    Load hilbert_elements.csv and collapse to one row per element.

    We aggregate stats at the element level so that later lookups like
    df.loc[el]["mean_coherence"] always return a scalar, not a Series.

    Returned columns (per element):
        element, mean_coherence, tf, doc_freq
    """
    if not os.path.exists(elements_csv):
        _log(f"[graphs] hilbert_elements.csv missing: {elements_csv}", emit)
        return pd.DataFrame()

    df = pd.read_csv(elements_csv)

    # Ensure element column
    if "element" not in df.columns and "token" in df.columns:
        df["element"] = df["token"]

    if "element" not in df.columns:
        _log("[graphs] No 'element' column in hilbert_elements.csv", emit)
        return pd.DataFrame()

    df["element"] = df["element"].astype(str)

    # Coherence and tf / df
    if "mean_coherence" not in df.columns:
        df["mean_coherence"] = 0.0
    if "tf" not in df.columns:
        df["tf"] = 1.0
    if "doc_freq" not in df.columns and "df" in df.columns:
        df["doc_freq"] = df["df"]
    if "doc_freq" not in df.columns:
        df["doc_freq"] = 1.0

    for col in ("mean_coherence", "tf", "doc_freq"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Collapse to one row per element
    agg = (
        df.groupby("element", as_index=False)[["mean_coherence", "tf", "doc_freq"]]
        .agg(
            {
                "mean_coherence": "mean",
                "tf": "sum",
                "doc_freq": "sum",
            }
        )
        .reset_index(drop=True)
    )

    return agg


def _load_edges(edges_csv: str, emit=DEFAULT_EMIT) -> pd.DataFrame:
    if not os.path.exists(edges_csv):
        _log(f"[graphs] edges.csv missing: {edges_csv}", emit)
        return pd.DataFrame()

    df = pd.read_csv(edges_csv)
    for col in ("source", "target"):
        if col not in df.columns:
            _log(f"[graphs] edges.csv missing '{col}' column", emit)
            return pd.DataFrame()

    df["source"] = df["source"].astype(str)
    df["target"] = df["target"].astype(str)
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["weight"] = df["weight"].apply(_safe_float)

    # Drop self-loops and obviously bad rows
    df = df[df["source"] != df["target"]]

    return df


def _load_root_clusters(results_dir: str, emit=DEFAULT_EMIT) -> Dict[str, str]:
    """element -> root_element mapping from element_roots.csv (if available)."""
    path = os.path.join(results_dir, "element_roots.csv")
    if not os.path.exists(path):
        return {}

    try:
        df = pd.read_csv(path)
    except Exception as e:
        _log(f"[graphs] Failed to read element_roots.csv: {e}", emit)
        return {}

    if "element" not in df.columns or "root_element" not in df.columns:
        return {}

    mapping = {}
    for _, row in df[["element", "root_element"]].dropna().iterrows():
        mapping[str(row["element"])] = str(row["root_element"])
    return mapping


def _load_compound_membership(results_dir: str, emit=DEFAULT_EMIT) -> Dict[str, str]:
    """
    element -> compound_id mapping from informational_compounds.json (if available).
    """
    path = os.path.join(results_dir, "informational_compounds.json")
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        _log(f"[graphs] Failed to read informational_compounds.json: {e}", emit)
        return {}

    if isinstance(raw, dict):
        comps = list(raw.values())
    elif isinstance(raw, list):
        comps = raw
    else:
        comps = []

    mapping: Dict[str, str] = {}
    for comp in comps:
        cid = str(comp.get("compound_id") or comp.get("id") or "C?")
        elems = comp.get("elements") or comp.get("element_ids") or []
        if isinstance(elems, str):
            elems = [e.strip() for e in elems.split(",") if e.strip()]
        for el in elems:
            mapping[str(el)] = cid

    return mapping


# optional: compound temperature (if present) for subtle colour tinting
def _load_compound_temperature(results_dir: str, emit=DEFAULT_EMIT) -> Dict[str, float]:
    path = os.path.join(results_dir, "informational_compounds.json")
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as e:
        _log(f"[graphs] Failed to read compound temperatures: {e}", emit)
        return {}

    temps: Dict[str, float] = {}
    if isinstance(raw, dict):
        items = raw.values()
    elif isinstance(raw, list):
        items = raw
    else:
        items = []

    for comp in items:
        cid = str(comp.get("compound_id") or comp.get("id") or "C?")
        t = comp.get("temperature")
        if t is None:
            continue
        temps[cid] = _safe_float(t, 0.5)

    return temps


# -----------------------------------------------------------------------------#
# Cluster / colour helpers
# -----------------------------------------------------------------------------#

def _build_cluster_ids(
    G: nx.Graph,
    root_map: Dict[str, str],
    compound_map: Dict[str, str],
) -> Dict[str, str]:
    """
    Decide a cluster id for each element, in order of preference:
      1. compound_id (informational_compounds.json)
      2. root_element  (element_roots.csv)
      3. connected component id
    """
    cluster_id: Dict[str, str] = {}

    # 1) compounds
    for n in G.nodes():
        el = str(n)
        if el in compound_map:
            cluster_id[el] = compound_map[el]

    # 2) root elements
    for n in G.nodes():
        el = str(n)
        if el in cluster_id:
            continue
        if el in root_map:
            cluster_id[el] = root_map[el]

    # 3) connected components
    comp_prefix = "K"
    for idx, comp in enumerate(nx.connected_components(G), start=1):
        cid = f"{comp_prefix}{idx:03d}"
        for el in comp:
            el = str(el)
            cluster_id.setdefault(el, cid)

    return cluster_id


def _blend(c1, c2, alpha: float) -> Tuple[float, float, float, float]:
    """Linear RGBA blend (tuples in, tuple out)."""
    r1, g1, b1, a1 = c1
    r2, g2, b2, a2 = c2
    return (
        (1 - alpha) * r1 + alpha * r2,
        (1 - alpha) * g1 + alpha * g2,
        (1 - alpha) * b1 + alpha * b2,
        (1 - alpha) * a1 + alpha * a2,
    )


def _make_color_map(
    cluster_ids: Dict[str, str],
    compound_temps: Dict[str, float],
) -> Dict[str, Any]:
    """
    cluster_id -> RGBA tuple.

    - base palette from tab20
    - if cluster_id is a compound_id with a temperature in compound_temps,
      we apply a warm/cool tint, so hot compounds appear slightly reddish,
      cold compounds slightly bluish.
    """
    unique_clusters = sorted(set(cluster_ids.values()))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_clusters), 1))

    # normalise temperatures to [0, 1]
    vals = np.array(list(compound_temps.values()), dtype=float)
    if vals.size > 0:
        t_min, t_max = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max - t_min < 1e-9:
            t_min, t_max = 0.0, 1.0
    else:
        t_min, t_max = 0.0, 1.0

    def norm_t(x: float) -> float:
        if t_max - t_min < 1e-9:
            return 0.5
        return max(0.0, min(1.0, (x - t_min) / (t_max - t_min + 1e-9)))

    color_map: Dict[str, Any] = {}

    for i, cid in enumerate(unique_clusters):
        base_arr = cmap(i % cmap.N)
        base = tuple(float(c) for c in base_arr)

        if cid in compound_temps:
            t = norm_t(compound_temps[cid])
            # blend between cool blue and warm red
            cool = (0.37, 0.65, 0.98, 1.0)   # light blue
            warm = (0.98, 0.55, 0.47, 1.0)   # light red
            temp_col = _blend(cool, warm, t)
            color_map[cid] = _blend(base, temp_col, 0.55)
        else:
            color_map[cid] = base

    return color_map


# -----------------------------------------------------------------------------#
# Node sizing and label selection
# -----------------------------------------------------------------------------#

def _normalise(series: pd.Series) -> pd.Series:
    arr = series.to_numpy(dtype=float)
    if arr.size == 0:
        return pd.Series(np.zeros_like(arr), index=series.index)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mn) or not np.isfinite(mx) or mx - mn < 1e-9:
        return pd.Series(np.zeros_like(arr), index=series.index)
    return pd.Series((arr - mn) / (mx - mn), index=series.index)


def _compute_node_sizes(G: nx.Graph, df: pd.DataFrame) -> Dict[str, float]:
    """
    Node size ~ mix of normalised coherence, tf and degree.
    df is indexed by element.
    """
    # degree (raw)
    deg = dict(G.degree())
    deg_series = pd.Series({str(k): float(v) for k, v in deg.items()})

    # align coherence and tf with graph nodes
    coh_series = pd.Series(
        {
            str(el): float(df.loc[el]["mean_coherence"])
            if el in df.index
            else 0.0
            for el in G.nodes()
        }
    )

    tf_series = pd.Series(
        {
            str(el): float(df.loc[el]["tf"])
            if el in df.index
            else 0.0
            for el in G.nodes()
        }
    )

    nd = _normalise(deg_series)
    nc = _normalise(coh_series)
    nt = _normalise(tf_series)

    sizes: Dict[str, float] = {}
    for node in G.nodes():
        key = str(node)
        score = 0.4 * nc.get(key, 0.0) + 0.35 * nt.get(key, 0.0) + 0.25 * nd.get(
            key, 0.0
        )
        # base radius + scaled; exponent emphasises big players
        sizes[key] = 40.0 + 260.0 * (score ** 1.25)
    return sizes


def _pick_label_nodes(
    node_sizes: Dict[str, float],
    n_nodes: int
) -> Tuple[List[str], List[str]]:
    """
    Choose two tiers of label nodes:

      - primary: biggest nodes that should always be labelled
      - secondary: additional labels, thinned for readability
    """
    if not node_sizes:
        return [], []

    ordered = sorted(node_sizes.items(), key=lambda kv: kv[1], reverse=True)
    names = [n for n, _ in ordered]

    # simple heuristics
    if n_nodes <= 150:
        primary_n = min(30, len(names))
        secondary_n = min(90, len(names))  # total
    elif n_nodes <= 400:
        primary_n = min(35, len(names))
        secondary_n = min(70, len(names))
    else:
        primary_n = min(40, len(names))
        secondary_n = min(55, len(names))

    primary = names[:primary_n]
    secondary = names[primary_n:secondary_n]

    return primary, secondary


# -----------------------------------------------------------------------------#
# Layout & drawing
# -----------------------------------------------------------------------------#

def _compute_layout(G: nx.Graph) -> Dict[str, Tuple[float, float]]:
    """
    Hybrid layout:

      1) spring_layout to get global shape
      2) kamada_kawai_layout to refine locally

    This tends to produce compact, visually pleasing clusters.
    """
    n = max(G.number_of_nodes(), 1)
    # first pass: rough spring layout
    k = 0.9 / np.sqrt(n)
    pos = nx.spring_layout(G, k=k, iterations=40, seed=42, weight="weight")

    try:
        # second pass: fine-tune with kamada_kawai using initial positions
        pos = nx.kamada_kawai_layout(G, pos=pos, weight="weight")
    except Exception:
        # fall back to spring-only if something goes wrong
        pass

    return pos


def _camera_frame(pos: Dict[str, Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Compute a tight camera frame using percentile cropping to avoid extreme
    outliers dominating the canvas.
    """
    if not pos:
        return -1.0, 1.0, -1.0, 1.0

    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if xs.size == 0:
        return -1.0, 1.0, -1.0, 1.0

    x_min, x_max = np.percentile(xs, [2, 98])
    y_min, y_max = np.percentile(ys, [2, 98])

    # small margin
    dx = (x_max - x_min) * 0.1 + 1e-9
    dy = (y_max - y_min) * 0.1 + 1e-9

    return float(x_min - dx), float(x_max + dx), float(y_min - dy), float(y_max + dy)


def _draw_compound_hulls(
    ax,
    pos: Dict[str, Tuple[float, float]],
    compound_map: Dict[str, str],
    node_set: List[str],
) -> None:
    """
    Draw soft convex hulls around sufficiently large compounds.
    """
    if not _HAS_SCIPY:
        return

    if not node_set or not pos:
        return

    comp_to_nodes: Dict[str, List[str]] = {}
    for n in node_set:
        cid = compound_map.get(str(n))
        if cid:
            comp_to_nodes.setdefault(cid, []).append(str(n))

    for cid, nodes in comp_to_nodes.items():
        if len(nodes) < MIN_HULL_SIZE:
            continue

        pts_list = [pos[n] for n in nodes if n in pos]
        if len(pts_list) < 3:
            continue

        pts = np.array(pts_list, dtype=float)
        if pts.shape[0] < 3:
            continue

        try:
            hull = ConvexHull(pts)
        except Exception:
            continue

        poly = pts[hull.vertices]

        ax.fill(
            poly[:, 0],
            poly[:, 1],
            facecolor=(0.5, 0.8, 1.0, 0.08),
            edgecolor=(0.65, 0.9, 1.0, 0.30),
            linewidth=1.2,
            zorder=0,
        )


def _draw_legend(
    fig,
    color_map: Dict[str, Any],
    compound_temps: Dict[str, float],
    cluster_ids: Dict[str, str],
) -> None:
    """
    Draw a side-legend: cluster colour boxes + temperature bar.
    """
    # ---- LEFT SIDE: colour swatches ----
    swatch_ax = fig.add_axes([0.015, 0.10, 0.12, 0.80])
    swatch_ax.set_facecolor("#050713")
    swatch_ax.set_axis_off()

    unique_clusters = sorted(set(cluster_ids.values()))
    max_show = min(18, len(unique_clusters))  # avoid infinite legend

    y = 0.95
    dy = 0.05

    for cid in unique_clusters[:max_show]:
        rgba_raw = color_map.get(cid, (0.7, 0.7, 0.7, 1.0))
        rgba = tuple(float(x) for x in rgba_raw)
        swatch_ax.add_patch(
            plt.Rectangle(
                (0.05, y - 0.025),
                0.12,
                0.035,
                facecolor=rgba,
                edgecolor="white",
                linewidth=0.4,
            )
        )
        swatch_ax.text(
            0.20,
            y - 0.007,
            cid,
            fontsize=7,
            color="white",
            va="center",
            ha="left",
        )
        y -= dy

    swatch_ax.text(
        0.05,
        0.97,
        "Clusters / Compounds",
        fontsize=9,
        color="white",
        ha="left",
        va="center",
    )

    # ---- RIGHT SIDE: temperature scale ----
    temp_ax = fig.add_axes([0.015, 0.02, 0.12, 0.05])
    temp_ax.set_axis_off()

    if len(compound_temps) > 0:
        # gradient from cold to hot
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        temp_ax.imshow(
            gradient,
            cmap=plt.cm.plasma,
            aspect="auto",
            extent=[0, 1, 0, 0.4],
        )
        temp_ax.text(
            0,
            -0.1,
            "Cold",
            fontsize=7,
            color="white",
            ha="left",
            va="center",
        )
        temp_ax.text(
            1,
            -0.1,
            "Hot",
            fontsize=7,
            color="white",
            ha="right",
            va="center",
        )
        temp_ax.text(
            0.5,
            0.55,
            "Compound Temperature",
            fontsize=8,
            color="white",
            ha="center",
            va="center",
        )


def _draw_snapshot(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    cluster_ids: Dict[str, str],
    color_map: Dict[str, Any],
    node_sizes: Dict[str, float],
    compound_map: Dict[str, str],
    compound_temps: Dict[str, float],
    out_path: str,
    emit=DEFAULT_EMIT,
) -> None:
    """
    Draw a single snapshot for a given subgraph SG.
    """
    plt.figure(figsize=(16, 12), facecolor="#050713")
    ax = plt.gca()
    ax.set_facecolor("#050713")
    ax.set_axis_off()

    nodes = list(G.nodes())
    if len(nodes) == 0:
        plt.savefig(out_path, dpi=260)
        plt.close()
        _log(f"[graphs] Wrote empty graph {out_path}", emit)
        return

    # Build a safe subset of positions for these nodes
    sub_pos = {n: pos[n] for n in nodes if n in pos}
    if len(sub_pos) == 0:
        plt.savefig(out_path, dpi=260)
        plt.close()
        _log(f"[graphs] Wrote graph with no positions {out_path}", emit)
        return

    # camera framing
    x_min, x_max, y_min, y_max = _camera_frame(sub_pos)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # optional hulls for compounds (behind everything)
    _draw_compound_hulls(ax, sub_pos, compound_map, nodes)

    # Colours and sizes
    node_color = [
        tuple(color_map.get(cluster_ids.get(str(n), "default"), (0.2, 0.9, 0.7, 1.0)))
        for n in nodes
    ]
    node_size = [node_sizes.get(str(n), 40.0) for n in nodes]

    # Edges
    weights = [d.get("weight", 1.0) for _, _, d in G.edges(data=True)]
    if len(weights) > 0:
        w_arr = np.array([_safe_float(w, 1.0) for w in weights], dtype=float)
        if w_arr.size == 0:
            widths_list: List[float] = [0.4] * len(weights)
        else:
            w_min = float(np.nanmin(w_arr))
            w_max = float(np.nanmax(w_arr))
            if (
                not np.isfinite(w_min)
                or not np.isfinite(w_max)
                or (w_max - w_min) < 1e-9
            ):
                widths_list = [0.4] * len(weights)
            else:
                widths_arr = 0.2 + 1.8 * (w_arr - w_min) / (w_max - w_min + 1e-9)
                widths_list = [float(x) for x in widths_arr]
    else:
        widths_list = []

    # Edge alpha tuned to density
    edge_alpha = 0.08 if len(nodes) > 600 else 0.16

    nx.draw_networkx_edges(
        G,
        sub_pos,
        width=widths_list if len(widths_list) > 0 else 0.4,
        edge_color="#5b4c8a",
        alpha=edge_alpha,
    )

    # Halo / glow: draw slightly bigger, transparent nodes underneath
    halo_sizes = [s * 1.6 for s in node_size]
    nx.draw_networkx_nodes(
        G,
        sub_pos,
        nodelist=nodes,
        node_color=node_color,
        node_size=halo_sizes,
        linewidths=0.0,
        alpha=0.13,
    )

    # Main nodes
    nx.draw_networkx_nodes(
        G,
        sub_pos,
        nodelist=nodes,
        node_color=node_color,
        node_size=node_size,
        linewidths=0.4,
        edgecolors="#050713",
    )

    # Labels (two-tier)
    primary_nodes, secondary_nodes = _pick_label_nodes(node_sizes, len(nodes))
    primary_nodes = [n for n in primary_nodes if n in G.nodes()]
    secondary_nodes = [n for n in secondary_nodes if n in G.nodes()]

    # primary labels
    if len(primary_nodes) > 0:
        primary_labels = {n: str(n) for n in primary_nodes}
        texts = nx.draw_networkx_labels(
            G,
            sub_pos,
            labels=primary_labels,
            font_size=10,
            font_weight="bold",
            font_color="#ffffff",
            alpha=0.98,
        )
        if _HAS_PATH_EFFECTS:
            for txt in texts.values():
                txt.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=3.0, foreground="#050713"),
                        patheffects.Normal(),
                    ]
                )

    # secondary labels
    if len(secondary_nodes) > 0:
        secondary_labels = {n: str(n) for n in secondary_nodes}
        texts2 = nx.draw_networkx_labels(
            G,
            sub_pos,
            labels=secondary_labels,
            font_size=8,
            font_weight="regular",
            font_color="#e5e7eb",
            alpha=0.92,
        )
        if _HAS_PATH_EFFECTS:
            for txt in texts2.values():
                txt.set_path_effects(
                    [
                        patheffects.Stroke(linewidth=2.3, foreground="#050713"),
                        patheffects.Normal(),
                    ]
                )

    # Legend
    _draw_legend(plt.gcf(), color_map, compound_temps, cluster_ids)

    plt.tight_layout(pad=0.1)
    plt.savefig(out_path, dpi=260)
    plt.close()

    _log(f"[graphs] Wrote {out_path}", emit)
    emit("artifact", {"kind": "graph_snapshot", "path": out_path})


# -----------------------------------------------------------------------------#
# Public entry point
# -----------------------------------------------------------------------------#

def generate_graph_snapshots(results_dir: str, emit=DEFAULT_EMIT) -> None:
    """
    Main entry point called by the orchestrator.

    Reads:
      - hilbert_elements.csv
      - edges.csv
      - element_roots.csv              (optional)
      - informational_compounds.json   (optional)

    Writes graph_*.png snapshots into results_dir.
    """
    elements_csv = os.path.join(results_dir, "hilbert_elements.csv")
    edges_csv = os.path.join(results_dir, "edges.csv")

    df = _load_elements(elements_csv, emit)
    edges = _load_edges(edges_csv, emit)

    if df.empty or edges.empty:
        _log("[graphs] Skipping snapshots — elements or edges empty.", emit)
        return

    # index by element for fast lookup
    df = df.set_index("element")

    # build full graph using only elements that appear in edges
    # (this automatically removes isolated elements).
    used_elements = set(edges["source"]).union(set(edges["target"]))

    G = nx.Graph()
    for el in used_elements:
        if el in df.index:
            G.add_node(str(el))

    for _, row in edges.iterrows():
        s = str(row["source"])
        t = str(row["target"])
        if s in G and t in G:
            G.add_edge(s, t, weight=_safe_float(row.get("weight", 1.0)))

    if G.number_of_nodes() == 0:
        _log("[graphs] Graph has no nodes after filtering; skipping snapshots.", emit)
        return

    # Restrict to "large enough" connected components
    comp_sizes: Dict[str, int] = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for n in comp:
            comp_sizes[str(n)] = size

    large_nodes = [n for n in G.nodes() if comp_sizes.get(str(n), 0) >= MIN_COMPONENT_SIZE]
    if len(large_nodes) == 0:
        _log(
            f"[graphs] No components with size >= {MIN_COMPONENT_SIZE}; "
            "using full graph instead.",
            emit,
        )
    else:
        G = G.subgraph(large_nodes).copy()

    if G.number_of_nodes() == 0:
        _log("[graphs] Graph empty after component filtering; skipping snapshots.", emit)
        return

    # cluster ids, compounds and colours
    root_map = _load_root_clusters(results_dir, emit)
    compound_map = _load_compound_membership(results_dir, emit)
    compound_temps = _load_compound_temperature(results_dir, emit)

    cluster_ids = _build_cluster_ids(G, root_map, compound_map)
    color_map = _make_color_map(cluster_ids, compound_temps)

    # node sizes and layout once
    node_sizes = _compute_node_sizes(G, df)
    pos = _compute_layout(G)

    # choose node ordering: by size (importance), restricted to current G
    ordered_nodes = sorted(
        G.nodes(), key=lambda n: node_sizes.get(str(n), 0.0), reverse=True
    )
    total = len(ordered_nodes)

    # Progressive checkpoints similar to UI slider; percentages + absolute caps
    candidate_counts = [5, 10, 15, 20, 25, 50, 75, 100, 150, 200, 250, 300, 500, total]
    checkpoints = sorted(set(min(c, total) for c in candidate_counts if c > 0))

    for N in checkpoints:
        sub_nodes = ordered_nodes[:N]
        SG = G.subgraph(sub_nodes).copy()
        out_name = f"graph_{N if N != total else 'full'}.png"
        out_path = os.path.join(results_dir, out_name)
        _draw_snapshot(
            SG,
            pos,
            cluster_ids,
            color_map,
            node_sizes,
            compound_map,
            compound_temps,
            out_path,
            emit,
        )

    _log("[graphs] Completed all graph snapshots.", emit)

def generate_informative_graph_snapshot(
    out_dir,
    elements_df,
    edges_df,
    molecule_df,
    enriched_spans,
    emit=None
):
    """
    Produce a significantly improved graph layout with:
    - edge thresholding
    - edge fading
    - node scaling
    - cluster simplification
    - spectral + spring layout hybrid
    - enriched metadata
    """

    os.makedirs(out_dir, exist_ok=True)

    # Filter edges by weight
    if "weight" in edges_df.columns:
        edges_df = edges_df[edges_df["weight"] > 0.15]

    G = nx.Graph()

    # Add nodes
    for _, row in elements_df.iterrows():
        el = str(row["element"])
        ent = row.get("mean_entropy", 0.5)
        coh = row.get("mean_coherence", 0.5)
        label = enriched_spans.get(el, {}).get("labels", [])

        G.add_node(el, entropy=float(ent), coherence=float(coh), labels=label)

    # Add edges
    for _, row in edges_df.iterrows():
        G.add_edge(str(row["source"]), str(row["target"]), weight=row.get("weight", 1.0))

    # Layout
    try:
        spectral = nx.spectral_layout(G, dim=2)
        spring = nx.spring_layout(G, pos=spectral, k=0.18, iterations=200)
        pos = spring
    except Exception:
        pos = nx.spring_layout(G)

    # Node sizes: coherence
    size = 300 + 1500 * np.array([G.nodes[n]["coherence"] for n in G.nodes])

    # Node colors: entropy
    entropy_vals = np.array([G.nodes[n]["entropy"] for n in G.nodes])

    plt.figure(figsize=(18, 12))
    nx.draw_networkx_edges(
        G,
        pos,
        width=[0.4 * G[u][v]["weight"] for u, v in G.edges],
        alpha=0.25,
        edge_color="#999999"
    )
    c = plt.cm.coolwarm(entropy_vals)
    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=size,
        node_color=c,
        linewidths=0.2,
        edgecolors="black",
        alpha=0.9
    )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "informative_graph.png"), dpi=300)
    plt.close()

    if emit:
        emit("log", {"stage": "graphs", "message": "Informative graph written."})
