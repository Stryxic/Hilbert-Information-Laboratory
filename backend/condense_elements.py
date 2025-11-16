# =============================================================================
# condense_elements.py — Advanced Element Condensation Layer (Final Stable)
# =============================================================================
#
# Fully compatible with hilbert_orchestrator.py and FastAPI backend.
# Safe for cluster sizes 1..N, robust embeddings, guaranteed return structure.
#
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# External emit callback fallback
DEFAULT_EMIT = lambda *_: None


# -----------------------------------------------------------------------------
# Logging helper
# -----------------------------------------------------------------------------
def _log(msg: str, emit=DEFAULT_EMIT):
    print(msg)
    try:
        emit("log", {"message": msg})
    except Exception:
        pass


# -----------------------------------------------------------------------------
# Embedding / entropy parsing
# -----------------------------------------------------------------------------
def _parse_embeddings(df: pd.DataFrame) -> np.ndarray:
    if "embedding" not in df.columns:
        raise ValueError("hilbert_elements.csv must include embedding column")

    first = df.iloc[0]["embedding"]

    if isinstance(first, str):
        vecs = df["embedding"].apply(lambda s: np.array(json.loads(s)))
    elif isinstance(first, list):
        vecs = df["embedding"].apply(lambda v: np.array(v))
    elif isinstance(first, np.ndarray):
        vecs = df["embedding"].apply(lambda v: v)
    else:
        raise ValueError(f"Unsupported embedding format: {type(first)}")

    X = np.vstack(vecs.to_numpy())
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms


def _ensure_entropy(df: pd.DataFrame) -> np.ndarray:
    if "entropy" in df.columns:
        series = df["entropy"]
    elif "mean_entropy" in df.columns:
        series = df["mean_entropy"]
    else:
        return np.zeros(len(df), dtype=float)

    arr = pd.to_numeric(series, errors="coerce").fillna(series.median()).to_numpy()
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr.astype(float)


# -----------------------------------------------------------------------------
# Adaptive threshold
# -----------------------------------------------------------------------------
def adaptive_sim_threshold(entropy, base=0.90, min_t=0.80, max_t=0.97):
    if entropy <= 0.01:
        return max_t
    if entropy >= 5.0:
        return min_t
    scale = np.exp(-entropy / 3.0)
    return min_t + (max_t - min_t) * scale


# -----------------------------------------------------------------------------
# Coarse clustering
# -----------------------------------------------------------------------------
def coarse_clusters(X, entropies, emit=DEFAULT_EMIT):
    n = len(X)
    S = cosine_similarity(X)

    visited = np.zeros(n, dtype=bool)
    clusters = []

    for i in range(n):
        if visited[i]:
            continue

        thr = adaptive_sim_threshold(entropies[i])
        group = [i]
        visited[i] = True

        for j in range(i + 1, n):
            if not visited[j] and S[i, j] >= thr:
                group.append(j)
                visited[j] = True

        clusters.append(group)

    _log(f"[condense] coarse clustering produced {len(clusters)} groups", emit)
    return clusters, S


# -----------------------------------------------------------------------------
# Fine refinement
# -----------------------------------------------------------------------------
def refine_cluster(cluster, X, S):
    if len(cluster) <= 2:
        return [cluster]

    local = X[cluster]
    local_sim = cosine_similarity(local)

    mask = local_sim < 0.9999
    vals = local_sim[mask]

    if vals.size == 0:
        return [cluster]

    med = float(np.median(vals))
    thr = max(0.85, min(0.98, med))

    visited = np.zeros(len(cluster), dtype=bool)
    subs = []

    for i in range(len(cluster)):
        if visited[i]:
            continue
        sub = [cluster[i]]
        visited[i] = True
        for j in range(i + 1, len(cluster)):
            if not visited[j] and local_sim[i, j] >= thr:
                sub.append(cluster[j])
                visited[j] = True
        subs.append(sub)

    return subs


# -----------------------------------------------------------------------------
# Root selection
# -----------------------------------------------------------------------------
def pick_root(df_sub: pd.DataFrame):
    df_sub = df_sub.copy()

    sort_cols = []
    ascending = []

    if "mean_coherence" in df_sub.columns:
        sort_cols.append("mean_coherence"); ascending.append(False)
    if "entropy" in df_sub.columns:
        sort_cols.append("entropy"); ascending.append(True)
    if "mean_entropy" in df_sub.columns and "entropy" not in df_sub.columns:
        sort_cols.append("mean_entropy"); ascending.append(True)
    if "tf" in df_sub.columns:
        sort_cols.append("tf"); ascending.append(False)
    if "doc_freq" in df_sub.columns:
        sort_cols.append("doc_freq"); ascending.append(False)

    if not sort_cols:
        return df_sub.iloc[0]

    return df_sub.sort_values(by=sort_cols, ascending=ascending).iloc[0]


# -----------------------------------------------------------------------------
# Main condensation procedure
# -----------------------------------------------------------------------------
def condense_elements(df: pd.DataFrame, emit=DEFAULT_EMIT):
    if df.empty:
        _log("[condense] empty element table", emit)
        return pd.DataFrame(), {}, []

    df = df.copy()

    if "entropy" not in df.columns and "mean_entropy" in df.columns:
        df["entropy"] = df["mean_entropy"]

    ent = _ensure_entropy(df)
    X = _parse_embeddings(df)

    # PCA if huge dimensional vectors
    if X.shape[1] > 256:
        _log("[condense] PCA → 128 dims", emit)
        n_components = min(128, X.shape[1], X.shape[0])
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    coarse, S = coarse_clusters(X, ent, emit=emit)

    clusters = []
    for c in coarse:
        clusters.extend(refine_cluster(c, X, S))

    _log(f"[condense] total refined clusters: {len(clusters)}", emit)

    roots = []
    cluster_map = {}
    cluster_metrics = []

    for cluster in clusters:
        sub = df.iloc[cluster]
        root = pick_root(sub)
        root_el = root["element"]
        roots.append(root)

        for idx in cluster:
            cluster_map[df.iloc[idx]["element"]] = root_el

        cs = cosine_similarity(X[cluster])
        mask = cs < 0.999999
        vals = cs[mask]

        if vals.size == 0:
            avg_sim = 1.0
        else:
            avg_sim = float(np.mean(vals))

        if "entropy" in sub.columns:
            entropy_span = float(sub["entropy"].max() - sub["entropy"].min())
        elif "mean_entropy" in sub.columns:
            entropy_span = float(sub["mean_entropy"].max() - sub["mean_entropy"].min())
        else:
            entropy_span = 0.0

        cluster_metrics.append({
            "root": root_el,
            "size": len(cluster),
            "avg_intra_similarity": avg_sim,
            "entropy_spread": entropy_span,
            "degenerate": bool(vals.size == 0),
        })

    roots_df = pd.DataFrame(roots).reset_index(drop=True)
    roots_df["cluster_size"] = [len(c) for c in clusters]

    return roots_df, cluster_map, cluster_metrics


# -----------------------------------------------------------------------------
# Entry point used by orchestrator
# -----------------------------------------------------------------------------
def run_condensation(results_dir: str, emit=DEFAULT_EMIT):
    p = Path(results_dir)
    elements_csv = p / "hilbert_elements.csv"

    if not elements_csv.exists():
        _log("[condense] hilbert_elements.csv missing — skip condensation", emit)
        return {
            "roots_csv": None,
            "clusters_json": None,
            "metrics_json": None,
            "n_total": 0,
            "n_roots": 0,
        }

    df = pd.read_csv(elements_csv)
    if df.empty:
        _log("[condense] hilbert_elements.csv is empty — skip", emit)
        return {
            "roots_csv": None,
            "clusters_json": None,
            "metrics_json": None,
            "n_total": 0,
            "n_roots": 0,
        }

    _log(f"[condense] Loaded {len(df)} elements. Starting condensation...", emit)

    roots_df, cluster_map, metrics = condense_elements(df, emit=emit)

    roots_csv = p / "element_roots.csv"
    clusters_json = p / "element_clusters.json"
    metrics_json = p / "element_cluster_metrics.json"

    roots_df.to_csv(roots_csv, index=False)
    json.dump(cluster_map, open(clusters_json, "w"), indent=2)
    json.dump(metrics, open(metrics_json, "w"), indent=2)

    _log(f"[condense] {len(df)} → {len(roots_df)} root elements written.", emit)

    return {
        "roots_csv": str(roots_csv),
        "clusters_json": str(clusters_json),
        "metrics_json": str(metrics_json),
        "n_total": len(df),
        "n_roots": len(roots_df),
    }


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "./results/hilbert_run"
    run_condensation(folder)
