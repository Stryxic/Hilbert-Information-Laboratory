"""
test_hilbert_native.py

Quick sanity check for the compiled `hilbert_native` extension.

Run from the same directory where hilbert_native*.pyd lives:

    (WebApp) python test_hilbert_native.py
"""

import math
import os
import sys

import hilbert_native as hn


def header(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def test_core():
    header("[1] Core / init / version")
    v = hn.version()
    print(f"hilbert_native.version() = {v}")
    hn.init()
    print("hn.init() OK")
    hn.stats()
    print("Stats printed.")


def test_lexical_and_spectral():
    header("[2] Lexical & Spectral")

    text = "Hilbert information chemistry lab online"
    ld = hn.lexical_density(text)
    print(f"lexical_density('{text}') = {ld:.4f}")

    vec = [0.1, 0.3, 0.6]
    ent = hn.spectral_entropy(vec)
    print(f"spectral_entropy({vec}) = {ent:.6f}")

    v1 = [1.0, 2.0, 3.0]
    v2 = [2.0, 4.0, 6.0]
    coh = hn.spectral_coherence(v1, v2)
    print(f"spectral_coherence({v1}, {v2}) = {coh:.6f}")

    # simple sanity checks (not strict asserts to avoid surprises)
    print("Expect: entropy in [0,1], coherence close to 1.0 for collinear vectors.")


def test_prime_functions():
    header("[3] Prime helix & stability")

    for x in [0.0, 1.0, 2.5, 5.0]:
        y = hn.prime_helix(x)
        print(f"prime_helix({x}) = {y:.6f}")

    ps = hn.prime_stability(1.0, 2.0)
    print(f"prime_stability(1.0, 2.0) = {ps:.6f}")
    print("Expect: values in [0,1], smoothly decaying with radius.")


def test_kv_store():
    header("[4] In-memory KV store")

    hn.kv_clear()
    print("kv_clear() OK.")

    ok1 = hn.kv_put("alpha", 3.14159)
    ok2 = hn.kv_put("beta", 2.71828)
    print(f"kv_put('alpha', 3.14159) -> {ok1}")
    print(f"kv_put('beta', 2.71828) -> {ok2}")

    v_alpha = hn.kv_get("alpha")
    v_beta = hn.kv_get("beta")
    v_missing = hn.kv_get("missing")

    print(f"kv_get('alpha')  = {v_alpha!r}")
    print(f"kv_get('beta')   = {v_beta!r}")
    print(f"kv_get('missing')= {v_missing!r} (expected: None)")

    hn.kv_clear()
    print("kv_clear() again OK.")


def test_graph_export():
    header("[5] Graph export (CSV)")

    edges = [
        (0, 1, 0.5),
        (1, 2, 0.8),
        (2, 0, 0.3),
    ]
    out_path = "test_edges_py.csv"

    ok = hn.graph_export_edges(out_path, edges)
    print(f"graph_export_edges('{out_path}', {edges}) -> {ok}")

    if ok and os.path.exists(out_path):
        with open(out_path, "r", encoding="utf-8") as f:
            head = "".join(f.readlines()[:4])
        print("--- CSV preview ---")
        print(head.strip())
        print("-------------------")
    else:
        print("Export failed or file not found.")


def main():
    try:
        test_core()
        test_lexical_and_spectral()
        test_prime_functions()
        test_kv_store()
        test_graph_export()
    finally:
        # Always try to shut down cleanly.
        header("[X] Shutdown")
        try:
            hn.shutdown()
            print("hn.shutdown() OK.")
        except Exception as e:
            print(f"Shutdown raised: {e!r}")

    print("\nAll tests executed. If no exceptions and outputs look sane, binding is healthy.")


if __name__ == "__main__":
    main()
