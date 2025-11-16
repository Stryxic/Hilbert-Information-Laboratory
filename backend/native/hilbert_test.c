// ============================================================================
// hilbert_test.c — Comprehensive diagnostic suite for Hilbert Information Chemistry
// ============================================================================
//
// Build (Windows):
//   cl /TC /O2 /EHsc /Fe:hilbert_test.exe hilbert_test.c hilbert_native.lib /link /SUBSYSTEM:CONSOLE
//
// Run:
//   hilbert_test.exe
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "hilbert_native.h"

#define N 5
#define D 3

// Helper to print float arrays
static void print_array(const char* name, const float* a, int n) {
    printf("%s: ", name);
    for (int i = 0; i < n; i++) printf("%.3f ", a[i]);
    printf("\n");
}

int main(void) {
    printf("=== Hilbert Native Backend Health Check ===\n");
    hilbert_init();
    hilbert_print_banner();

    // ---------------------------------------------------------------------
    // 1. Memory allocation sanity
    // ---------------------------------------------------------------------
    printf("\n[1] Memory subsystem\n");
    float* buf = (float*)hilbert_alloc(10 * sizeof(float));
    if (buf) printf("Allocated 10 floats OK.\n");
    hilbert_free(buf);
    printf("Freed memory OK.\n");

    // ---------------------------------------------------------------------
    // 2. Spectral layer
    // ---------------------------------------------------------------------
    printf("\n[2] Spectral layer tests\n");
    float X[N * D] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        2, 3, 4,
        0, 1, 2
    };
    float Y[N * D];
    lsa_embed(X, N, D, Y);
    printf("LSA embed complete.\n");

    float sim[N * N];
    cosine_similarity_matrix(Y, N, D, sim);
    printf("Cosine similarity matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%6.3f ", sim[i * N + j]);
        printf("\n");
    }

    int idx[N * 2];
    float wgt[N * 2];
    topk_neighbors(Y, N, D, 2, idx, wgt);
    printf("Top-2 neighbors for each row:\n");
    for (int i = 0; i < N; i++) {
        printf("Row %d: (%d, %.3f), (%d, %.3f)\n",
               i, idx[i * 2], wgt[i * 2], idx[i * 2 + 1], wgt[i * 2 + 1]);
    }

    printf("Entropy(Y[0])=%.3f\n", spectral_entropy(&Y[0], D));
    printf("Coherence(Y[0],Y[1])=%.3f\n", spectral_coherence(&Y[0], &Y[D], D));

    // Edge cases
    printf("Entropy(NULL)=%.3f\n", spectral_entropy(NULL, D));
    printf("Coherence(NULL,NULL)=%.3f\n", spectral_coherence(NULL, NULL, D));

    // ---------------------------------------------------------------------
    // 3. Phrase/text feature extraction
    // ---------------------------------------------------------------------
    printf("\n[3] Phrase/Text layer\n");
    uint32_t seq[5] = {1, 2, 3, 4, 5};
    float M[25] = {0};
    phrase_cooccurrence(seq, 5, 2, M);
    printf("Co-occurrence symmetry check: M[0,1]=%.1f, M[1,0]=%.1f\n", M[1], M[5]);
    printf("Lexical density('hello world!')=%.3f\n", lexical_density("hello world!"));
    printf("Lexical density(empty)=%.3f\n", lexical_density(""));
    printf("Semantic overlap(Y0,Y1)=%.3f\n", semantic_overlap(Y, Y + D, D));

    // ---------------------------------------------------------------------
    // 4. Graph layer
    // ---------------------------------------------------------------------
    printf("\n[4] Graph construction & metrics\n");
    uint32_t nodes[4] = {0, 1, 2, 3};
    float wts[4] = {0.9f, 0.3f, 0.7f, 0.5f};
    int m = 0;
    edge_t* edges = build_cooccurrence_graph(nodes, 4, wts, &m);
    printf("Edges built: %d\n", m);

    normalize_edge_weights(edges, m);
    printf("Graph stability=%.3f\n", compute_graph_stability(edges, m));
    printf("Graph density=%.3f\n", compute_graph_density(edges, m, 4));

    float deg[4];
    compute_degree_distribution(edges, m, 4, deg);
    print_array("Degrees", deg, 4);

    int labels[4];
    int comps = graph_connected_components(edges, m, 4, labels);
    printf("Connected components=%d\n", comps);

    // Export graph
    graph_export_edges_csv("test_edges.csv", edges, m);
    graph_export_nodes_csv("test_nodes.csv", 4, NULL, labels);
    graph_export_json("test_graph.json", 4, edges, m, NULL, labels, NULL);
    printf("Graph exports written.\n");

    hilbert_free(edges);

    // ---------------------------------------------------------------------
    // 5. Persistence / signal stability
    // ---------------------------------------------------------------------
    printf("\n[5] Temporal persistence\n");
    float entropy_series[5] = {0.1f, 0.2f, 0.5f, 0.4f, 0.2f};
    float coherence_series[5] = {0.9f, 0.8f, 0.7f, 0.9f, 0.95f};
    float persistence[5];
    compute_signal_stability(entropy_series, coherence_series, 5, persistence);
    print_array("Persistence", persistence, 5);

    float S = compute_stability_score(0.3f, 0.8f);
    printf("Stability score(0.3,0.8)=%.3f\n", S);
    float avg = moving_average(coherence_series, 5, 3);
    printf("Moving average(3)=%.3f\n", avg);

    // ---------------------------------------------------------------------
    // 6. Prime helix / numeric transforms
    // ---------------------------------------------------------------------
    printf("\n[6] Prime helix transforms\n");
    for (int i = 0; i < 5; i++) {
        double x = (double)i;
        printf("helix(%.1f)=%.6f\n", x, prime_helix_transform(x));
    }
    double st = prime_stability(1.0, 2.0);
    printf("prime_stability(1,2)=%.6f\n", st);
    double src[4] = {0.1, 0.3, 0.6, 0.9}, dst[4];
    prime_series_transform(src, dst, 4);
    printf("series_transform -> %.3f %.3f %.3f %.3f\n", dst[0], dst[1], dst[2], dst[3]);
    double field[5] = {0, 1, 0, 1, 0};
    prime_diffusion(field, 5, 0.2, 3);
    printf("diffused field: ");
    for (int i = 0; i < 5; i++) printf("%.3f ", field[i]);
    printf("\n");

    // ---------------------------------------------------------------------
    // 7. Prime store (in-memory key-value)
    // ---------------------------------------------------------------------
    printf("\n[7] Prime store in-memory (key-value)\n");
    prime_store_clear_kv();
    prime_store_put_kv("alpha", 3.14);
    prime_store_put_kv("beta", 2.718);

    int found = 0;
    double v = prime_store_get_kv("alpha", &found);
    printf("get(alpha)=%.3f found=%d\n", v, found);

    v = prime_store_get_kv("missing", &found);
    printf("get(missing)=%.3f found=%d\n", v, found);

    prime_store_clear_kv();
    printf("KV store cleared OK.\n");

    // ---------------------------------------------------------------------
    // 8. On-disk persistence test
    // ---------------------------------------------------------------------
printf("\n[8] PrimeStore file I/O test\n");
const char* path = "test_data.pstore";
PrimeStore* ps = prime_store_open(path, 1);
if (!ps) {
    printf("Failed to open prime store file.\n");
} else {
    printf("Opened prime store file OK.\n");
    // Append a few records
    double vec1[3] = {1.1, 2.2, 3.3};
    double vec2[3] = {4.4, 5.5, 6.6};
    prime_store_append(ps, vec1, 3, 100);
    prime_store_append(ps, vec2, 3, 200);
    printf("Appended 2 records. Count=%llu\n",
           (unsigned long long)prime_store_count(ps));

    // Read back record 1
    uint32_t dim = 3;
    double buf2[3];
    uint64_t ts = 0;
    prime_store_get(ps, 1, buf2, &dim, &ts);
    printf("Read record 1 ts=%llu dim=%u -> %.2f %.2f %.2f\n",
           (unsigned long long)ts, dim, buf2[0], buf2[1], buf2[2]);

    // Export CSV
    prime_store_export_csv(ps, "test_data.csv", 10);
    printf("Exported CSV: test_data.csv\n");

    prime_store_close(ps);
}

// Compact and reopen
ps = prime_store_open(path, 1);
if (ps) {
    prime_store_compact(ps);
    prime_store_close(ps);
    printf("Compacted store successfully.\n");
}

    // ---------------------------------------------------------------------
    // 9. Diagnostics & version
    // ---------------------------------------------------------------------
    printf("\n[9] Diagnostics\n");
    hilbert_print_stats();
    printf("Hilbert version: %.5f\n", hilbert_version());
    printf("Status message (OK)=%s\n", hilbert_status_message(HILBERT_OK));
    printf("Status message (ERR_ALLOC)=%s\n", hilbert_status_message(HILBERT_ERR_ALLOC));

    hilbert_shutdown();
    printf("\n=== End of Hilbert backend test ===\n");
    return 0;
}
