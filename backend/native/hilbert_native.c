// =============================================================================
// hilbert_native.c - Unified C Interface for Hilbert Information Chemistry Lab
// =============================================================================
//
// Implements the API defined in hilbert_native.h and used by hilbert_pybind.c:
//
//  - Versioning and lifecycle
//  - Memory helpers and output directory
//  - Spectral / coherence / stability metrics (delegating to hilbert_math.c)
//  - Simple CSV graph export for edges
//  - Lightweight JSON append/read store (in-memory per "kind")
//  - Simulation wrappers (delegating to hilbert_simulation.c)
//
// The intent is to keep this file as a thin coordination layer. The "science"
// lives in hilbert_math.c and hilbert_simulation.c; Python owns LSA and corpus
// management.
// =============================================================================

#define HILBERT_NATIVE_EXPORTS 1

#include "hilbert_native.h"
#include "hilbert_simulation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// ----------------------------------------------------------------------------
// Forward declarations from hilbert_math.c
// ----------------------------------------------------------------------------
double hilbert_spectral_entropy(const double *v, int n);
double hilbert_coherence_score(const double *A, int rows, int cols);
double hilbert_information_stability(double entropy, double coherence);
double hilbert_temporal_decay(double prev_val, double next_val, double alpha);

// ----------------------------------------------------------------------------
// Global state
// ----------------------------------------------------------------------------

static char g_output_dir[1024] = ".";

// Very small in-memory "store" for JSON records keyed by "kind"
typedef struct JsonStoreNode {
    char *kind;
    char *data;  // newline-separated JSON records
    struct JsonStoreNode *next;
} JsonStoreNode;

static JsonStoreNode *g_store_head = NULL;

// -----------------------------------------------------------------------------
// Versioning and lifecycle
// -----------------------------------------------------------------------------
double hilbert_version(void) {
    // Encode version as major.minor where minor is two digits
    // Example: 0.20 for "0.2"
    return 0.20;
}

int hilbert_init(void) {
    // For now, nothing to initialize beyond clearing state.
    // Could seed RNG here if needed for simulations.
    srand((unsigned int)time(NULL));
    strcpy(g_output_dir, ".");
    return 0;
}

void hilbert_shutdown(void) {
    // Free JSON store
    JsonStoreNode *node = g_store_head;
    while (node) {
        JsonStoreNode *next = node->next;
        free(node->kind);
        free(node->data);
        free(node);
        node = next;
    }
    g_store_head = NULL;
}

// -----------------------------------------------------------------------------
// Memory Management
// -----------------------------------------------------------------------------
void *hilbert_alloc(size_t bytes) {
    if (bytes == 0) return NULL;
    return malloc(bytes);
}

void hilbert_free(void *p) {
    if (p) free(p);
}

void hilbert_set_output_dir(const char *path) {
    if (!path) return;
    size_t len = strlen(path);
    if (len >= sizeof(g_output_dir)) len = sizeof(g_output_dir) - 1;
    memcpy(g_output_dir, path, len);
    g_output_dir[len] = '\0';
}

// -----------------------------------------------------------------------------
// Spectral and Informational Metrics
// -----------------------------------------------------------------------------
double hilbert_spectral_entropy(const double *v, int n);
double hilbert_coherence_score(const double *A, int rows, int cols);
double hilbert_information_stability(double entropy, double coherence);
double hilbert_temporal_decay(double prev_val, double next_val, double alpha);

// We simply re-expose the math functions through the HILBERT_API.
// The linker will bind these to hilbert_math.c.

double hilbert_spectral_entropy(const double *v, int n);
double hilbert_coherence_score(const double *A, int rows, int cols);
double hilbert_information_stability(double entropy, double coherence);
double hilbert_temporal_decay(double prev_val, double next_val, double alpha);

// -----------------------------------------------------------------------------
// Graph Export - write edges as CSV to path
// -----------------------------------------------------------------------------
int hilbert_graph_export_edges(const char *path,
                               const char **src,
                               const char **tgt,
                               const double *w,
                               int n_edges) {
    if (!path || !src || !tgt || !w || n_edges <= 0) return -1;

    FILE *fp = fopen(path, "w");
    if (!fp) return -2;

    // Header row
    fprintf(fp, "source,target,weight\n");

    for (int i = 0; i < n_edges; ++i) {
        const char *s = src[i] ? src[i] : "";
        const char *t = tgt[i] ? tgt[i] : "";
        double weight = w[i];

        // Minimal CSV escaping: replace any embedded quotes with single quote
        // (this is sufficient for our lab output, which uses simple IDs).
        char sbuf[512];
        char tbuf[512];

        size_t si = 0, ti = 0;
        for (const char *p = s; *p && si < sizeof(sbuf) - 1; ++p) {
            sbuf[si++] = (*p == '"') ? '\'' : *p;
        }
        sbuf[si] = '\0';

        for (const char *p = t; *p && ti < sizeof(tbuf) - 1; ++p) {
            tbuf[ti++] = (*p == '"') ? '\'' : *p;
        }
        tbuf[ti] = '\0';

        fprintf(fp, "\"%s\",\"%s\",%.10g\n", sbuf, tbuf, weight);
    }

    fclose(fp);
    return 0;
}

// -----------------------------------------------------------------------------
// JSON Store API (Binary-JSON Record Storage)
// -----------------------------------------------------------------------------
//
// This implementation is intentionally simple and in-memory:
//
//   - hilbert_append_json(kind, json)   appends json + '\n' to an internal
//     buffer keyed by "kind".
//   - hilbert_read_store(kind)         returns a heap-allocated copy of that
//     buffer (caller must free via hilbert_free).
//
// Python typically uses this to accumulate "elements", "compounds", "timeline",
// etc, before writing them to disk.
// -----------------------------------------------------------------------------
static JsonStoreNode *find_or_create_store(const char *kind) {
    if (!kind) return NULL;

    JsonStoreNode *node = g_store_head;
    while (node) {
        if (strcmp(node->kind, kind) == 0) return node;
        node = node->next;
    }

    // Create new node
    JsonStoreNode *new_node = (JsonStoreNode *)malloc(sizeof(JsonStoreNode));
    if (!new_node) return NULL;
    new_node->kind = _strdup(kind);
    new_node->data = _strdup("");
    new_node->next = g_store_head;
    g_store_head = new_node;
    return new_node;
}

int hilbert_append_json(const char *kind, const char *json_record) {
    if (!kind || !json_record) return -1;

    JsonStoreNode *node = find_or_create_store(kind);
    if (!node) return -2;

    size_t old_len = node->data ? strlen(node->data) : 0;
    size_t add_len = strlen(json_record) + 1; // +1 for newline
    size_t new_len = old_len + add_len;

    char *buf = (char *)malloc(new_len + 1);
    if (!buf) return -3;

    if (node->data) {
        memcpy(buf, node->data, old_len);
    }
    memcpy(buf + old_len, json_record, add_len - 1);
    buf[new_len - 1] = '\n';
    buf[new_len] = '\0';

    free(node->data);
    node->data = buf;
    return 0;
}

char *hilbert_read_store(const char *kind) {
    if (!kind) return NULL;

    JsonStoreNode *node = g_store_head;
    while (node) {
        if (strcmp(node->kind, kind) == 0) {
            size_t len = node->data ? strlen(node->data) : 0;
            char *copy = (char *)hilbert_alloc(len + 1);
            if (!copy) return NULL;
            if (len > 0 && node->data) {
                memcpy(copy, node->data, len);
            }
            copy[len] = '\0';
            return copy;
        }
        node = node->next;
    }
    return NULL;
}

// -----------------------------------------------------------------------------
// Diagnostics
// -----------------------------------------------------------------------------
void hilbert_banner(void) {
    fprintf(stderr,
            "Hilbert Information Chemistry Native Core (v%.2f)\n"
            " - Spectral entropy / coherence / stability metrics\n"
            " - JSON store and graph export\n"
            " - Simulation core enabled\n",
            hilbert_version());
}

int hilbert_selftest(void) {
    // Very small self-test to verify metrics are sane.
    double v[4] = {1.0, 1.0, 1.0, 1.0};
    double H = hilbert_spectral_entropy(v, 4);
    if (H < 0.99 || H > 1.01) {
        return -1;
    }

    double A[6] = {
        1.0, 0.0, 0.0,
        1.0, 0.0, 0.0
    };
    double C = hilbert_coherence_score(A, 2, 3);
    if (C < 0.99 || C > 1.01) {
        return -2;
    }

    double S = hilbert_information_stability(H, C);
    if (S <= 0.0 || S > 1.0) {
        return -3;
    }

    double d = hilbert_temporal_decay(0.0, 1.0, 0.5);
    if (d < 0.49 || d > 0.51) {
        return -4;
    }

    return 0;
}

// -----------------------------------------------------------------------------
// Simulation API wrappers
// -----------------------------------------------------------------------------
int hilbert_sim_init(int n_agents) {
    return hilbert_sim_init(n_agents);
}

void hilbert_sim_step(double dt) {
    hilbert_sim_step(dt);
}

void hilbert_sim_inject(double x, double y, double radius, int regime) {
    hilbert_sim_inject(x, y, radius, regime);
}

void hilbert_sim_metrics(double *mean, double *pol, double *entropy) {
    hilbert_sim_metrics(mean, pol, entropy);
}

void hilbert_sim_export_state(const char *path_json) {
    hilbert_sim_export_state(path_json);
}
