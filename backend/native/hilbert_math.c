// =============================================================================
// hilbert_math.c - Core numerical routines for Hilbert Information Chemistry
// =============================================================================
//
// This module implements the "physics" style metrics that your thesis uses:
//
//  - Spectral entropy H(v) over an eigenvalue / singular-value spectrum
//  - Coherence C(A) over a local field of vectors
//  - Information stability S(H, C) as a combined stability field
//  - Temporal decay for timeline-aware metrics
//
// All routines are designed to be:
//   - Numerically stable (avoid log(0), division by 0)
//   - Thread agnostic (no static internal state)
//   - Simple C API, driven from Python/pybind.
//
// =============================================================================

#include <math.h>
#include <stddef.h>

// Small epsilon for numerical safety
#ifndef HILBERT_EPS
#define HILBERT_EPS 1e-12
#endif

// -----------------------------------------------------------------------------
// Helper: clamp to [0,1]
// -----------------------------------------------------------------------------
static double clamp01(double x) {
    if (x < 0.0) return 0.0;
    if (x > 1.0) return 1.0;
    return x;
}

// -----------------------------------------------------------------------------
// Spectral entropy H - given a real-valued vector v (length n)
//
// Interpretation (thesis-aligned):
//   - v is a spectral energy vector, typically eigenvalues or singular values.
//   - We convert v to probabilities p_i >= 0 with sum p_i = 1.
//   - Then compute Shannon entropy in bits, normalized to [0,1] by log2(n):
//
//       H = - Σ_i p_i log2(p_i) / log2(n)
//
//   - If the spectrum is "flat" -> H ~ 1 (high uncertainty).
//   - If the spectrum is highly concentrated -> H ~ 0 (low uncertainty).
// -----------------------------------------------------------------------------
double hilbert_spectral_entropy(const double *v, int n) {
    if (!v || n <= 0) return 0.0;

    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        double x = v[i];
        if (x < 0.0) x = -x;  // fallback: use magnitude if signs appear
        sum += x;
    }
    if (sum <= HILBERT_EPS) {
        // Degenerate spectrum - treat as perfectly "ordered"
        return 0.0;
    }

    double H = 0.0;
    for (int i = 0; i < n; ++i) {
        double p = v[i];
        if (p < 0.0) p = -p;
        p /= sum;
        if (p > HILBERT_EPS) {
            H -= p * (log(p) / log(2.0));  // log2(p)
        }
    }

    // Normalize to [0,1] using maximum entropy log2(n)
    double H_max = (n > 1) ? (log((double)n) / log(2.0)) : 1.0;
    if (H_max <= HILBERT_EPS) return 0.0;

    double H_norm = H / H_max;
    return clamp01(H_norm);
}

// -----------------------------------------------------------------------------
// Coherence C - average pairwise cosine similarity of row-vectors in A
//
// A is a row-major matrix of shape (rows, cols):
//   - Each row is a local field vector (e.g., element embedding, compound field).
//   - We compute pairwise cosine similarity over all distinct pairs (i < j),
//     and return the mean of those similarities.
//
//   C ≈ average_{i<j}  (a_i · a_j) / (||a_i|| * ||a_j||)
//
// Return value is in [-1, 1], but in practice your pipeline usually lives
// in [0,1], and downstream code will clamp to [0,1] where needed.
// -----------------------------------------------------------------------------
double hilbert_coherence_score(const double *A, int rows, int cols) {
    if (!A || rows <= 1 || cols <= 0) return 0.0;

    double sum_sim = 0.0;
    long long count = 0;

    for (int i = 0; i < rows; ++i) {
        const double *vi = A + (size_t)i * (size_t)cols;

        // Precompute norm of row i
        double norm_i_sq = 0.0;
        for (int k = 0; k < cols; ++k) {
            double x = vi[k];
            norm_i_sq += x * x;
        }
        if (norm_i_sq <= HILBERT_EPS) continue;
        double norm_i = sqrt(norm_i_sq);

        for (int j = i + 1; j < rows; ++j) {
            const double *vj = A + (size_t)j * (size_t)cols;

            double dot = 0.0;
            double norm_j_sq = 0.0;
            for (int k = 0; k < cols; ++k) {
                double y = vj[k];
                dot += vi[k] * y;
                norm_j_sq += y * y;
            }
            if (norm_j_sq <= HILBERT_EPS) continue;
            double norm_j = sqrt(norm_j_sq);

            double denom = norm_i * norm_j;
            if (denom <= HILBERT_EPS) continue;

            double sim = dot / denom;
            sum_sim += sim;
            count += 1;
        }
    }

    if (count == 0) return 0.0;
    double C = sum_sim / (double)count;

    // Optional clamping to [0,1] since the rest of the pipeline expects
    // a "stability-like" coherence.
    if (C < 0.0) C = 0.0;
    if (C > 1.0) C = 1.0;
    return C;
}

// -----------------------------------------------------------------------------
// Information stability S(H, C)
//
// Thesis-aligned intuition:
//   - H in [0,1] is normalized entropy over the spectral field.
//   - C in [0,1] is coherence over the local informational field.
//   - Stability is high when entropy is low (ordered) and coherence is high.
//
// A simple, interpretable combination:
//
//   S = 0.5 * (C + (1 - H))
//
// which lives in [0,1] and can be thought of as a Hilbert "stability field".
// -----------------------------------------------------------------------------
double hilbert_information_stability(double entropy, double coherence) {
    double H = clamp01(entropy);
    double C = clamp01(coherence);
    double S = 0.5 * (C + (1.0 - H));
    return clamp01(S);
}

// -----------------------------------------------------------------------------
// Temporal decay - single-pole exponential smoothing
//
//   prev_val: previous value of the metric
//   next_val: newly observed value
//   alpha   : smoothing factor in [0,1]
//
//   result = (1 - alpha) * prev_val + alpha * next_val
//
// This is used by timeline-aware code in Python to maintain running
// informational fields over time slices.
// -----------------------------------------------------------------------------
double hilbert_temporal_decay(double prev_val, double next_val, double alpha) {
    if (alpha < 0.0) alpha = 0.0;
    if (alpha > 1.0) alpha = 1.0;
    return (1.0 - alpha) * prev_val + alpha * next_val;
}
