/*
===============================================================================
 hilbert_simulation.c
-------------------------------------------------------------------------------
 Minimal self-contained simulation layer for the Hilbert Information Chemistry Lab.

 This file centralizes the simulation-related functions used by:
   - hilbert_native.c   (via hilbert_simulation.h)
   - hilbert_pybind.c   (Python bindings)
   - FastAPI endpoints  (through hilbert_native.pyd)

 It implements a lightweight agent-based informational field model:

   * Agents live in a 2D [0,1] x [0,1] space.
   * Each agent has:
       - belief    in [0,1]
       - regime    0=info, 1=misinfo, 2=disinfo
   * Steps:
       - random walk (diffusion)
       - weak relaxation toward global mean (coherence)
       - small noise term (stochasticity)
   * Disinfo injection:
       - selects agents in a given radius
       - pushes their regime/belief toward the requested regime
   * MetricsOut:
       - mean_belief
       - polarization (variance/dispersion of belief)
       - entropy over regime distribution

 This is intentionally simple but structurally aligned with your thesis and
 existing visualization logic. You can swap in your full resonance/OMP model
 internally while keeping this public API stable.

===============================================================================
*/

#include "hilbert_simulation.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// -----------------------------------------------------------------------------
// Internal structures
// -----------------------------------------------------------------------------

typedef struct {
    double x;       // position in [0,1]
    double y;       // position in [0,1]
    double belief;  // scalar belief in [0,1]
    int    regime;  // 0=info, 1=misinfo, 2=disinfo
} Agent;

// Global simulation state
static Agent *g_agents      = NULL;
static int    g_num_agents  = 0;
static int    g_initialized = 0;

// Tunable parameters
static const double DIFFUSION_STEP   = 0.02;   // spatial step size
static const double RELAX_STRENGTH   = 0.05;   // pull toward mean
static const double NOISE_STRENGTH   = 0.02;   // random noise in belief

// -----------------------------------------------------------------------------
// Utility
// -----------------------------------------------------------------------------

static double clamp01(double v) {
    if (v < 0.0) return 0.0;
    if (v > 1.0) return 1.0;
    return v;
}

static double frand01(void) {
    return (double)rand() / (double)RAND_MAX;
}

static void ensure_seeded(void) {
    static int seeded = 0;
    if (!seeded) {
        seeded = 1;
        srand((unsigned int)time(NULL));
    }
}

// -----------------------------------------------------------------------------
// Public API: initialization
// -----------------------------------------------------------------------------

int sim_init(int n_agents) {
    ensure_seeded();

    if (n_agents <= 0) n_agents = 256;

    if (g_agents) {
        free(g_agents);
        g_agents = NULL;
    }

    g_agents = (Agent *)calloc((size_t)n_agents, sizeof(Agent));
    if (!g_agents) {
        fprintf(stderr, "[hilbert_sim] Failed to allocate %d agents\n", n_agents);
        g_num_agents = 0;
        g_initialized = 0;
        return 0;
    }

    g_num_agents  = n_agents;
    g_initialized = 1;

    // Initialize agents in a roughly neutral but slightly noisy informational field.
    for (int i = 0; i < g_num_agents; ++i) {
        Agent *a = &g_agents[i];
        a->x = frand01();
        a->y = frand01();

        // Start beliefs around 0.5 with small noise.
        double noise = (frand01() - 0.5) * 0.2;
        a->belief = clamp01(0.5 + noise);

        // Regime: mostly info, some mis/dis sprinkled in.
        double r = frand01();
        if (r < 0.70)
            a->regime = 0; // info
        else if (r < 0.90)
            a->regime = 1; // misinfo
        else
            a->regime = 2; // disinfo
    }

    return 1;
}

// -----------------------------------------------------------------------------
// Public API: one simulation step
// -----------------------------------------------------------------------------

void sim_step(double dt) {
    if (!g_initialized || !g_agents || g_num_agents <= 0) return;
    if (dt <= 0.0) dt = 0.1;

    // 1) Compute global mean belief for a simple "field" influence.
    double sum_belief = 0.0;
    for (int i = 0; i < g_num_agents; ++i) {
        sum_belief += g_agents[i].belief;
    }
    double mean_belief = (g_num_agents > 0)
        ? (sum_belief / (double)g_num_agents)
        : 0.5;

    // 2) Update each agent.
    for (int i = 0; i < g_num_agents; ++i) {
        Agent *a = &g_agents[i];

        // Random walk in space.
        double dx = (frand01() - 0.5) * DIFFUSION_STEP;
        double dy = (frand01() - 0.5) * DIFFUSION_STEP;
        a->x = clamp01(a->x + dx);
        a->y = clamp01(a->y + dy);

        // Relax belief slightly toward global mean (coherence pressure).
        a->belief += RELAX_STRENGTH * dt * (mean_belief - a->belief);

        // Add noise (entropy).
        a->belief += NOISE_STRENGTH * dt * ((frand01() - 0.5) * 2.0);

        // Clamp.
        a->belief = clamp01(a->belief);
    }
}

// -----------------------------------------------------------------------------
// Public API: inject dis/mis-information in a spatial region
// -----------------------------------------------------------------------------

void sim_inject_disinfo(double x, double y, double radius, int regime) {
    if (!g_initialized || !g_agents || g_num_agents <= 0) return;

    if (radius <= 0.0) radius = 0.15;
    double r2 = radius * radius;

    // Regime clamp: 0=info, 1=misinfo, 2=disinfo
    if (regime < 0) regime = 0;
    if (regime > 2) regime = 2;

    // Target belief center for each regime.
    double target_belief;
    switch (regime) {
        case 0:  target_belief = 0.8; break; // confident / informational
        case 1:  target_belief = 0.5; break; // ambiguous / misinfo
        default: target_belief = 0.2; break; // corrosive / disinfo
    }

    for (int i = 0; i < g_num_agents; ++i) {
        Agent *a = &g_agents[i];
        double dx = a->x - x;
        double dy = a->y - y;
        double d2 = dx*dx + dy*dy;
        if (d2 <= r2) {
            double w = 1.0 - (d2 / r2);  // stronger at center
            a->regime = regime;
            a->belief = clamp01(a->belief * (1.0 - 0.5*w) + target_belief * (0.5*w));
        }
    }
}

// -----------------------------------------------------------------------------
// Public API: metrics
// -----------------------------------------------------------------------------

MetricsOut compute_metrics_values(void) {
    MetricsOut m;
    m.mean_belief = 0.0;
    m.polarization = 0.0;
    m.entropy = 0.0;

    if (!g_initialized || !g_agents || g_num_agents <= 0) {
        return m;
    }

    // Mean belief
    double sum = 0.0;
    for (int i = 0; i < g_num_agents; ++i) {
        sum += g_agents[i].belief;
    }
    double mean = sum / (double)g_num_agents;

    // Polarization: variance of belief (0=uniform, high=polarized)
    double var = 0.0;
    for (int i = 0; i < g_num_agents; ++i) {
        double d = g_agents[i].belief - mean;
        var += d * d;
    }
    var /= (double)g_num_agents;

    // Regime entropy: how mixed info/mis/dis are.
    double counts[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < g_num_agents; ++i) {
        int r = g_agents[i].regime;
        if (r < 0) r = 0;
        if (r > 2) r = 2;
        counts[r] += 1.0;
    }

    double ent = 0.0;
    for (int r = 0; r < 3; ++r) {
        if (counts[r] > 0.0) {
            double p = counts[r] / (double)g_num_agents;
            ent -= p * log(p + 1e-18);
        }
    }

    m.mean_belief  = mean;
    m.polarization = var;   // you may map this into [0,1] if desired
    m.entropy      = ent;   // natural-log entropy; can be normalized if needed

    return m;
}

// -----------------------------------------------------------------------------
// Public API: export current state to JSON
// -----------------------------------------------------------------------------

void sim_export_json(const char *path_json) {
    if (!path_json || !g_initialized || !g_agents || g_num_agents <= 0) {
        return;
    }

    FILE *fp = fopen(path_json, "w");
    if (!fp) {
        fprintf(stderr, "[hilbert_sim] Failed to open %s for writing\n", path_json);
        return;
    }

    fprintf(fp, "{\n");
    fprintf(fp, "  \"num_agents\": %d,\n", g_num_agents);

    MetricsOut m = compute_metrics_values();
    fprintf(fp, "  \"metrics\": {\n");
    fprintf(fp, "    \"mean_belief\": %.6f,\n", m.mean_belief);
    fprintf(fp, "    \"polarization\": %.6f,\n", m.polarization);
    fprintf(fp, "    \"entropy\": %.6f\n", m.entropy);
    fprintf(fp, "  },\n");

    fprintf(fp, "  \"agents\": [\n");
    for (int i = 0; i < g_num_agents; ++i) {
        const Agent *a = &g_agents[i];
        fprintf(
            fp,
            "    {\"x\":%.5f,\"y\":%.5f,\"belief\":%.5f,\"regime\":%d}%s\n",
            a->x,
            a->y,
            a->belief,
            a->regime,
            (i + 1 < g_num_agents) ? "," : ""
        );
    }
    fprintf(fp, "  ]\n");
    fprintf(fp, "}\n");

    fclose(fp);
}

