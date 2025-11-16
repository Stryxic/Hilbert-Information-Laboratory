/*
===============================================================================
 hilbert_native.h — Unified C Interface for the Hilbert Information Chemistry Lab
-------------------------------------------------------------------------------
 Provides C-exported entry points for:

   • Spectral and field-level analysis (Hilbert core)
   • Numerical metrics and informational stability computations
   • Data persistence and graph export
   • Agent-based simulation (informational compounds)

 This file also declares the internal simulation backend so hilbert_native.c
 can link correctly to hilbert_simulation.c.
===============================================================================
*/

#ifndef HILBERT_NATIVE_H
#define HILBERT_NATIVE_H

#include <stddef.h>   // size_t
#include <stdint.h>   // standard integer types

// -----------------------------------------------------------------------------
// Windows DLL Export / Import
// -----------------------------------------------------------------------------
#ifdef _WIN32
  #ifdef HILBERT_NATIVE_EXPORTS
    #define HILBERT_API __declspec(dllexport)
  #else
    #define HILBERT_API __declspec(dllimport)
  #endif
#else
  #define HILBERT_API
#endif


// ============================================================================
// Versioning and Lifecycle
// ============================================================================
HILBERT_API double hilbert_version(void);
HILBERT_API int    hilbert_init(void);
HILBERT_API void   hilbert_shutdown(void);


// ============================================================================
// Memory Management
// ============================================================================
HILBERT_API void*  hilbert_alloc(size_t bytes);
HILBERT_API void   hilbert_free(void* p);
HILBERT_API void   hilbert_set_output_dir(const char* path);


// ============================================================================
// Spectral and Informational Metrics
// ============================================================================
HILBERT_API double hilbert_spectral_entropy(const double* v, int n);

HILBERT_API double hilbert_coherence_score(const double* A,
                                           int rows,
                                           int cols);

HILBERT_API double hilbert_information_stability(double entropy,
                                                 double coherence);

HILBERT_API double hilbert_temporal_decay(double prev_val,
                                          double next_val,
                                          double alpha);


// ============================================================================
// Graph Export
// ============================================================================
HILBERT_API int hilbert_graph_export_edges(const char* path,
                                           const char** src,
                                           const char** tgt,
                                           const double* w,
                                           int n_edges);


// ============================================================================
// JSON Store API
// ============================================================================
HILBERT_API int   hilbert_append_json(const char* kind,
                                      const char* json_record);

HILBERT_API char* hilbert_read_store(const char* kind);


// ============================================================================
// Diagnostics
// ============================================================================
HILBERT_API void hilbert_banner(void);
HILBERT_API int  hilbert_selftest(void);


// ============================================================================
// Simulation API — High-Level Wrappers (exported to Python)
// ============================================================================
HILBERT_API int  hilbert_sim_init(int n_agents);
HILBERT_API void hilbert_sim_step(double dt);
HILBERT_API void hilbert_sim_inject(double x,
                                    double y,
                                    double radius,
                                    int regime);
HILBERT_API void hilbert_sim_metrics(double* mean,
                                     double* pol,
                                     double* entropy);
HILBERT_API void hilbert_sim_export_state(const char* path_json);


// ============================================================================
// INTERNAL SIMULATION BACKEND — required for linking
// ============================================================================
// These are *not* exported. These are the functions actually implemented in
// hilbert_simulation.c. hilbert_native.c calls these so they must be visible.

int  hilbert_simulation_init(int n_agents);
void hilbert_simulation_step(double dt);
void hilbert_simulation_inject(double x,
                               double y,
                               double radius,
                               int regime);
void hilbert_simulation_metrics(double* mean,
                                double* pol,
                                double* entropy);
void hilbert_simulation_export_state(const char* path_json);

#endif // HILBERT_NATIVE_H
