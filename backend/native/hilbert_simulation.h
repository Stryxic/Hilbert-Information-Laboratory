#ifndef HILBERT_SIMULATION_H
#define HILBERT_SIMULATION_H

typedef struct {
    double mean_belief;
    double polarization;
    double entropy;
} MetricsOut;

int        sim_init(int n_agents);
void       sim_step(double dt);
void       sim_inject_disinfo(double x, double y, double radius, int regime);
MetricsOut compute_metrics_values(void);
void       sim_export_json(const char *path_json);

#endif
