/**
 * @brief C++ Shared library to accelerate execution of python simulation
 * @author Yann Charbon <yann.charbon@heig-vd.ch>
 * @note See README for more information
 */

#pragma once

typedef struct {
    int parent;
    int child;
} Edge;

extern "C" {
    int evaluate_dag_performance_up(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps, int *bottleneck_factors);
    int evaluate_dag_performance_down(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps, int *bottleneck_factors);
    int evaluate_dag_performance_double_flux(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps, int *bottleneck_factors);
    Edge** generate_subset_dags(float **adj_matrix, int nodes_count, int *generated_dags_count, int skip_factor, bool test_mode, bool no_skip);
    void free_all_possible_tree(Edge **all_possible_trees_c, int dags_count);
}