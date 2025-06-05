"""
Author: Eric Tran <eric.tran@heig-vd.ch>
"""
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from DAGGeneratorGA import DAGGeneratorGA
from DAGDatasetGenerator import DAGDatasetGenerator
import copy
import time
import datetime

KEEP_DAG_COUNT = 1
KEEP_RANDOM_DAG = False
MAX_STEPS = -1
VERBOSE = False
MIN_NODES = 41
MAX_NODES = 60
N_ITERATIONS = 10

generator = DAGDatasetGenerator()
dga = DAGGeneratorGA()

def check_validity_dags(dags):
    for i, dag in enumerate(dags):
        G = nx.DiGraph()
        G.add_edges_from(dag)
        nodes = nx.bfs_tree(G, 0).nodes
        if len(nodes) != N:
            print(f"Issue with DAG {i}")

def serialize_dag(edges: list):
    str_dag = ';'.join(f"{parent}:{node}" for parent, node in edges)
    return str_dag


def test_crosscheck_with_simulator():
    benchmark_results = []
    dga_results = []
    filename_prefix = str(datetime.datetime.now()).replace(":", "_")
    print("--------test_crosscheck_with_simulator--------")
    for n in range(MIN_NODES, MAX_NODES + 1):
        for it in range(N_ITERATIONS):
            print(f"Run for {n} nodes, iteration {it+1}/{N_ITERATIONS}")
            dags = []
            while len(dags) == 0:
                # Generate a random adjacency matrix
                adj_matrix, density_factor = generator.generate_random_adj_matrix(n)
                dags = generator.generate_n_subset_dags_c(10, adj_matrix) # Verify adjacent matrix is usable
            # Genetic Algoritm
            start_time = time.time()
            best_dga_dag, best_dga_perf = dga.fit(adj_matrix, population_size=500, generations=1)
            end_time = time.time()
            dga_results.append([n, best_dga_perf, end_time - start_time, serialize_dag(best_dga_dag)])
            # Benchmark
            start_time = time.time()
            benchmark_list = generator.generate_subset_dags(adj_matrix)
            best_benchmark_dags, best_benchmark_perfs = generator.get_best_dag_double_flux(benchmark_list, adj_matrix, KEEP_DAG_COUNT, KEEP_RANDOM_DAG, max_steps=MAX_STEPS, verbose=VERBOSE)
            end_time = time.time()
            benchmark_results.append([n, best_benchmark_perfs[0], end_time - start_time, serialize_dag(best_benchmark_dags[0])])

            # Export at each iteration to avoid data loss 
            df_dga = pd.DataFrame(dga_results, columns=['n_nodes', 'perf', 'run_time', 'dag'])
            df_benchmark = pd.DataFrame(benchmark_results, columns=['n_nodes', 'perf', 'run_time', 'dag'])
            df_dga.to_csv(filename_prefix + "_dga.csv", index=None)
            df_benchmark.to_csv(filename_prefix + "_benchmark.csv", index=None)

def test_routing_ga_scale():
    dga_results = []
    filename_prefix = str(datetime.datetime.now()).replace(":", "_")
    print("--------test_routing_ga_scale--------")
    for n in range(MIN_NODES, MAX_NODES + 1):
        it = 0
        while it < N_ITERATIONS:
            print(f"Run for {n} nodes, iteration {it+1}/{N_ITERATIONS}")
            # Generate a random adjacency matrix
            print("Generate adjacent matrix")
            adj_matrix, density_factor = generator.generate_random_adj_matrix(n)
            all_combined_lq = np.sum(adj_matrix, axis=1)
            if any(all_combined_lq == 0.0):
                print("Malformed adjacent matrix, nodes without link quality. Pass")
                continue

            # Genetic Algoritm
            print("Run genetic algorithm")
            start_time = time.time()
            try:
                best_dga_dag, best_dga_perf = dga.fit(adj_matrix, population_size=500, generations=1, verbose=True)
            except Exception as e:
                print(e)
                continue
            end_time = time.time()
            dga_results.append([n, best_dga_perf, end_time - start_time, serialize_dag(best_dga_dag)])

            # Export at each iteration to avoid data loss 
            print("Export data")
            df_dga = pd.DataFrame(dga_results, columns=['n_nodes', 'perf', 'run_time', 'dag'])
            df_dga.to_csv(filename_prefix + "_dga_scale_run.csv", index=None)
            it += 1


if __name__ == '__main__':
    #test_crosscheck_with_simulator()
    test_routing_ga_scale()