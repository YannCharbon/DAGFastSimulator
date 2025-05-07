"""
Author: Yann Charbon <yann.charbon@heig-vd.ch>
Author: Eric Tran <eric.tran@heig-vd.ch>
"""
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt
import random
import time
import datetime
import concurrent.futures
import os
import gc
import heapq

from pathlib import Path
import ctypes

# Define the C struct in ctypes
class Edge(ctypes.Structure):
    _fields_ = [("parent", ctypes.c_int), ("child", ctypes.c_int)]


def callback_future_end(future):
    try:
        future.result()
    except Exception as e:
        print(e)
        raise e

class DAGDatasetGenerator:
    def __init__(self):
        # Enable interactive mode for non-blocking plotting
        plt.ion()

    """
    Runs 'count' times the simulation for  a mesh network of 'n' nodes in parallel with pure C simulation.

    The simulation is performed in two independent passes (UP and DOWN traffic). This is not the best way
    to simulate a real-world behaviour but is faster than the combined version. Please use 'run_double_flux'
    if more accuracy is needed (recommended).
    """
    def run_up_down(self, n, count, keep_dags_count=1, keep_random_dags=False, max_workers=os.cpu_count(), verbose=False, dags_folder_path="dags"):
        dags_path = Path(dags_folder_path)
        dags_path.mkdir(exist_ok=True)

        generated_count = 0

        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_once_up_down, n, keep_dags_count, keep_random_dags, verbose) for i in range(count)}
            for future in concurrent.futures.as_completed(futures):
                best_dags, best_perfs, adj_matrix = future.result()
                # Initialize an empty MultiDiGraph
                G = nx.MultiDiGraph()

                # Add all edges from the adjacency matrix (topology edges)
                for u, v in zip(*adj_matrix.nonzero()):  # Find all nonzero entries
                    G.add_edge(u, v, key=(0, 0), link_quality=float(adj_matrix[u][v]), edge_type='topology')

                # Add edges from each of the best DAGs
                for idx, dag in enumerate(best_dags):
                    for edge in dag:
                        G.add_edge(
                            edge[0], edge[1], key=(idx + 1, 1),  # Unique key for each DAG edge
                            link_quality=float(adj_matrix[edge[0]][edge[1]]),
                            edge_type=f'dag_{idx}'
                        )

                # Write the graph to a CSV file
                futures.remove(future)
                uid = str(datetime.datetime.now()).replace(":", "_")
                result_name = f"topo_{uid}.csv"
                nx.write_edgelist(G, dags_path / Path(result_name), delimiter=',')

                generated_count += 1

                if verbose == False:
                    print(f"> Status: [{generated_count} / {count}]", end='\r')

        end_time = time.time()
        print(f"Total runtime : {end_time - start_time}")

    """
    Runs 'count' times the simulation for a mesh network of 'n' nodes in parallel with pure C simulation.

    The simulation is performed in a single pass (combining UP and DOWN traffic). This achieves to simulate
    a simplified real-world mesh network accurately. It is slightly slower than the 'run_up_down'.
    """
    def run_double_flux(self, n, count, keep_dags_count=1, keep_random_dags=False, max_workers=os.cpu_count(), verbose=False, dags_folder_path="dags"):
        dags_path = Path(dags_folder_path)
        dags_path.mkdir(exist_ok=True)

        generated_count = 0

        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            for i in range(count):
                future = executor.submit(self.run_once_double_flux, n, keep_dags_count, keep_random_dags, verbose)
                future.add_done_callback(callback_future_end)
                futures.add(future)

            for future in concurrent.futures.as_completed(futures):
                best_dags, best_perfs, adj_matrix = future.result()
                # Initialize an empty MultiDiGraph
                G = nx.MultiDiGraph()

                # Add all edges from the adjacency matrix (topology edges)
                for u, v in zip(*adj_matrix.nonzero()):  # Find all nonzero entries
                    G.add_edge(u, v, key=(0, 0), link_quality=float(adj_matrix[u][v]), edge_type='topology')

                # Add edges from each of the best DAGs
                for idx, dag in enumerate(best_dags):
                    for edge in dag:
                        G.add_edge(
                            edge[0], edge[1], key=(idx + 1, 1),  # Unique key for each DAG edge
                            link_quality=float(adj_matrix[edge[0]][edge[1]]),
                            edge_type=f'dag_{idx}'
                        )

                # Write the graph to a CSV file
                futures.remove(future)
                uid = str(datetime.datetime.now()).replace(":", "_")
                result_name = f"topo_{uid}.csv"
                nx.write_edgelist(G, dags_path / Path(result_name), delimiter=',')

                generated_count += 1

                if verbose == False:
                    print(f"> Status: [{generated_count} / {count}]", end='\r')

        end_time = time.time()
        print(f"Total runtime : {end_time - start_time}")


    """
    Runs the simulation (two pass up/down) for a single topology of n nodes.

    The following sequence is performed:
    - Generate a random adjacency matrix of size 'n x n'
    - Compute a subset of all the possible DAGs that can be found in the topology formed by the adjacency matrix
    - Find the best DAG for the subset by running simulation on the whole subset
    """
    def run_once_up_down(self, n, keep_dags_count=1, keep_random_dags=False, verbose=False):
        dags = []
        adj_matrix = []
        best_dags = []
        while len(best_dags) < keep_dags_count:
            while (len(dags) == 0):
                # Generate a random adjacency matrix
                adj_matrix, density_factor = self.generate_random_adj_matrix(n)
                if verbose:
                    print("Density factor = {}".format(density_factor))

                # Get all the possible DAGs within this topology
                if n < 10:
                    dags = self.generate_subset_dags(adj_matrix, no_skip=True)
                else:
                    dags = self.generate_subset_dags(adj_matrix)
                if verbose:
                        print(f"Number of DAGs generated: {len(dags)}")
                        if len(best_dags) < keep_dags_count:
                            print(f"Warning: Only generated {len(best_dags)} DAG(s), requested {keep_dags_count}. Regenerating")

            # Compute the best performing DAG within the topology
            best_dags, best_perfs = self.get_best_dag_up_down(dags, adj_matrix, keep_dags_count, keep_random_dags, verbose=verbose)
            if verbose:
                print("best dag is {} perf = {}".format(best_dags[0], best_perfs[0]))
                np.set_printoptions(formatter={'all': lambda x: "{:.4g},".format(x)})
                print(adj_matrix)

        return best_dags, best_perfs, adj_matrix

    """
    Runs the simulation (combined double flux) for a single topology of n nodes.

    The following sequence is performed:
    - Generate a random adjacency matrix of size 'n x n'
    - Compute a subset of all the possible DAGs that can be found in the topology formed by the adjacency matrix
    - Find the best DAG for the subset by running simulation on the whole subset

    Note: max_steps can be provided to impose the maximum number of steps to be adaptative during the get_best_dag_double_flux computation (can be useful to keep an accurate precision on best_perfs content)
    """
    def run_once_double_flux(self, n, keep_dags_count=1, keep_random_dags=False, max_steps=-1, verbose=False):
        dags = []
        adj_matrix = []
        best_dags = []
        while len(best_dags) < keep_dags_count:
            while (len(dags) == 0):
                # Generate a random adjacency matrix
                adj_matrix, density_factor = self.generate_random_adj_matrix(n)
                if verbose:
                    print("Density factor = {}".format(density_factor))
                # Get all the possible DAGs within this topology
                if n < 11:
                    dags = self.generate_subset_dags(adj_matrix, no_skip=True)
                else:
                    dags = self.generate_subset_dags(adj_matrix)
                if verbose:
                    print(f"Number of DAGs generated: {len(dags)}")
                    if len(best_dags) < keep_dags_count:
                        print(f"Warning: Only generated {len(best_dags)} DAG(s), requested {keep_dags_count}. Regenerating")

            # Compute the best performing DAG within the topology
            best_dags, best_perfs = self.get_best_dag_double_flux(dags, adj_matrix, keep_dags_count, keep_random_dags, max_steps=max_steps, verbose=verbose)
            if verbose:
                print("best dag is {} perf = {}".format(best_dags[0], best_perfs[0]))
                np.set_printoptions(formatter={'all': lambda x: "{:.4g},".format(x)})
                print(adj_matrix)

        return best_dags, best_perfs, adj_matrix

    """
    Generates a random adjacency matrix of size 'n x n'

    A fixed density factor can be provided for testing purpose (random if == 0.0).
    """
    def generate_random_adj_matrix(self, n, fixed_density_factor=0.0):
        def check_integrity(a):
            def dfs(v, visited, matrix):
                visited[v] = True
                for i in range(len(matrix)):
                    if matrix[v][i] > 0 and not visited[i]:
                        dfs(i, visited, matrix)

            visited = [False] * len(a)
            dfs(0, visited, a)

            # If any node is not visited, then the graph is not fully connected
            return all(visited)

        def symmetrize(a):
            a_sym = a + a.T - np.diag(a.diagonal())
            return a_sym * np.max(a) / np.max(a_sym)

        def limit_neighbors(a, max_neighbors=4):
            for i in range(len(a)):
                non_zero_indices = np.nonzero(a[i])[0]
                if len(non_zero_indices) > max_neighbors:
                    # Get the indices of the top 3 strongest connections
                    top_indices = np.argsort(a[i][non_zero_indices])[-max_neighbors:]
                    # Zero out all other connections
                    to_zero = np.setdiff1d(non_zero_indices, non_zero_indices[top_indices])
                    a[i][to_zero] = 0
                    a[:, i][to_zero] = 0  # Symmetrize manually
            return a

        # Init matrix
        a = np.zeros((n, n))

        density_factor = 0.0

        # Link with best quality = 1.0 No connection = 0.0
        # the ' - 2 * np.random.rand()' controls the density of the interconnections
        rng = np.random.default_rng() # Required in multiprocessing to avoid having same random values in all processes

        # A potential valid matrix must have at least enough edges to interconnect each node in the topology
        while np.asarray(a > 0.0).sum() < n - 1:
            if fixed_density_factor > 0.0:
                density_factor = fixed_density_factor
            else:
                density_factor = rng.random()
            a = np.maximum(rng.random((n, n)) * 2 - 1 - 1 * (1 - density_factor), np.zeros((n, n)))

        a = symmetrize(a)

        while not check_integrity(a):
            if fixed_density_factor > 0.0:
                density_factor = fixed_density_factor
            else:
                density_factor = rng.random()
            a = np.maximum(rng.random((n, n)) * 2 - 1 - 1 * (1 - density_factor), np.zeros((n, n)))
            if np.asarray(a > 0.0).sum() < n - 1:
                continue
            a = symmetrize(a)

        a = limit_neighbors(a)

        # Rescale link quality between 0.85 and 1.0 (values below represent a loss which is not realistic in reality)
        # Rescale values strictly greater than 0.0 to be between 0.85 and 1.0
        new_min, new_max = 0.85, 1.0    # New range

        # Apply rescaling only to elements greater than 0.0
        a = np.where(
            a > 0.0,
            a * (new_max - new_min) + new_min,
            a
        )

        np.fill_diagonal(a, 0.) # Remove all potentials link to itself
        return a, density_factor

    """
    Generate a subset of all the possible DAGs within a topology formed by the provided adjacency matrix

    A skip factor can be provided (0 == automatic).
    In test mode, the generated subset is deterministic (no random skipping)
    It is possible to generate all the DAGs possible from the topology by enabling no_skip.
    """
    def generate_subset_dags(self, adj_matrix, skip_factor=0, test_mode=False, no_skip=False):
        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.generate_subset_dags.argtypes = (ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_bool, ctypes.c_bool)
        lib.generate_subset_dags.restype = ctypes.POINTER(ctypes.POINTER(Edge))

        lib.free_all_possible_tree.argtypes = (ctypes.POINTER(ctypes.POINTER(Edge)), ctypes.c_int)

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        generated_dags_count = 0
        generated_dags_count_c = ctypes.c_int(generated_dags_count)

        all_possible_dags_c = lib.generate_subset_dags(adj_matrix_c, len(adj_matrix[0]), ctypes.pointer(generated_dags_count_c), skip_factor, test_mode, no_skip)

        # Convert the C result back to Python list of lists
        all_possible_dags = []
        for i in range(generated_dags_count_c.value):
            # Each row in `c_dag_array` is a pointer to an array of `Edge`
            dag_size = len(adj_matrix[0]) - 1  # Based on the tree's edge count
            dag = [(all_possible_dags_c[i][j].parent, all_possible_dags_c[i][j].child) for j in range(dag_size)]
            all_possible_dags.append(dag)

        lib.free_all_possible_tree(all_possible_dags_c, generated_dags_count_c)

        return all_possible_dags

    """
    Python wrapper for C DAG simulation (UP/DOWN).

    This computes the simulation in two independent passes. It computes the performance upwards (emptying the network) and the downwards (filling the network).
    The results are averaged over 'epoch_len' runs. The initial (UP) or target (down) packet count can be specified using 'packets_per_node'.
    'max_steps_up' and 'max_steps_down' can be used to early-stop the simulation in case of timeout or a quality threshold (e.g. stopping because the current DAG performs worse than others).

    It is possible to monitor the bottlenecks by passing an int array (bottleneck_factors). When any node tries to send to a target node which is busy, the bottleneck factor of the target node
    is incremented. The higher the bottleneck factor is, the busier a node is.

    Returns the UP and DOWN performance (i.e. the number of simulation steps to empty/fill the network). A lower value <=> better performance.
    """
    def evaluate_dag_performance_up_down(self, dag, adj_matrix, epoch_len=1, packets_per_node=15, max_steps_up=-1, max_steps_down=-1, bottleneck_factors=None):
        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_up.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_up.restype = ctypes.c_int

        lib.evaluate_dag_performance_down.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_down.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        c_dag = (Edge * len(dag))(
            *[Edge(parent, child) for parent, child in dag]
        )

        if bottleneck_factors:
            bottleneck_factors_c = (ctypes.c_int * len(bottleneck_factors))(*bottleneck_factors)
        else:
            bottleneck_factors_c = None  # NULL

        perf_up = int(lib.evaluate_dag_performance_up(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps_up, bottleneck_factors_c))
        perf_down = int(lib.evaluate_dag_performance_down(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps_down, bottleneck_factors_c))

        if bottleneck_factors:
            for i in range(len(bottleneck_factors)):
                bottleneck_factors[i] = bottleneck_factors_c[i]

        return dag, perf_up, perf_down

    """
    Python wrapper for C DAG simulation (double flux).

    This computes the simulation in a single pass. It computes the performance upwards (emptying the network) and the downwards (filling the network) together (the two flux interact together).
    The results are averaged over 'epoch_len' runs. The initial (UP) or target (down) packet count can be specified using 'packets_per_node'.
    'max_steps_up' and 'max_steps_down' can be used to early-stop the simulation in case of timeout or a quality threshold (e.g. stopping because the current DAG performs worse than others).

    It is possible to monitor the bottlenecks by passing an int array (bottleneck_factors). When any node tries to send to a target node which is busy, the bottleneck factor of the target node
    is incremented. The higher the bottleneck factor is, the busier a node is.

    Returns the performance (i.e. the number of simulation steps to empty/fill the network). A lower value <=> better performance.
    """
    def evaluate_dag_performance_double_flux(self, dag, adj_matrix, epoch_len=1, packets_per_node=15, max_steps=-1, bottleneck_factors=None):
        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_double_flux.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        lib.evaluate_dag_performance_double_flux.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        c_dag = (Edge * len(dag))(
            *[Edge(parent, child) for parent, child in dag]
        )

        if bottleneck_factors:
            bottleneck_factors_c = (ctypes.c_int * len(bottleneck_factors))(*bottleneck_factors)
        else:
            bottleneck_factors_c = None  # NULL

        perf = int(lib.evaluate_dag_performance_double_flux(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps, bottleneck_factors_c))

        if bottleneck_factors:
            for i in range(len(bottleneck_factors)):
                bottleneck_factors[i] = bottleneck_factors_c[i]

        return dag, perf


    """
    Runs the UP/DOWN simulation on each DAG to get the best performing one.
    """
    def get_best_dag_up_down(self, dags, adj_matrix, keep_dags_count=1, keep_random_dags=False, max_workers=os.cpu_count(), delta_threshold=0.8, reduce_ratio = 0.2, margin_max_step = 1.1, verbose=False):
        start_time = time.time()

        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_up.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_up.restype = ctypes.c_int

        lib.evaluate_dag_performance_down.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_down.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                dag = random.choice(dags)
                c_dag = (Edge * len(dag))(
                    *[Edge(parent, child) for parent, child in dag]
                )
                max_steps_up = int(min(lib.evaluate_dag_performance_up(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, -1), max_steps_up))
                max_steps_down = int(min(lib.evaluate_dag_performance_down(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, -1), max_steps_down))
            if verbose:
                print("Max steps UP = " + str(max_steps_up))
                print("Max steps DOWN = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        up_results = []
        down_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_dag_performance_up_down,
                    dags.pop(),
                    adj_matrix,
                    max_steps_up=max_steps_up,
                    max_steps_down=max_steps_down)
                for i in range(min(max_workers, len(dags)))}
            while futures:
                for future in concurrent.futures.as_completed(futures):
                    current_dag, perf_up, perf_down = future.result()
                    up_results.append((current_dag, perf_up))
                    down_results.append((current_dag, perf_down))

                    if max_steps_up == -1 and max_steps_down == -1:
                        max_steps_up = perf_up
                        max_steps_down = perf_down

                    if perf_up < max_steps_up:
                        max_steps_up = int(perf_up if (perf_up / max_steps_up) > delta_threshold else max_steps_up * (1 - reduce_ratio))

                    if perf_down < max_steps_down:
                        max_steps_down = int(perf_down if (perf_down / max_steps_down) > delta_threshold else max_steps_down * (1 - reduce_ratio))

                    futures.remove(future)
                    if len(dags):
                        futures.add(executor.submit(
                            self.evaluate_dag_performance_up_down,
                            dags.pop(),
                            adj_matrix,
                            max_steps_up=int(max_steps_up * margin_max_step),
                            max_steps_down=int(max_steps_down * margin_max_step))
                        )

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results]).astype(float)
        down_results_np = np.array([item[1] for item in down_results]).astype(float)

        # normalize and combine
        max_up = np.max(up_results_np)
        max_down = np.max(down_results_np)
        up_results_np /= max_up
        down_results_np /= max_down
        combined_results_np = (up_results_np + down_results_np) * (max_up + max_down) / 2

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        best_dag_up, best_perf_up = min(up_results, key=lambda x: x[1])
        best_dag_down, best_perf_down = min(down_results, key=lambda x: x[1])

        sorted_combined_results = combined_results
        sorted_combined_results.sort(key=lambda x: x[1], reverse=True)
        best_dag_up_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_up))
        best_dag_down_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_down))

        # Find the best DAG based on up and down performance
        best_dags = []
        best_perfs = []
        if keep_random_dags == False or (keep_dags_count == 1 and keep_random_dags == True):
            lowests = heapq.nsmallest(keep_dags_count, combined_results, key=lambda x: x[1])
            best_dags, best_perfs = zip(*lowests)
        else:
            absolute_best_dag, absolute_best_perf = min(combined_results, key=lambda x: x[1])
            num_choices = min(keep_dags_count - 1, len(combined_results))
            randoms = random.choices(combined_results, k=num_choices)
            best_dags = (absolute_best_dag, *(r[0] for r in randoms))
            best_perfs = (absolute_best_perf, *(r[1] for r in randoms))

        end_time = time.time()
        #print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        #print("Info: best DAG UP rank = {}/{} (perf {}) and best DAG DOWN rank = {}/{} (perf {}) compared to overall best score".format(best_dag_up_overall_score, len(combined_results), best_perf_up, best_dag_down_overall_score, len(combined_results), best_perf_down))

        return best_dags, best_perfs

    """
    Runs the double flux simulation on each DAG to get the best performing one.
    """
    def get_best_dag_double_flux(self, dags, adj_matrix, keep_dags_count=1, keep_random_dags=False, max_workers=os.cpu_count(), delta_threshold=0.8, reduce_ratio = 0.2, margin_max_step = 1.1, max_steps=-1, verbose=False):
        start_time = time.time()

        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_double_flux.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_double_flux.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        adaptative_max_steps = False
        if max_steps == -1:
            adaptative_max_steps = True
            if len(dags) > 800:
                iter = 50 if len(dags) >= 50 else len(dags)
                for _ in range(0, iter):
                    dag = random.choice(dags)
                    c_dag = (Edge * len(dag))(
                        *[Edge(parent, child) for parent, child in dag]
                    )
                    max_steps = int(min(lib.evaluate_dag_performance_double_flux(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, -1), max_steps))
                if verbose:
                    print("Max steps = " + str(max_steps))
            else:
                max_steps = -1

        idx = 0
        results = []    # For pre-allocated memory use following and adjust below [([(0, 0)] * len(adj_matrix[0]), 0) for _ in range(len(dags))]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            for i in range(min(max_workers, len(dags))):
                future = executor.submit(
                    self.evaluate_dag_performance_double_flux,
                    dags.pop(),
                    adj_matrix,
                    max_steps=max_steps)
                future.add_done_callback(callback_future_end)
                futures.add(future)

            while futures:
                for future in concurrent.futures.as_completed(futures):
                    current_dag, perf = future.result()
                    results.append((current_dag, perf))
                    idx += 1

                    if idx % 10000 == 0:
                        gc.collect()

                    if adaptative_max_steps:
                        if max_steps == -1:
                            max_steps = perf

                        if perf < max_steps:
                            max_steps = int(perf if (perf / max_steps) > delta_threshold else max_steps * (1 - reduce_ratio))

                    current_dag = None
                    perf = None
                    del current_dag, perf

                    futures.remove(future)
                    del future
                    if len(dags):
                        if adaptative_max_steps == True:
                            future = executor.submit(
                                self.evaluate_dag_performance_double_flux,
                                dags.pop(),
                                adj_matrix,
                                max_steps=int(max_steps * margin_max_step))
                        else:
                            future = executor.submit(
                                self.evaluate_dag_performance_double_flux,
                                dags.pop(),
                                adj_matrix,
                                max_steps=max_steps)
                        future.add_done_callback(callback_future_end)
                        futures.add(future)

        # Find the best DAG based on up and down performance
        best_dags = []
        best_perfs = []
        if keep_random_dags == False or (keep_dags_count == 1 and keep_random_dags == True):
            lowests = heapq.nsmallest(keep_dags_count, results, key=lambda x: x[1])
            best_dags, best_perfs = zip(*lowests)
        else:
            absolute_best_dag, absolute_best_perf = min(results, key=lambda x: x[1])
            num_choices = min(keep_dags_count - 1, len(results))
            randoms = random.choices(results, k=num_choices)
            best_dags = (absolute_best_dag, *(r[0] for r in randoms))
            best_perfs = (absolute_best_perf, *(r[1] for r in randoms))

        end_time = time.time()
        #print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        #print("Info: best DAG perf {}".format(best_perf))

        return best_dags, best_perfs

    """
    Plots the topology formed by an adjacency matrix
    """
    def draw_network(self, adj_matrix):
        plt.figure(figsize=(10,10))  # Create a new figure

        G = nx.Graph()
        G.graph["ranksep"] = "1.7"  # Vertical space between ranks
        G.graph["nodesep"] = "0.8"  # Horizontal space between nodes

        # Add nodes
        for i in range(len(adj_matrix)):
            G.add_node(i)

        # Add edges with Link Quality as weight
        for i in range(len(adj_matrix)):
            for j in range(i + 1, len(adj_matrix)):
                if adj_matrix[i][j] > 0:  # There is a link
                    G.add_edge(i, j, weight=adj_matrix[i][j])

        # Draw the network
        pos = graphviz_layout(G, prog="fdp")
        #pos = nx.shell_layout(G)  # Positions for all nodes

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=400)

        # Draw edges
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, pos, edgelist=edges)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")

        # Draw edge labels with Link Quality values
        edge_labels = {(i, j): f"{data['weight']:.2f}" for i, j, data in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Network Topology with Link Quality Values")
        plt.show(block=True)

    """
    Plots a DAG
    """
    def draw_dag(self, G:nx.DiGraph, adj_matrix):
        plt.figure(figsize=(10, 10))
        G.graph["ranksep"] = "1.7"  # Vertical space between ranks
        G.graph["nodesep"] = "0.8"  # Horizontal space between nodes
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True, node_size=400, node_color="lightblue", font_size=15)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{adj_matrix[u][v]:.2f}" for u, v in G.edges()})
        plt.title("Single DAG")
        plt.show(block=True)

    """
    Plots a DAG
    """
    def draw_dag(self, dag:list, adj_matrix):
        G = nx.DiGraph()
        G.add_edges_from(dag)
        plt.figure(figsize=(10, 10))
        G.graph["ranksep"] = "1.7"  # Vertical space between ranks
        G.graph["nodesep"] = "0.8"  # Horizontal space between nodes
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True, node_size=400, node_color="lightblue", font_size=15)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{adj_matrix[u][v]:.2f}" for u, v in G.edges()})
        plt.title("Single DAG")
        plt.show(block=True)

if __name__ == '__main__':
    print("NOP")

