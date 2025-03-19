import sys
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import pydot
import matplotlib.pyplot as plt
import itertools
from itertools import permutations, combinations
import random
from multiprocessing import Pool
import multiprocessing
import time
import datetime
import concurrent.futures
import math
import os
import gc

import psutil
import CythonDAGOperation
import copy
import cProfile
from pathlib import Path
import ctypes

# Define the C struct in ctypes
class Edge(ctypes.Structure):
    _fields_ = [("parent", ctypes.c_int), ("child", ctypes.c_int)]


""" How to use
from DAGDatasetGenerator import DAGDatasetGenerator

generator = DAGDatasetGenerator()

best_dag, best_perf, adj_matrix = generator.run_once(9)
print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
generator.draw_dag(best_dag, adj_matrix)
"""

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
    Runs the simulation for 'count' different random topologies from adjacency matrix of size n by n
    """
    def run(self, n, count):
        start_time = time.time()
        cumulated_time = 0

        filename = "topologies_perf_{}.txt".format(datetime.datetime.now()).replace(":", "_")
        f = open(filename, "x")
        f.close()

        for i in range(0, count):
            item_start_time = time.time()
            best_dag, best_perf, adj_matrix = self.run_once(n)
            f = open(filename, "a")
            f.write(str((adj_matrix, best_dag.edges())))
            f.close()
            cumulated_time += time.time() - item_start_time
            print("------------------------------")
            print("Progression {}/{} minute(s)".format(int(cumulated_time / 60), int(cumulated_time / (i + 1) * count / 60)))
            print("------------------------------")

        print("===========================================")
        print("Generated {} topologies in {} [s]".format(count, time.time() - start_time))



        return

    """
    Run the simulation in parallel
    """
    def run_parallel(self, n, count, max_workers=os.cpu_count()):
        dags_path = Path('dags')
        dags_path.mkdir(exist_ok=True)

        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_once_with_adaptive_steps, n) for i in range(count)}
            for future in concurrent.futures.as_completed(futures):
                best_dag, best_perf, adj_matrix = future.result()
                link_quality_edges = dict()
                for edge in best_dag.edges:
                   link_quality_edges[edge] = float(adj_matrix[edge[0]][edge[1]])
                nx.set_edge_attributes(best_dag, link_quality_edges, 'link_quality')
                futures.remove(future)
                result_name = "topologies_{}".format(datetime.datetime.now()).replace(":", "_")
                filename = Path(result_name + "_best_dag.csv")
                nx.write_edgelist(best_dag, dags_path/filename, delimiter=',')
                filename = Path(result_name + "_adj_matrix.txt")
                np.savetxt(dags_path/filename, np.array(adj_matrix), delimiter=',')

        end_time = time.time()
        print(f"Total runtime : {end_time - start_time}")

    """
    Run the simulation in parallel with pure C simulation
    """
    def run_parallel_pure_c(self, n, count, max_workers=os.cpu_count()):
        dags_path = Path('dags')
        dags_path.mkdir(exist_ok=True)

        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.run_once_with_adaptive_steps_pure_c, n) for i in range(count)}
            for future in concurrent.futures.as_completed(futures):
                best_dag, best_perf, adj_matrix = future.result()
                link_quality_edges = dict()
                for edge in best_dag:
                   link_quality_edges[edge] = float(adj_matrix[edge[0]][edge[1]])
                G = nx.from_numpy_array(adj_matrix, 'link_quality',create_using=nx.MultiDiGraph)
                #G = nx.DiGraph()
                G.add_edges_from(best_dag, link_type='dag')
                nx.set_edge_attributes(G, link_quality_edges, 'link_quality')
                futures.remove(future)
                result_name = "topologies_{}".format(datetime.datetime.now()).replace(":", "_")
                filename = Path(result_name + "_best_dag.csv")
                nx.write_edgelist(G, dags_path/filename, delimiter=',')
                filename = Path(result_name + "_adj_matrix.txt")
                np.savetxt(dags_path/filename, np.array(adj_matrix), delimiter=',')

        end_time = time.time()
        print(f"Total runtime : {end_time - start_time}")

    """
    Run the simulation in parallel with pure C double flux simulation
    """
    def run_parallel_double_flux_pure_c(self, n, count, max_workers=os.cpu_count(), verbose=False):
        dags_path = Path('dags')
        dags_path.mkdir(exist_ok=True)

        generated_count = 0

        start_time = time.time()
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = set()
            for i in range(count):
                future = executor.submit(self.run_once_with_adaptive_steps_double_flux_pure_c, n, verbose)
                future.add_done_callback(callback_future_end)
                futures.add(future)

            for future in concurrent.futures.as_completed(futures):
                best_dag, best_perf, adj_matrix = future.result()
                link_quality_edges = dict()
                for edge in best_dag:
                    edge += (1,) # Use key 1 for the dag and 0 for the topology in the multidigraph
                    link_quality_edges[edge] = {'link_quality': float(adj_matrix[edge[0]][edge[1]]), 'edge_type': 'dag'}

                G = nx.from_numpy_array(adj_matrix, edge_attr='link_quality',create_using=nx.MultiDiGraph)
                G.add_edges_from(link_quality_edges)
                nx.set_edge_attributes(G, link_quality_edges)
                futures.remove(future)
                result_name = "topologies_{}.csv".format(datetime.datetime.now()).replace(":", "_")
                nx.write_edgelist(G, dags_path/Path(result_name), delimiter=',')

                generated_count += 1

                if verbose == False:
                    print(f"Status: [{generated_count} / {count}]", end='\r')

        end_time = time.time()
        print(f"Total runtime : {end_time - start_time}")

    """
    Runs the simulation for one random topology of size n by n
    """
    def run_once(self, n):
        dags = []
        adj_matrix = []
        while (len(dags) == 0):
            # Generate a random adjacency matrix
            adj_matrix, density_factor = self.generate_random_adj_matrix(n)
            print("Density factor = {}".format(density_factor))

            # Get all the possible DAGs within this topology
            dags = self.generate_subset_dags(adj_matrix)
            print(f"Number of DAGs generated: {len(dags)}")

        # Compute the best performing DAG within the topology
        best_dag, best_perf = self.get_best_dag_parallel_up_down(dags, adj_matrix)
        print("best dag is {} perf = {}".format(best_dag.edges, best_perf))

        return best_dag, best_perf, adj_matrix

    def run_once_with_adaptive_steps(self, n):
        dags = []
        adj_matrix = []
        while (len(dags) == 0):
            # Generate a random adjacency matrix
            adj_matrix, density_factor = self.generate_random_adj_matrix(n)
            print("Density factor = {}".format(density_factor))

            # Get all the possible DAGs within this topology
            dags = self.generate_subset_dags(adj_matrix)
            print(f"Number of DAGs generated: {len(dags)}")

        # Compute the best performing DAG within the topology
        best_dag, best_perf = self.get_best_dag_parallel_with_adaptative_steps(dags, adj_matrix)
        print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
        np.set_printoptions(formatter={'all': lambda x: "{:.4g},".format(x)})
        print(adj_matrix)

        return best_dag, best_perf, adj_matrix

    def run_once_with_adaptive_steps_pure_c(self, n):
        dags = []
        adj_matrix = []
        while (len(dags) == 0):
            # Generate a random adjacency matrix
            adj_matrix, density_factor = self.generate_random_adj_matrix(n)
            print("Density factor = {}".format(density_factor))

            # Get all the possible DAGs within this topology
            dags = self.generate_subset_dags_pure_c(adj_matrix)
            print(f"Number of DAGs generated: {len(dags)}")

        # Compute the best performing DAG within the topology
        best_dag, best_perf = self.get_best_dag_parallel_with_adaptative_steps_pure_c(dags, adj_matrix)
        print("best dag is {} perf = {}".format(best_dag, best_perf))
        np.set_printoptions(formatter={'all': lambda x: "{:.4g},".format(x)})
        print(adj_matrix)

        return best_dag, best_perf, adj_matrix

    def run_once_with_adaptive_steps_double_flux_pure_c(self, n, verbose=False):
        dags = []
        adj_matrix = []
        while (len(dags) == 0):
            # Generate a random adjacency matrix
            adj_matrix, density_factor = self.generate_random_adj_matrix(n)
            if verbose:
                print("Density factor = {}".format(density_factor))
            # Get all the possible DAGs within this topology
            dags = self.generate_subset_dags_pure_c(adj_matrix)
            if verbose:
                print(f"Number of DAGs generated: {len(dags)}")

        # Compute the best performing DAG within the topology
        best_dag, best_perf = self.get_best_dag_parallel_with_adaptative_steps_double_flux_pure_c(dags, adj_matrix, verbose=verbose)
        if verbose:
            print("best dag is {} perf = {}".format(best_dag, best_perf))
            np.set_printoptions(formatter={'all': lambda x: "{:.4g},".format(x)})
            print(adj_matrix)

        return best_dag, best_perf, adj_matrix

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

    def generate_dags(self, adj_matrix):
        nodes = list(range(len(adj_matrix)))
        all_edges = [(i, j) for i in range(len(adj_matrix)) for j in range(i+1, len(adj_matrix)) if adj_matrix[i][j] > 0]
        num_nodes = len(nodes)
        all_possible_trees = []

        def dfs_tree(adj_matrix, visited, current_node, tree_edges):
            visited[current_node] = True
            for neighbor in range(len(adj_matrix)):
                if adj_matrix[current_node][neighbor] > 0 and not visited[neighbor]:
                    tree_edges.append((current_node, neighbor))
                    dfs_tree(adj_matrix, visited, neighbor, tree_edges)

        # Generate all combinations of edges that can form a tree (n-1 edges)
        for edges in combinations(all_edges, num_nodes - 1):
            tree_matrix = np.zeros_like(adj_matrix)
            for edge in edges:
                tree_matrix[edge[0]][edge[1]] = 1
                tree_matrix[edge[1]][edge[0]] = 1
            # Check if the selected edges form a connected tree
            visited = [False] * num_nodes
            tree_edges = []
            dfs_tree(tree_matrix, visited, 0, tree_edges)
            if len(tree_edges) == num_nodes - 1 and all(visited):
                all_possible_trees.append(tree_edges)

        digraphs = []
        for tree_edges in all_possible_trees:
            G = nx.DiGraph()
            G.add_edges_from(tree_edges)
            digraphs.append(G)
        return digraphs

    def generate_subset_dags(self, adj_matrix, test_mode=False):
        #start_time = time.time()

        nodes = list(range(len(adj_matrix)))
        all_edges = [(i, j) for i in range(len(adj_matrix)) for j in range(i+1, len(adj_matrix)) if adj_matrix[i][j] > 0]
        num_nodes = len(nodes)
        all_possible_trees = []
        def dfs_tree(adj_matrix, visited, current_node, tree_edges):
            visited[current_node] = True
            for neighbor in range(len(adj_matrix)):
                if adj_matrix[current_node][neighbor] > 0 and not visited[neighbor]:
                    tree_edges.append((current_node, neighbor))
                    dfs_tree(adj_matrix, visited, neighbor, tree_edges)
        # Generate subset combinations of edges that can form a tree (n-1 edges)
        rng = np.random.default_rng()
        # all_comb = list(combinations(all_edges, num_nodes - 1)) # Crash with big combination list
        n_edges = len(all_edges)
        k_nodes = num_nodes - 1
        total_combinations = math.factorial(n_edges)/(math.factorial(k_nodes)*math.factorial(n_edges - k_nodes))
        comb_iter = combinations(all_edges, k_nodes)
        #end_of_combination_computation = time.time()
        #print(f"Took {int((end_of_combination_computation - start_time) * 1e6)} [us] to compute combinations")
        print(f"Total number of combinations: {total_combinations}")

        start_of_tree_computation = time.time()
        for edges in comb_iter:
            tree_matrix = np.zeros_like(adj_matrix)
            for edge in edges:
                tree_matrix[edge[0]][edge[1]] = 1
                tree_matrix[edge[1]][edge[0]] = 1
            # Check if the selected edges form a connected tree
            visited = [False] * num_nodes
            tree_edges = []
            dfs_tree(tree_matrix, visited, 0, tree_edges)
            if len(tree_edges) == num_nodes - 1 and all(visited):
                all_possible_trees.append(tree_edges)

            # Progress with iterator by a random jump (scale by the number of nodes and number of edges)
            if test_mode == False:
                total_skip = rng.integers(k_nodes * n_edges)
            else:
                # Deterministic behaviour
                total_skip = int((k_nodes * n_edges) / 2)
            #print(f"Element skipped : {total_skip}")
            for _ in range(total_skip):
                if not len(next(comb_iter, [])):
                    break

        #end_of_tree_computation = time.time()
        #print(f"Took {int((end_of_tree_computation - start_of_tree_computation) * 1e6)} [us] to generate all dags")
        #start_of_format_conversion = time.time()

        #end_of_format_conversion = time.time()
        #print(f"Took {int((end_of_format_conversion - start_of_format_conversion) * 1e6)} [us] to convert format")
        #print(f"Total time {int((end_of_format_conversion - start_time) * 1e6)} [us]")

        return all_possible_trees

    # skip_factor=0 -> automatic determination
    def generate_subset_dags_pure_c(self, adj_matrix, skip_factor=0, test_mode=False, no_skip=False):
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

    def draw_dag(self, G, adj_matrix):
        plt.figure(figsize=(10, 10))
        G.graph["ranksep"] = "1.7"  # Vertical space between ranks
        G.graph["nodesep"] = "0.8"  # Horizontal space between nodes
        pos = graphviz_layout(G, prog="dot")
        nx.draw(G, pos, with_labels=True, node_size=400, node_color="lightblue", font_size=15)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{adj_matrix[u][v]:.2f}" for u, v in G.edges()})
        plt.title("Single DAG")
        plt.show(block=False)

    def draw_all_dags(self, dags, adj_matrix):
        for i, G in enumerate(dags):
            plt.figure(figsize=(10, 10))
            G.graph["ranksep"] = "1.7"  # Vertical space between ranks
            G.graph["nodesep"] = "0.8"  # Horizontal space between nodes
            pos = graphviz_layout(G, prog="dot")
            nx.draw(G, pos, with_labels=True, node_size=400, node_color="lightblue", font_size=15)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{adj_matrix[u][v]:.2f}" for u, v in G.edges()})
            plt.title(f"DAG {i+1}")
            plt.show(block=False)

    """
    Uplink global bandwidth measurement.

    This method computes the number of iterations it takes to empty the packet queue of each node within the network.

    Args:
        G (DiGraph): The DAG
        adj_matrix (ndarray): The adjacency matrix describing the topology
        epoch_len (int): Number of time to run the simulation to average result. This is required because there is a basic CSMA/CA mechanism which introduces some randomness.
        packets_per_node (int): Initial packet count in each node queue
        max_steps (int): Early stop condition. This helps not to run simulation uselessly if the number of steps is above a provided threshold. This can be dynamically updated when more and more DAGs have been evaluated to spare time.
    """
    def evaluate_dag_performance_up(self, dag, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
        G = nx.DiGraph()
        G.add_edges_from(dag)
        n = len(G.nodes)

        packets = {node: packets_per_node for node in G.nodes}
        transmit_intent = {node: True for node in G.nodes}  # Transmission intentions
        busy = {node: False for node in G.nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = packets_per_node
                    busy[i] = False
                    transmit_intent[i] = True

            steps = 0
            while any(packets[node] > 0 for node in G.nodes):
                steps += 1

                transmit_intent = {node: packets[node] > 0 for node in G.nodes}  # Transmission intentions

                # Process packet transmission for all nodes
                r = list(range(0, len(G.nodes)))
                np.random.shuffle(r)
                for i in r:
                    parent = list(G.nodes)[i]
                    if busy[parent] == True:
                        continue
                    children = list(G.successors(parent))
                    transmitting_child = -1
                    children_transmit_intents = [child for child in children if transmit_intent[child] == True]
                    if children_transmit_intents != []:
                        transmitting_child = random.choice(children_transmit_intents)

                    if transmitting_child >= 0:
                        # Determine if transmission is successful based on Link Quality
                        link_quality = adj_matrix[parent][transmitting_child]  # Get Link Quality value for the link
                        transmission_success = random.random() <= link_quality

                        if transmission_success:
                            packets[parent] += 1
                            packets[transmitting_child] -= 1
                            transmit_intent[transmitting_child] = False

                        busy[transmitting_child] = True


                # Reset the transmitting and receiving status for the next step
                busy = {node: False for node in G.nodes}
                # Root node never holds an packet
                packets[0] = 0

                if max_steps != -1 and steps > max_steps:
                    early_stop = True
                    break

            avg_steps += steps

            if early_stop == True:
                break

        if not early_stop:
            return avg_steps / epoch_len
        else:
            return max_steps


    """
    Downlink global bandwidth measurement.

    This method computes the number of iterations it takes to fill the packet queue of each node within the network to a given count.

    Args:
        G (DiGraph): The DAG
        adj_matrix (ndarray): The adjacency matrix describing the topology
        epoch_len (int): Number of time to run the simulation to average result. This is required because there is a basic CSMA/CA mechanism which introduces some randomness.
        packets_per_node (int): Final packet count required in each node queue to end simulation
        max_steps (int): Early stop condition. This helps not to run simulation uselessly if the number of steps is above a provided threshold. This can be dynamically updated when more and more DAGs have been evaluated to spare time.
    """
    def evaluate_dag_performance_down(self, dag, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
        G = nx.DiGraph()
        G.add_edges_from(dag)
        n = len(G.nodes)

        packets = {node: 0 for node in G.nodes}
        busy = {node: False for node in G.nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = 0
                    busy[i] = False

            steps = 0
            while any(packets[node] < packets_per_node for node in G.nodes):
                steps += 1

                # Root node inserts a packet into the network if it's not transmitting
                if packets[0] < packets_per_node and not busy[0]:
                    packets[0] += 1

                # Process packet transmission for all nodes
                r = list(range(0, len(G.nodes)))
                np.random.shuffle(r)
                for i in r:
                    node = list(G.nodes)[i]
                    if packets[node] > 0 and not busy[node]:  # Node has packets to send and is not already transmitting
                        children = list(G.successors(node))
                        if children:
                            # Choose a child randomly to try to send the packet to
                            child = random.choice(children)

                            if not busy[child]:  # Check if the child is not currently transmitting
                                # Determine if transmission is successful based on Link Quality
                                link_quality = adj_matrix[node][child]  # Get Link Quality value for the link
                                transmission_success = random.random() <= link_quality

                                if transmission_success and packets[child] < packets_per_node:
                                    packets[child] += 1
                                    packets[node] -= 1
                                    busy[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision

                # Reset the transmitting status for the next step
                busy = {node: False for node in G.nodes}

                if max_steps != -1 and steps > max_steps:
                    early_stop = True
                    break

            avg_steps += steps

            if early_stop == True:
                break

        if not early_stop:
            return avg_steps / epoch_len
        else:
            return max_steps

    """
    This method combines the UP and DOWN simulations into a single simulation
    """
    def evaluate_dag_performance_up_down(self, G, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
        n = len(G.nodes)

        packets_up = {node: packets_per_node for node in G.nodes}
        packets_down = {node: 0 for node in G.nodes}
        transmit_intent_up = {node: True for node in G.nodes}  # Transmission intentions
        busy = {node: False for node in G.nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets_up)):
                    # This is done to optimize execution time
                    packets_up[i] = packets_per_node
                    packets_down[i] = 0
                    busy[i] = False
                    transmit_intent_up[i] = True

            steps = 0

            up_finished = False
            down_finished = False
            while not up_finished or not down_finished:
                steps += 1

                # Root node Root node has a new packet to insert into the network. Does not send it yet.
                if packets_down[0] < packets_per_node:
                    packets_down[0] += 1

                transmit_intent_up = {node: packets_up[node] > 0 for node in G.nodes}  # Transmission intentions

                # Process packet transmission for all nodes
                r = list(range(0, len(G.nodes)))
                np.random.shuffle(r)
                for i in r:
                    # Decide whether the current node wants to transmit UP or DOWN
                    up_not_down = False
                    if not up_finished and not down_finished:
                        up_not_down = True if random.random() >= 0.5 else False
                    elif up_finished:
                        up_not_down = False     # UP operation done -> only DOWN has to run
                    elif down_finished:
                        up_not_down = True      # DOWN operation done -> only UP has to run

                    if up_not_down == True: # UP
                        parent = list(G.nodes)[i]
                        if busy[parent] == True:
                            continue
                        children = list(G.successors(parent))
                        transmitting_child = -1
                        children_transmit_intents = [child for child in children if transmit_intent_up[child] == True]
                        if children_transmit_intents != []:
                            transmitting_child = random.choice(children_transmit_intents)

                        if transmitting_child >= 0:
                            # Determine if transmission is successful based on Link Quality
                            link_quality = adj_matrix[parent][transmitting_child]  # Get Link Quality value for the link
                            transmission_success = random.random() <= link_quality

                            if transmission_success:
                                if packets_up[parent] < packets_per_node:  # simulate full queue
                                    packets_up[parent] += 1
                                    packets_up[transmitting_child] -= 1
                                transmit_intent_up[transmitting_child] = False

                            busy[transmitting_child] = True
                    else:   # DOWN
                        node = list(G.nodes)[i]
                        if packets_down[node] > 0 and not busy[node]:  # Node has packets to send and is not already transmitting
                            children = list(G.successors(node))
                            if children:
                                # Choose a child randomly to try to send the packet to
                                child = random.choice(children)

                                if not busy[child]:  # Check if the child is not currently transmitting
                                    # Determine if transmission is successful based on Link Quality
                                    link_quality = adj_matrix[node][child]  # Get Link Quality value for the link
                                    transmission_success = random.random() <= link_quality

                                    if transmission_success:
                                        if packets_down[child] < packets_per_node:  # simulate full queue
                                            packets_down[child] += 1
                                            packets_down[node] -= 1
                                        busy[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision

                                    busy[parent] = True # Mark the parent as transmitting. It is also the case when transmission success is false because it simulates a collision by the fact the child is busy.


                # Reset the transmitting and receiving status for the next step
                busy = {node: False for node in G.nodes}
                # Root node never holds an packet
                packets_up[0] = 0

                if max_steps != -1 and steps > max_steps:
                    early_stop = True
                    break

                # Update end conditions
                up_finished = not any(packets_up[node] > 0 for node in G.nodes)
                down_finished = not any(packets_down[node] < packets_per_node for node in G.nodes)

            avg_steps += steps

            if early_stop == True:
                break

        if not early_stop:
            return int(avg_steps / epoch_len)
        else:
            return max_steps

    def evaluate_dag_performance_combined(self, eval_up, eval_down, G, adj_matrix, epoch_len=1, packets_per_node=15, max_steps_up=-1, max_steps_down=-1):
        perf_up = eval_up(G, adj_matrix, max_steps=max_steps_up)
        perf_down = eval_down(G, adj_matrix, max_steps=max_steps_down)
        return G, perf_up, perf_down

    def evaluate_dag_performance_combined_pure_c(self, dag, adj_matrix, epoch_len=1, packets_per_node=15, max_steps_up=-1, max_steps_down=-1):
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

        perf_up = int(lib.evaluate_dag_performance_up(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps_up))
        perf_down = int(lib.evaluate_dag_performance_down(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps_down))
        return dag, perf_up, perf_down

    def evaluate_dag_performance_double_flux_pure_c(self, dag, adj_matrix, epoch_len=1, packets_per_node=15, max_steps=-1):
        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_up_down.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_up_down.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        c_dag = (Edge * len(dag))(
            *[Edge(parent, child) for parent, child in dag]
        )
        perf = int(lib.evaluate_dag_performance_up_down(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, max_steps))
        return dag, perf

    def get_best_dag(self, dags, adj_matrix, eval_func):
        start_time = time.time()

        best_perf = eval_func(dags[0], adj_matrix)
        best_dag = dags[0]

        for i, dag in enumerate(dags[1:]):
            print("{}".format(i), end='\r')
            perf = self.eval_func(dag, adj_matrix)
            if perf < best_perf:
                best_perf = perf
                best_dag = dag

        end_time = time.time()
        print("Computing best DAG took {}".format(end_time - start_time))

        return (best_dag, best_perf)

    def evaluate_dag(self, index, dag, adj_matrix, eval_func, max_steps=-1):
        if index % 100 == 0:
            print(f'\r{index}', end='', flush=True)
        return dag, eval_func(dag, adj_matrix, max_steps=max_steps)

    def get_best_dag_parallel(self, dags, adj_matrix, eval_func):
        start_time = time.time()

        # Prepare the arguments as a list of tuples
        args = [(i, dag, adj_matrix, self.eval_func, -1) for i, dag in enumerate(dags)]

        # Use a multiprocessing pool to parallelize the evaluation
        with multiprocessing.Pool() as pool:
            results = pool.starmap(self.evaluate_dag, args)

        # Find the best DAG based on performance
        best_dag, best_perf = min(results, key=lambda x: x[1])

        end_time = time.time()
        print("Computing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        return best_dag, best_perf

    def cython_get_best_dag_parallel_up_down(self, dags, adj_matrix):
        start_time = time.time()

        dags_g = []
        for dag in dags:
            G = nx.DiGraph()
            G.add_edges_from(dag)
            dags_g.append(G)

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                dag_g = random.choice(dags_g)
                max_steps_up = min(CythonDAGOperation.cython_evaluate_dag_performance_up(dag_g, adj_matrix), max_steps_up)
                max_steps_down = min(CythonDAGOperation.cython_evaluate_dag_performance_down(dag_g, adj_matrix), max_steps_down)
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        # Prepare the arguments as a list of tuples
        args_up = [(i, dag, adj_matrix, CythonDAGOperation.cython_evaluate_dag_performance_up, max_steps_up) for i, dag in enumerate(dags_g)]
        args_down = [(i, dag, adj_matrix, CythonDAGOperation.cython_evaluate_dag_performance_down, max_steps_down) for i, dag in enumerate(dags_g)]

        # Use a multiprocessing pool to parallelize the evaluation
        with multiprocessing.Pool() as pool:
            print("\nComputing UP\n")
            up_results = pool.starmap(self.evaluate_dag, args_up, chunksize=100)
            print("\nComputing Down\n")
            down_results = pool.starmap(self.evaluate_dag, args_down, chunksize=100)

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results])
        down_results_np = np.array([item[1] for item in down_results])

        # normalize and combine
        up_results_np /= np.max(up_results_np)
        down_results_np /= np.max(down_results_np)
        combined_results_np = up_results_np + down_results_np

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        #print(len(up_results))
        #print(len(down_results))
        #print(len(combined_results))

        best_dag_up, best_perf_up = min(up_results, key=lambda x: x[1])
        best_dag_down, best_perf_down = min(down_results, key=lambda x: x[1])

        sorted_combined_results = combined_results
        sorted_combined_results.sort(key=lambda x: x[1], reverse=True)
        best_dag_up_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_up))
        best_dag_down_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_down))

        # Find the best DAG based on up and down performance
        best_dag, best_perf = min(combined_results, key=lambda x: x[1])

        end_time = time.time()
        print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        print("Info: best DAG UP rank = {}/{} (perf {}) and best DAG DOWN rank = {}/{} (perf {}) compared to overall best score".format(best_dag_up_overall_score, len(combined_results), best_perf_up, best_dag_down_overall_score, len(combined_results), best_perf_down))

        return best_dag, best_perf


    def get_best_dag_parallel_up_down(self, dags, adj_matrix):
        start_time = time.time()

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                max_steps_up = min(self.evaluate_dag_performance_up(random.choice(dags), adj_matrix), max_steps_up)
                max_steps_down = min(self.evaluate_dag_performance_down(random.choice(dags), adj_matrix), max_steps_down)
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        # Prepare the arguments as a list of tuples
        args_up = [(i, dag, adj_matrix, self.evaluate_dag_performance_up, max_steps_up) for i, dag in enumerate(dags)]
        args_down = [(i, dag, adj_matrix, self.evaluate_dag_performance_down, max_steps_down) for i, dag in enumerate(dags)]

        # Use a multiprocessing pool to parallelize the evaluation
        with multiprocessing.Pool() as pool:
            print("\nComputing UP\n")
            up_results = pool.starmap(self.evaluate_dag, args_up, chunksize=100)
            print("\nComputing Down\n")
            down_results = pool.starmap(self.evaluate_dag, args_down, chunksize=100)

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results])
        down_results_np = np.array([item[1] for item in down_results])

        # normalize and combine
        up_results_np /= np.max(up_results_np)
        down_results_np /= np.max(down_results_np)
        combined_results_np = up_results_np + down_results_np

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        #print(len(up_results))
        #print(len(down_results))
        #print(len(combined_results))

        best_dag_up, best_perf_up = min(up_results, key=lambda x: x[1])
        best_dag_down, best_perf_down = min(down_results, key=lambda x: x[1])

        sorted_combined_results = combined_results
        sorted_combined_results.sort(key=lambda x: x[1], reverse=True)
        best_dag_up_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_up))
        best_dag_down_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_down))

        # Find the best DAG based on up and down performance
        best_dag, best_perf = min(combined_results, key=lambda x: x[1])

        end_time = time.time()
        print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        print("Info: best DAG UP rank = {}/{} (perf {}) and best DAG DOWN rank = {}/{} (perf {}) compared to overall best score".format(best_dag_up_overall_score, len(combined_results), best_perf_up, best_dag_down_overall_score, len(combined_results), best_perf_down))

        return best_dag, best_perf

    """
    def get_best_dag_parallel_combined(self, dags, adj_matrix):
        start_time = time.time()

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps = 0
        iter = 100 if len(dags) >= 100 else len(dags)
        for _ in range(0, iter):
            max_steps += self.evaluate_dag_performance_combined(random.choice(dags), adj_matrix)
        max_steps = int(max_steps / iter)
        print("Max steps = " + str(max_steps))

        # Prepare the arguments as a list of tuples
        args = [(i, dag, adj_matrix, self.evaluate_dag_performance_combined, max_steps * 1.1) for i, dag in enumerate(dags)]

        # Use a multiprocessing pool to parallelize the evaluation
        with multiprocessing.Pool() as pool:
            results = pool.starmap(self.evaluate_dag, args, chunksize=200)

        # Find the best DAG based on performance
        best_dag, best_perf = min(results, key=lambda x: x[1])

        end_time = time.time()
        print("Computing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        return best_dag, best_perf
    """

    def get_best_dag_parallel_with_adaptative_steps(self, dags, adj_matrix, max_workers=os.cpu_count(), delta_threshold=0.8, reduce_ratio = 0.2, margin_max_step = 1.1):
        start_time = time.time()
        dags_g = []
        for dag in dags:
            G = nx.DiGraph()
            G.add_edges_from(dag)
            dags_g.append(G)

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                max_steps_up = min(CythonDAGOperation.cython_evaluate_dag_performance_up(random.choice(dags_g), adj_matrix), max_steps_up)
                max_steps_down = min(CythonDAGOperation.cython_evaluate_dag_performance_down(random.choice(dags_g), adj_matrix), max_steps_down)
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        up_results = []
        down_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_dag_performance_combined,
                    CythonDAGOperation.cython_evaluate_dag_performance_up,
                    CythonDAGOperation.cython_evaluate_dag_performance_down,
                    dags_g.pop() if dags_g else None,  # Prevent pop() from failing
                    adj_matrix,
                    max_steps_up=max_steps_up,
                    max_steps_down=max_steps_down)
                for _ in range(min(max_workers, len(dags_g)))  # Use len(dags_g)
            }
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
                    if dags_g:  # Check if there are remaining DAGs
                        futures.add(executor.submit(
                            self.evaluate_dag_performance_combined,
                            CythonDAGOperation.cython_evaluate_dag_performance_up,
                            CythonDAGOperation.cython_evaluate_dag_performance_down,
                            dags_g.pop(),
                            adj_matrix,
                            max_steps_up=int(max_steps_up * margin_max_step),
                            max_steps_down=int(max_steps_down * margin_max_step))
                        )

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results]).astype(float)
        down_results_np = np.array([item[1] for item in down_results]).astype(float)

        # normalize and combine
        up_results_np /= np.max(up_results_np)
        down_results_np /= np.max(down_results_np)
        combined_results_np = up_results_np + down_results_np

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        #print(len(up_results_np))
        #print(len(down_results_np))
        #print(len(combined_results))

        best_dag_up, best_perf_up = min(up_results, key=lambda x: x[1])
        best_dag_down, best_perf_down = min(down_results, key=lambda x: x[1])

        sorted_combined_results = combined_results
        sorted_combined_results.sort(key=lambda x: x[1], reverse=True)
        best_dag_up_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_up))
        best_dag_down_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_down))

        # Find the best DAG based on up and down performance
        best_dag, best_perf = min(combined_results, key=lambda x: x[1])

        end_time = time.time()
        print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        print("Info: best DAG UP rank = {}/{} (perf {}) and best DAG DOWN rank = {}/{} (perf {}) compared to overall best score".format(best_dag_up_overall_score, len(combined_results), best_perf_up, best_dag_down_overall_score, len(combined_results), best_perf_down))

        return best_dag, best_perf

    def get_best_dag_parallel_with_adaptative_steps_pure_c(self, dags, adj_matrix, max_workers=os.cpu_count(), delta_threshold=0.8, reduce_ratio = 0.2, margin_max_step = 1.1):
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
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        up_results = []
        down_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_dag_performance_combined_pure_c,
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
                            self.evaluate_dag_performance_combined_pure_c,
                            dags.pop(),
                            adj_matrix,
                            max_steps_up=int(max_steps_up * margin_max_step),
                            max_steps_down=int(max_steps_down * margin_max_step))
                        )

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results]).astype(float)
        down_results_np = np.array([item[1] for item in down_results]).astype(float)

        # normalize and combine
        up_results_np /= np.max(up_results_np)
        down_results_np /= np.max(down_results_np)
        combined_results_np = up_results_np + down_results_np

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        #print(len(up_results_np))
        #print(len(down_results_np))
        #print(len(combined_results))

        best_dag_up, best_perf_up = min(up_results, key=lambda x: x[1])
        best_dag_down, best_perf_down = min(down_results, key=lambda x: x[1])

        sorted_combined_results = combined_results
        sorted_combined_results.sort(key=lambda x: x[1], reverse=True)
        best_dag_up_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_up))
        best_dag_down_overall_score = [list(item[0]) for item in sorted_combined_results].index(list(best_dag_down))

        # Find the best DAG based on up and down performance
        best_dag, best_perf = min(combined_results, key=lambda x: x[1])

        end_time = time.time()
        print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        print("Info: best DAG UP rank = {}/{} (perf {}) and best DAG DOWN rank = {}/{} (perf {}) compared to overall best score".format(best_dag_up_overall_score, len(combined_results), best_perf_up, best_dag_down_overall_score, len(combined_results), best_perf_down))

        return best_dag, best_perf

    def get_best_dag_parallel_with_adaptative_steps_double_flux_pure_c(self, dags, adj_matrix, max_workers=os.cpu_count(), delta_threshold=0.8, reduce_ratio = 0.2, margin_max_step = 1.1, verbose=False):
        start_time = time.time()

        dll_name = "CDAGOperation/libCDAGOperation.so"
        dllabspath = os.path.dirname(os.path.abspath(__file__)) + os.path.sep + dll_name
        lib = ctypes.CDLL(dllabspath)

        lib.evaluate_dag_performance_up_down.argtypes = (ctypes.POINTER(Edge), ctypes.c_int, ctypes.POINTER(ctypes.POINTER(ctypes.c_float)), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        lib.evaluate_dag_performance_up_down.restype = ctypes.c_int

        CMatrixType = ctypes.POINTER(ctypes.c_float) * len(adj_matrix)
        adj_matrix_c = CMatrixType(
            *[ctypes.cast((ctypes.c_float * len(row))(*row), ctypes.POINTER(ctypes.c_float)) for row in adj_matrix]
        )

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                dag = random.choice(dags)
                c_dag = (Edge * len(dag))(
                    *[Edge(parent, child) for parent, child in dag]
                )
                max_steps = int(min(lib.evaluate_dag_performance_up_down(c_dag, len(dag), adj_matrix_c, len(adj_matrix[0]), 2, 15, -1), max_steps))
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
                    self.evaluate_dag_performance_double_flux_pure_c,
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
                        future = executor.submit(
                            self.evaluate_dag_performance_double_flux_pure_c,
                            dags.pop(),
                            adj_matrix,
                            max_steps=int(max_steps * margin_max_step))
                        future.add_done_callback(callback_future_end)
                        futures.add(future)
                    # Print state for all futures
                    if verbose:
                        print('---------------------------------------------------')
                        for future in futures:
                            print(f"ID: {id(future)}, state: {future._state}")
                        print('---------------------------------------------------')

        # Find the best DAG based on up and down performance
        best_dag, best_perf = min(results, key=lambda x: x[1])

        end_time = time.time()
        #print("\nComputing best DAG in parallel took {:.2f} seconds".format(end_time - start_time))

        #print("Info: best DAG perf {}".format(best_perf))

        return best_dag, best_perf


if __name__ == '__main__':
    generator = DAGDatasetGenerator()
    #generator.run_once_with_adaptive_steps(15)



    # Test

    adj_matrix = [[0.00, 0.40, 0.19, 0.00, 0.00, 0.00, 0.10],
 [0.40, 0.00, 0.12, 0.00, 0.58, 0.00, 0.00],
 [0.19, 0.12, 0.00, 0.00, 0.00, 0.31, 0.00],
 [0.00, 0.00, 0.00, 0.09, 0.15, 0.02, 0.00],
 [0.00, 0.58, 0.00, 0.15, 0.00, 0.22, 0.00],
 [0.00, 0.00, 0.31, 0.02, 0.22, 0.00, 0.19],
 [0.10, 0.00, 0.00, 0.00, 0.00, 0.19, 0.00]]



    #adj_matrix = np.loadtxt("dags/topologies_2024-11-06 16_29_44.968765_adj_matrix.txt", delimiter=',').tolist()
    #generator.draw_network(adj_matrix)
    #adj_matrix = np.loadtxt("dags/topologies_2024-11-06 16_48_43.661391_adj_matrix.txt", delimiter=',').tolist()
    #generator.draw_network(adj_matrix)


    # Generate a random adjacency matrix
    #adj_matrix, density_factor = generator.generate_random_adj_matrix(10)
    #print(np.array2string(adj_matrix, separator=', ', formatter={'float_kind':lambda x: f"{x:.2f}"}, suppress_small=True))
    #print("Density factor = {}".format(density_factor))


    # Get all the possible DAGs within this topology
    #dags_python = generator.generate_subset_dags(adj_matrix, True)
    dags_pure_c = generator.generate_subset_dags_pure_c(adj_matrix, True)


    #print(f"Number of DAGs generated using Python: {len(dags_python)}")
    #print(type(dags_python))
    #for digraph in dags_python:
    #    print(digraph.edges())
    print(f"Number of DAGs generated using Pure C: {len(dags_pure_c)}")
    #print(type(dags_pure_c))
    #for digraph in dags_pure_c:
    #    print(digraph.edges())

    perf_up = generator.evaluate_dag_performance_up(dags_pure_c[0], adj_matrix, 2, 15, -1)
    perf_down = generator.evaluate_dag_performance_down(dags_pure_c[0], adj_matrix, 2, 15, -1)
    perf_up_down = generator.evaluate_dag_performance_up_down(dags_pure_c[0], adj_matrix, 2, 15, -1)

    print(f"Perf UP = {perf_up}, perf DOWN = {perf_down}, perf UP/DOWN = {perf_up_down}")

