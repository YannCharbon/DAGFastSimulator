import numpy as np
import networkx as nx
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
import CythonDAGOperation 
import copy
import cProfile
from pathlib import Path
""" How to use
from DAGDatasetGenerator import DAGDatasetGenerator

generator = DAGDatasetGenerator()

best_dag, best_perf, adj_matrix = generator.run_once(9)
print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
generator.draw_dag(best_dag, adj_matrix)
"""
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
                rssi_edges = dict()
                for edge in best_dag.edges:
                   rssi_edges[edge] = adj_matrix[edge[0]][edge[1]]
                nx.set_edge_attributes(best_dag, rssi_edges, 'rssi')
                futures.remove(future)
                filename = Path("topologies_perf_{}.csv".format(datetime.datetime.now()).replace(":", "_"))
                nx.write_edgelist(best_dag, dags_path/filename, delimiter=',')

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

        return best_dag, best_perf, adj_matrix

    def generate_random_adj_matrix(self, n):
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

        # RSSI with best quality = 1.0 No connection = 0.0
        # the ' - 2 * np.random.rand()' controls the density of the interconnections
        rng = np.random.default_rng() # Required in multiprocessing to avoid having same random values in all processes 
        density_factor = rng.random()
        a = np.maximum(rng.random((n, n)) * 2 - 1 - 1 * (1 - density_factor), np.zeros((n, n)))
        a = symmetrize(a)

        while not check_integrity(a):
            density_factor = rng.random()
            a = np.maximum(rng.random((n, n)) * 2 - 1 - 1 * (1 - density_factor), np.zeros((n, n)))
            a = symmetrize(a)

        a = limit_neighbors(a)

        return a, density_factor

    def draw_network(self, adj_matrix):
        plt.figure()  # Create a new figure

        G = nx.Graph()

        # Add nodes
        for i in range(len(adj_matrix)):
            G.add_node(i)

        # Add edges with RSSI as weight
        for i in range(len(adj_matrix)):
            for j in range(i + 1, len(adj_matrix)):
                if adj_matrix[i][j] > 0:  # There is a link
                    G.add_edge(i, j, weight=adj_matrix[i][j])

        # Draw the network
        pos = nx.shell_layout(G)  # Positions for all nodes

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700)

        # Draw edges
        edges = G.edges(data=True)
        nx.draw_networkx_edges(G, pos, edgelist=edges)

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=20, font_family="sans-serif")

        # Draw edge labels with RSSI values
        edge_labels = {(i, j): f"{data['weight']:.2f}" for i, j, data in edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Network Topology with RSSI Values")
        plt.show(block=False)

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

    def generate_subset_dags(self, adj_matrix):
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
        print(f"Total number of combinations: {total_combinations}")
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
            total_skip = rng.integers(math.ceil(k_nodes * n_edges))
            #print(f"Element skipped : {total_skip}")
            for _ in range(total_skip):
                if not len(next(comb_iter, [])):
                    break

        digraphs = []
        for tree_edges in all_possible_trees:
            G = nx.DiGraph()
            G.add_edges_from(tree_edges)
            digraphs.append(G)
        return digraphs

    def draw_dag(self, G, adj_matrix):
        plt.figure()
        pos = nx.shell_layout(G)
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=15)
        nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{adj_matrix[u][v]:.2f}" for u, v in G.edges()})
        plt.title("Single DAG")
        plt.show(block=False)

    def draw_all_dags(self, dags, adj_matrix):
        for i, G in enumerate(dags):
            plt.figure()
            pos = nx.shell_layout(G)
            nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", font_size=15)
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
    def evaluate_dag_performance_up(self, G, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
        n = len(G.nodes)

        packets = {node: packets_per_node for node in G.nodes}
        transmit_intent = {node: True for node in G.nodes}  # Transmission intentions
        transmitting = {node: False for node in G.nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = 0
                    transmitting[i] = False
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
                    if transmitting[parent] == True:
                        continue
                    children = list(G.successors(parent))
                    transmitting_child = -1
                    children_transmit_intents = [child for child in children if transmit_intent[child] == True]
                    if children_transmit_intents != []:
                        transmitting_child = random.choice(children_transmit_intents)

                    if transmitting_child >= 0:
                        # Determine if transmission is successful based on RSSI
                        rssi = adj_matrix[parent][transmitting_child]  # Get RSSI value for the link
                        transmission_success = random.random() <= rssi

                        if transmission_success:
                            packets[parent] += 1
                            packets[transmitting_child] -= 1

                        transmitting[transmitting_child] = True


                # Reset the transmitting and receiving status for the next step
                transmitting = {node: False for node in G.nodes}
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
    def evaluate_dag_performance_down(self, G, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
        n = len(G.nodes)

        packets = {node: 0 for node in G.nodes}
        transmitting = {node: False for node in G.nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = 0
                    transmitting[i] = False

            steps = 0
            while any(packets[node] < packets_per_node for node in G.nodes):
                steps += 1

                # Root node inserts a packet into the network if it's not transmitting
                if packets[0] < packets_per_node and not transmitting[0]:
                    packets[0] += 1

                # Process packet transmission for all nodes
                r = list(range(0, len(G.nodes)))
                np.random.shuffle(r)
                for i in r:
                    node = list(G.nodes)[i]
                    if packets[node] > 0 and not transmitting[node]:  # Node has packets to send and is not already transmitting
                        children = list(G.successors(node))
                        if children:
                            # Choose a child randomly to try to send the packet to
                            child = random.choice(children)

                            if not transmitting[child]:  # Check if the child is not currently transmitting
                                # Determine if transmission is successful based on RSSI
                                rssi = adj_matrix[node][child]  # Get RSSI value for the link
                                transmission_success = random.random() <= rssi

                                if transmission_success and packets[child] < packets_per_node:
                                    packets[child] += 1
                                    packets[node] -= 1
                                    transmitting[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision

                # Reset the transmitting status for the next step
                transmitting = {node: False for node in G.nodes}

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


    def evaluate_dag_performance_combined(self, eval_up, eval_down, G, adj_matrix, epoch_len=1, packets_per_node=15, max_steps_up=-1, max_steps_down=-1):
        perf_up = eval_up(G, adj_matrix, max_steps=max_steps_up)
        perf_down = eval_down(G, adj_matrix, max_steps=max_steps_down)
        return G, perf_up, perf_down

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

        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                max_steps_up = min(CythonDAGOperation.cython_evaluate_dag_performance_up(random.choice(dags), adj_matrix), max_steps_up)
                max_steps_down = min(CythonDAGOperation.cython_evaluate_dag_performance_down(random.choice(dags), adj_matrix), max_steps_down)
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        # Prepare the arguments as a list of tuples
        args_up = [(i, dag, adj_matrix, CythonDAGOperation.cython_evaluate_dag_performance_up, max_steps_up) for i, dag in enumerate(dags)]
        args_down = [(i, dag, adj_matrix, CythonDAGOperation.cython_evaluate_dag_performance_down, max_steps_down) for i, dag in enumerate(dags)]

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

        print(len(up_results))
        print(len(down_results))
        print(len(combined_results))

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

        print(len(up_results))
        print(len(down_results))
        print(len(combined_results))

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
        # Pre-compute step threshold. This way we don't compute performance for bad DAGs because it is not relevant and waists execution time.
        max_steps_up = 1e9  # very high value just for initialization
        max_steps_down = 1e9  # very high value just for initialization
        if len(dags) > 800:
            iter = 50 if len(dags) >= 50 else len(dags)
            for _ in range(0, iter):
                max_steps_up = min(CythonDAGOperation.cython_evaluate_dag_performance_up(random.choice(dags), adj_matrix), max_steps_up)
                max_steps_down = min(CythonDAGOperation.cython_evaluate_dag_performance_down(random.choice(dags), adj_matrix), max_steps_down)
            print("Max steps = " + str(max_steps_up))
            print("Max steps = " + str(max_steps_down))
        else:
            max_steps_up = -1
            max_steps_down = -1

        dags_stack_copy = copy.deepcopy(dags)
        up_results = []
        down_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.evaluate_dag_performance_combined, 
                    CythonDAGOperation.cython_evaluate_dag_performance_up, 
                    CythonDAGOperation.cython_evaluate_dag_performance_down,
                    dags_stack_copy.pop(),
                    adj_matrix,
                    max_steps_up=max_steps_up,
                    max_steps_down=max_steps_down)
                for i in range(min(max_workers, len(dags_stack_copy)))}
            for future in concurrent.futures.as_completed(futures):
                current_dag, perf_up, perf_down = future.result()
                up_results.append((current_dag, perf_up))
                down_results.append((current_dag, perf_down))

                if max_steps_up == -1 and max_steps_down == -1:
                    max_steps_up = perf_up
                    max_steps_down = perf_down

                if perf_up < max_steps_up:
                    max_steps_up = perf_up if (perf_up / max_steps_up) > delta_threshold else max_steps_up * (1 - reduce_ratio)

                if perf_down < max_steps_down:
                    max_steps_down = perf_down if (perf_down / max_steps_down) > delta_threshold else max_steps_down * (1 - reduce_ratio)
                    
                futures.remove(future)
                if len(dags_stack_copy):
                    futures.add(executor.submit(
                        self.evaluate_dag_performance_combined, 
                        CythonDAGOperation.cython_evaluate_dag_performance_up, 
                        CythonDAGOperation.cython_evaluate_dag_performance_down,
                        dags_stack_copy.pop(),
                        max_steps_up=max_steps_up * margin_max_step,
                        max_steps_down=max_steps_down * margin_max_step)
                    )

        # Combine the two lists
        up_results_np = np.array([item[1] for item in up_results])
        down_results_np = np.array([item[1] for item in down_results])

        # normalize and combine
        up_results_np /= np.max(up_results_np)
        down_results_np /= np.max(down_results_np)
        combined_results_np = up_results_np + down_results_np

        combined_results = [(up_results[i][0], int(item)) for i, item in enumerate(combined_results_np)]

        print(len(up_results_np))
        print(len(down_results_np))
        print(len(combined_results))

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


if __name__ == '__main__':
    generator = DAGDatasetGenerator()
    generator.run_once_with_adaptive_steps(15)