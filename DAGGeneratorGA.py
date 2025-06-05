"""
Author: Eric Tran <eric.tran@heig-vd.ch>
"""
import numpy as np
import networkx as nx
from DAGDatasetGenerator import DAGDatasetGenerator
import sys
import time
import copy

class DAGGeneratorGA:
    def __init__(self):
        self.rng = np.random.default_rng()
        self.verbose = False

    """Return the nodes in the subtree rooted at `root`."""
    def get_subtree(self, G, root):
        return nx.descendants(G, root) | {root}


    def crossover_with_enforcing(self, parent1: nx.DiGraph, parent2: nx.DiGraph, adj_matrix, root=0):
        """
        Tree-preserving crossover: Create new tree by merging two selected trees (parent1: reference, parent2: crossover tree)
        """
        cx_tree = copy.deepcopy(parent1) 

        nodes1 = list(cx_tree.nodes - {root})
        nodes2 = list(parent2.nodes - {root})

        if not nodes1 or not nodes2:
            return None  # Nothing to swap

        # Randomly select a crossover point (subtree root, not the main root)
        crossover_node1 = self.rng.choice(nodes1)

        # We need to limit the choice based on existing link quality from crossover_node1 
        crossover_candidates = np.where((adj_matrix[crossover_node1] > 0.0) & (np.arange(adj_matrix.shape[0]) != root))[0]
        if not len(crossover_candidates): # Can't swap, only link to root
            return None
        crossover_node2 = self.rng.choice(crossover_candidates)

        # Get subtrees rooted at these nodes
        subtree1_nodes = self.get_subtree(cx_tree, crossover_node1)
        subtree2_nodes = self.get_subtree(parent2, crossover_node2)

        # Get subtree graphs
        subtree2 = parent2.subgraph(subtree2_nodes).copy()

        # Remove subtrees
        cx_tree.remove_nodes_from(subtree1_nodes)
        cx_tree.remove_nodes_from(subtree2_nodes)

        # Attach subtree2 to p1
        cx_tree.add_nodes_from(parent2.subgraph(subtree2).nodes(data=True))
        cx_tree.add_edges_from(parent2.subgraph(subtree2).edges())

        # Connect to parent
        parent_edge = list(parent1.in_edges(crossover_node1))
        if parent_edge:
            parent = parent_edge[0][0]
            cx_tree.add_edge(parent, crossover_node2)


        # Reattach all orphan nodes to the main tree
        all_connected_nodes = nx.bfs_tree(cx_tree, root).nodes  
        orphan_nodes = parent1.nodes - all_connected_nodes
        if len(orphan_nodes):
           for current_node in orphan_nodes:
               # Randomly select right candidates
               candidates = np.where(adj_matrix[current_node] > 0.0)[0] 
               selected_node = self.rng.choice(candidates)
               if current_node == root: # Candidates are child nodes to connect wth the root node
                    cx_tree.add_edge(root, selected_node)
               else: # Candidates are parent nodes to connect with the child node
                    cx_tree.add_edge(selected_node, current_node)
        return cx_tree
        
    
    """
    Randomly reassign parent of a node
    """
    def mutate(self, ref_dag, adj_matrix, ref_graph, root=0):
        """Randomly change parent of a node to another"""
        dag = ref_dag.copy()
        # Select mutation node
        nodes_candidates = list(set(dag.nodes) - {root})
        mutation_node = self.rng.choice(nodes_candidates)
        # Remove edge linking mutation_node and parent
        edge = list(dag.in_edges(mutation_node))[0]
        dag.remove_edge(*edge)

        # Add link to a random node with an active link quality and not part of the detached subtree
        all_children_nodes = nx.descendants(dag, mutation_node)
        parent_candidates = np.where((adj_matrix[mutation_node] > 0.0) & (np.isin(np.arange(adj_matrix.shape[0]), all_children_nodes, invert=True)))[0]
        if len(parent_candidates):
            new_parent = self.rng.choice(list(parent_candidates))
            dag.add_edge(new_parent, mutation_node)
        return dag

    """
    """
    def is_valid_dag(self, dag):
        for child, parent in enumerate(dag):
            if parent is not None:
                # Check if there’s a valid edge in the adjacency matrix
                if self.adj_matrix[parent][child] == 0:
                    return False
        return True


    # Function to fix invalid dags by reassigning invalid edges
    def fix_invalid_dag(self, dag):
        fixed_dag = dag[:]
        for child, parent in enumerate(dag):
            if parent is not None and self.adj_matrix[parent][child] == 0:
                # Reassign to a valid parent
                valid_parents = [p for p in range(child) if self.adj_matrix[p][child] > 0]
                if valid_parents:
                    fixed_dag[child] = self.rng.choice(valid_parents)
        return fixed_dag

    def generate(self, population: list[list], adj_matrix, population_size: int = 2000, mutation_rate: float = 1.0, generations: int = 10):
        """
        Generate new trees
        """
        initial_n_pop = len(population)
        if len(population) == 1:# Need at least 2 population to perform crossover
            population.append(population[0].copy())

        for gen in range(generations):
            new_population = population[:initial_n_pop] # We always keep the starting population in order to compare perf with new ones
            while len(new_population) < population_size:
                ref_graph = nx.from_numpy_array(adj_matrix)
                parent1, parent2 = self.rng.choice(population, size=2)
                parent1_tree = nx.from_edgelist(parent1, create_using=nx.DiGraph)
                parent2_tree = nx.from_edgelist(parent2, create_using=nx.DiGraph)
                child = self.crossover_with_enforcing(parent1_tree, parent2_tree, adj_matrix)
                if child is None:
                    continue

                if self.rng.random() < mutation_rate:
                    child = self.mutate(child, adj_matrix, ref_graph)
                
                if not nx.is_tree(child):
                    continue
                new_population.append(list(child.edges(data=False)))

            population = new_population
        return population
    
    def initialize_dags(self, n_dags, adj_matrix: np.array, root:int = 0, max_children:int = 5):
        """
        Initialize tree layer by layer randomly while enforcing link quality rule
        """
        n_nodes = adj_matrix.shape[0]
        generated_dags = []
        while len(generated_dags) < n_dags:
            if self.verbose:
                print(f"Current # dags: {len(generated_dags)}")
            orphan_nodes = set(np.arange(n_nodes)) - {root}
            dag = nx.DiGraph()
            dag.add_node(root)
            parents_to_process = [root]
            # Create dag randomly, layer by layer
            while len(parents_to_process):
                if self.verbose:
                    print(f"Nodes to process: {orphan_nodes}")
                parent = parents_to_process.pop()
                all_childrens = np.where((adj_matrix[parent] > 0.0) & (np.isin(np.arange(adj_matrix.shape[0]), list(orphan_nodes))))[0]
                if len(all_childrens):
                    selected_childrens = self.rng.choice(all_childrens, self.rng.integers(1, min(len(all_childrens), max_children) + 1))
                    new_edges = [(parent, child) for child in selected_childrens]
                    orphan_nodes -= set(selected_childrens)
                    dag.add_edges_from(new_edges) 
                    parents_to_process += selected_childrens.tolist()

            # Attach all remaining orphans if they were not processed previously
            orphans_to_attach = list(orphan_nodes)
            cur_it = 0
            while len(orphans_to_attach): 
                if self.verbose:
                    print(f"Orphan to attach: {orphans_to_attach}")
                child = orphans_to_attach.pop(0)
                parent_candidates = np.where((adj_matrix[child] > 0.0) & (np.isin(np.arange(adj_matrix.shape[0]), list(dag.nodes))))[0]
                if not len(parent_candidates):
                    if cur_it > len(orphans_to_attach): # Small trick, if we cross the full list without adding any new edge, we are in a deadlock situation
                        raise Exception("Loop between orphan nodes")
                    orphans_to_attach.append(child)
                    cur_it += 1
                    continue
                selected_parent = self.rng.choice(parent_candidates)
                dag.add_edge(selected_parent, child)
                cur_it = 0

            if not nx.is_tree(dag):
                print("Not a tree, carry on initialization")
                continue
            generated_dags.append(list(dag.edges()))
        return generated_dags

    def fit(self, adj_matrix, population_size: int = 2000, mutation_rate: float = 1.0, generations: int = 10, early_stop_score = 10, max_iter_stagnation = 5, verbose = False):
        """
        Fitting to find best tree based on adjacent matrix
        """
        self.verbose = verbose

        n_dags = 10
        generator = DAGDatasetGenerator()
        if self.verbose:
            print("Initialize DAG")
        dags = self.initialize_dags(10, adj_matrix)

        if self.verbose:
            print("Initial simulation perf")
        dags, perfs= generator.get_best_dag_double_flux(copy.deepcopy(dags), adj_matrix, keep_dags_count=1, keep_random_dags=False, max_steps=-1)
        dags = list(dags)
        ref_steps = perfs[0]
        no_improvement = 0
        best_perf = -1
        best_dags = dags
        while True:
            if self.verbose:
                print("Perform generation")
            all_generated_dags = self.generate(dags, adj_matrix, population_size, mutation_rate, generations)

            if self.verbose:
                print("Run simulation perf")
            dags, perfs = generator.get_best_dag_double_flux(all_generated_dags, adj_matrix, keep_dags_count=n_dags, keep_random_dags=False, max_steps=ref_steps, verbose=False)

            dags = list(dags) # Convert tuple structure to list
            if best_perf == -1 or perfs[0] + early_stop_score < best_perf:
                best_perf = perfs[0]
                best_dags = dags
                no_improvement = 0
                if self.verbose:
                    print(f"Found better dag, perf: {best_perf}")
            else:
                no_improvement += 1
                if self.verbose:
                    print(f"No improvement({no_improvement})")

            if no_improvement > max_iter_stagnation: # We stop the fitting process if there is no improvement after multiple iterations
                break

        return best_dags[0], best_perf


if __name__ == '__main__':
    KEEP_DAG_COUNT = 1
    KEEP_RANDOM_DAG = False
    MAX_STEPS = -1
    VERBOSE = False
    N = 20
    dga = DAGGeneratorGA()
    generator = DAGDatasetGenerator()

    start_ga = time.time()
    adj_matrix, density_factor = generator.generate_random_adj_matrix(N)
    best_dag, best_perf = dga.fit(adj_matrix, generations=1, population_size=500)
    end_ga = time.time()
    print(f"Perf: {best_perf}")