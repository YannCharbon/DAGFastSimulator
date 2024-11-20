/**
 * @brief C++ Shared library to accelerate execution of python simulation
 * @author Yann Charbon <yann.charbon@heig-vd.ch>
 * @note See README for more information
 */

#include "CDAGOperation.h"
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>
#include <vector>
#include <list>
#include <stack>
#include <gmp.h>

using namespace std;

#ifndef MAX_NODES
#define MAX_NODES   50
#endif

#ifndef MAX_CHILDREN
#define MAX_CHILDREN   10
#endif

#ifndef LOG_TIMINGS
#define LOG_TIMINGS     0
#endif

#ifndef VERBOSE
#define VERBOSE     0
#endif


class CombinationIterator {
public:
    CombinationIterator(const std::vector<Edge>& edges, int k)
        : edges(edges), k(k), indices(k), combination(k) {  // pre-allocate `combination`
        for (int i = 0; i < k; ++i) {
            indices[i] = i;
        }
    }

    bool hasNext() const {
        return !finished;
    }

    const std::vector<Edge>& next() {
        for (int i = 0; i < k; ++i) {
            combination[i] = edges[indices[i]];
        }
        advance();
        return std::move(combination);  // Optimization by compiler
    }

    void skipCombinations(int skip_count) {
        for (int i = 0; i < skip_count && hasNext(); ++i) {
            advance();
        }
    }

private:
    const std::vector<Edge>& edges;
    int k;
    std::vector<int> indices;
    std::vector<Edge> combination;  // reuse this for each combination
    bool finished = false;

    void advance() {
        int n = edges.size();
        for (int i = k - 1; i >= 0; --i) {
            if (indices[i] != i + n - k) {
                ++indices[i];
                for (int j = i + 1; j < k; ++j) {
                    indices[j] = indices[j - 1] + 1;
                }
                return;
            }
        }
        finished = true;
    }
};


static inline long get_microseconds() {
    struct timespec time;
    clock_gettime(CLOCK_MONOTONIC, &time);
    return (time.tv_sec * 1000000L) + (time.tv_nsec / 1000);
}

// Used in DOWN link simulation. Checks if there is any node which is not filled with the target amount of packets.
static inline bool any_packet_missing(int node_count, int packets_per_node, int *packets) {
    int i = node_count;
    while (packets[--i] >= packets_per_node && i > 0);
    return (bool)i;
}

static inline void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

static inline int get_children(Edge *edges, int edges_count, int parent_node, int *children, int children_count) {
    children_count = 0;
    memset(children, -1, MAX_CHILDREN);
    for (int i = 0; i < edges_count; i++) {
        if (edges[i].parent == parent_node) {
            children[children_count++] = edges[i].child;
            if (children_count >= MAX_CHILDREN) {
                break;
            }
        }
    }
    return children_count;
}

static inline int get_children_transmit_intents(int *children_transmit_intents, int children_transmit_intents_count, int *children, int children_count, bool *transmit_intent) {
    children_transmit_intents_count = 0;
    memset(children_transmit_intents, false, MAX_CHILDREN);

    for (int i = 0; i < children_count; i++) {
        if (transmit_intent[children[i]] == true) {
            children_transmit_intents[children_transmit_intents_count++] = children[i];
        }
    }

    return children_transmit_intents_count;
}

static inline bool transmission_success(float link_quality) {
    if ((float)rand()/(float)RAND_MAX <= link_quality) {
        return true;
    }

    return false;
}

static inline int compute_sparsity_metric(const int *array, int size) {
    if (size <= 0) {
        return 0; // Handle empty array case
    }

    int sum = 0;
    for (int i = 0; i < size; i++) {
        sum += array[i];
    }

    // Integer division for the mean
    int mean = sum / size;

    int abs_dev_sum = 0;
    for (int i = 0; i < size; i++) {
        abs_dev_sum += abs(array[i] - mean);
    }

    // Return the average absolute deviation
    return abs_dev_sum / size;
}


/**
    Uplink global bandwidth measurement.

    This method computes the number of iterations it takes to empty the packet queue of each node within the network.
 */
int evaluate_dag_performance_up(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps) {
    int packets[MAX_NODES];
    bool transmit_intent[MAX_NODES];
    bool transmitting[MAX_NODES];

    int random_indexes[MAX_NODES];

    int children[MAX_CHILDREN];
    int children_count = 0;
    int children_transmit_intents[MAX_CHILDREN];
    int children_transmit_intents_count = 0;

    int total_packets = (nodes_count - 1) * packets_per_node;

    if (epoch_len <= 0) {
        epoch_len = 2;
    }
    if (packets_per_node <= 0) {
        packets_per_node = 15;
    }
    if (max_steps <= 0) {
        max_steps = -1;
    }


    for (int i = 0; i < nodes_count; i++) {
        random_indexes[i] = i;
    }
    // Seed the random number generator
    srand(time(NULL));

    int avg_steps = 0;
    bool early_stop = false;

    for (int epoch = 0; epoch < epoch_len; epoch++) {

        for (int i = 0; i < nodes_count; i++) {
            packets[i] = packets_per_node;
        }
        memset(transmitting, false, nodes_count);
        memset(transmit_intent, true, nodes_count);
        total_packets = (nodes_count - 1) * packets_per_node;


        int steps = 0;
        while (total_packets > 0) {
            steps++;

            for (int i = 0; i < nodes_count; i++) {
                if (packets[i] > 0) {
                    transmit_intent[i] = true;
                } else {
                    transmit_intent[i] = false;
                }
            }

            // Process packet transmission for all nodes with randomized priority
            shuffle(random_indexes, nodes_count);
            for (int i = 0; i < nodes_count; i++) {
                int parent = random_indexes[i];
                if (transmitting[parent]) {
                    continue;
                }

                children_count = get_children(edges, edges_count, parent, children, children_count);
                if (children_count > 0) {
                    children_transmit_intents_count = get_children_transmit_intents(children_transmit_intents, children_transmit_intents_count, children, children_count, transmit_intent);
                    if (children_transmit_intents_count > 0) {
                        int transmitting_child = children_transmit_intents[rand() % children_transmit_intents_count];
                        if (transmitting[transmitting_child] == false) {
                            float link_quality = adj_matrix[parent][transmitting_child];

                            if (transmission_success(link_quality)) {
                                packets[parent]++;
                                packets[transmitting_child]--;
                                if (parent == 0) {
                                    total_packets--;
                                }
                                transmit_intent[transmitting_child] = false;
                                transmitting[parent] = true;    // Parent is not actually transmitting, but it is busy while receiving from child
                            }

                            transmitting[transmitting_child] = true;
                        }
                    }
                }
            }

            // Reset the transmitting and receiving status for the next step
            memset(transmitting, false, nodes_count);

            // Root node never holds an packet
            packets[0] = 0;

            if (max_steps != -1 && steps > max_steps) {
                early_stop = true;
                break;
            }
        }

        avg_steps += steps;

        if (early_stop) {
            break;
        }
    }

    if (!early_stop) {
        return avg_steps / epoch_len;
    } else {
        return max_steps;
    }
}

/**
    Downlink global bandwidth measurement.

    This method computes the number of iterations it takes to fill the packet queue of each node within the network to a given count.
 */
int evaluate_dag_performance_down(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps) {
    int packets[MAX_NODES];
    bool transmitting[MAX_NODES];

    int random_indexes[MAX_NODES];

    int children[MAX_CHILDREN];
    int children_count = 0;

    if (epoch_len <= 0) {
        epoch_len = 2;
    }
    if (packets_per_node <= 0) {
        packets_per_node = 15;
    }
    if (max_steps <= 0) {
        max_steps = -1;
    }


    for (int i = 0; i < nodes_count; i++) {
        random_indexes[i] = i;
    }
    // Seed the random number generator
    srand(time(NULL));

    int avg_steps = 0;
    bool early_stop = false;

    for (int epoch = 0; epoch < epoch_len; epoch++) {

        for (int i = 0; i < nodes_count; i++) {
            packets[i] = 0;
        }
        memset(transmitting, false, nodes_count);

        int steps = 0;
        while (any_packet_missing(nodes_count, packets_per_node, packets)) {
            steps++;

            // Root node inserts a packet into the network if it's not transmitting
            if (packets[0] < packets_per_node && !transmitting[0]) {
                packets[0]++;
            }

            shuffle(random_indexes, nodes_count);
            for (int i = 0; i < nodes_count; i++) {
                int parent = random_indexes[i];

                // Node has packets to send and is not already transmitting
                if (packets[parent] > 0 && !transmitting[parent]) {
                    children_count = get_children(edges, edges_count, parent, children, children_count);
                    if (children_count > 0) {
                        // Choose a child randomly to try to send the packet to
                        int child = children[rand() % children_count];

                        // Check if the child is not currently transmitting
                        if (!transmitting[child]) {
                            float link_quality = adj_matrix[parent][child];

                            if (transmission_success(link_quality) && packets[child] < packets_per_node) {
                                packets[child]++;
                                packets[parent]--;
                                transmitting[child] = true; // Mark the child as busy (not actually transmitting but receiving from parent).
                            }

                            transmitting[parent] = true; // Mark the parent as transmitting. It is also the case when transmission success is false because it simulates a collision by the fact the child is busy.
                        }
                    }
                }
            }

            memset(transmitting, false, nodes_count);
            packets[0] = 0;

            if (max_steps != -1 && steps > max_steps) {
                early_stop = true;
                break;
            }
        }

        avg_steps += steps;

        if (early_stop) {
            break;
        }
    }

    if (!early_stop) {
        return avg_steps / epoch_len;
    } else {
        return max_steps;
    }
}


/**
    Uplink and downlink simultaneous global bandwidth measurement.

    This method computes the number of iterations it takes to empty and fill the packet queues of each node within the network.
 */
int evaluate_dag_performance_double_flux(Edge *edges, int edges_count, float **adj_matrix, int nodes_count, int epoch_len, int packets_per_node, int max_steps) {
    int packets_up[MAX_NODES];
    int packets_down[MAX_NODES];
    bool transmit_intent_up[MAX_NODES];
    bool transmitting[MAX_NODES];

    int random_indexes[MAX_NODES];

    int children[MAX_CHILDREN];
    int children_count = 0;
    int children_transmit_intents[MAX_CHILDREN];
    int children_transmit_intents_count = 0;

    int total_packets_up = (nodes_count - 1) * packets_per_node;

    if (epoch_len <= 0) {
        epoch_len = 2;
    }
    if (packets_per_node <= 0) {
        packets_per_node = 15;
    }
    if (max_steps <= 0) {
        max_steps = -1;
    }


    for (int i = 0; i < nodes_count; i++) {
        random_indexes[i] = i;
    }
    // Seed the random number generator
    srand(time(NULL));

    int avg_steps = 0;
    bool early_stop = false;

    for (int epoch = 0; epoch < epoch_len; epoch++) {

        for (int i = 0; i < nodes_count; i++) {
            packets_up[i] = packets_per_node;
            packets_down[i] = 0;
        }
        memset(transmitting, false, nodes_count);
        memset(transmit_intent_up, true, nodes_count);
        total_packets_up = (nodes_count - 1) * packets_per_node;

        bool down_finished = false;

        bool up_not_down = false;

        int steps = 0;
        while (total_packets_up > 0 || !down_finished) {
            steps++;

            // Root node Root node has a new packet to insert into the network. Does not send it yet.
            if (packets_down[0] < packets_per_node) {
                packets_down[0]++;
            }

            for (int i = 0; i < nodes_count; i++) {
                if (packets_up[i] > 0) {
                    transmit_intent_up[i] = true;
                } else {
                    transmit_intent_up[i] = false;
                }
            }

            // Process packet transmission for all nodes with randomized priority
            shuffle(random_indexes, nodes_count);
            for (int i = 0; i < nodes_count; i++) {
                int parent = random_indexes[i];

                // Decide whether the current node wants to transmit UP or DOWN
                if (total_packets_up > 0 && !down_finished) {
                    if (rand() % 100 >= 50) {
                        up_not_down = true;
                    } else {
                        up_not_down = false;
                    }
                } else if (total_packets_up == 0) {
                    up_not_down = false;    // UP operation done -> only DOWN has to run
                } else if (down_finished) {
                    up_not_down = true;     // DOWN operation done -> only UP has to run
                }

                if (up_not_down) {
                    if (transmitting[parent]) {
                        continue;
                    }

                    children_count = get_children(edges, edges_count, parent, children, children_count);
                    if (children_count > 0) {
                        children_transmit_intents_count = get_children_transmit_intents(children_transmit_intents, children_transmit_intents_count, children, children_count, transmit_intent_up);
                        if (children_transmit_intents_count > 0) {
                            int transmitting_child = children_transmit_intents[rand() % children_transmit_intents_count];
                            if (transmitting[transmitting_child] == false) {
                                float link_quality = adj_matrix[parent][transmitting_child];

                                if (transmission_success(link_quality)) {
                                    packets_up[parent]++;
                                    packets_up[transmitting_child]--;
                                    if (parent == 0) {
                                        total_packets_up--;
                                    }
                                    transmit_intent_up[transmitting_child] = false;
                                    transmitting[parent] = true;    // Parent is not actually transmitting, but it is busy while receiving from child
                                }

                                transmitting[transmitting_child] = true;
                            }
                        }
                    }
                } else {    // DOWN
                    // Node has packets to send and is not already transmitting
                    if (packets_down[parent] > 0 && !transmitting[parent]) {
                        children_count = get_children(edges, edges_count, parent, children, children_count);
                        if (children_count > 0) {
                            // Choose a child randomly to try to send the packet to
                            int child = children[rand() % children_count];

                            // Check if the child is not currently transmitting
                            if (!transmitting[child]) {
                                float link_quality = adj_matrix[parent][child];

                                if (transmission_success(link_quality) && packets_down[child] < packets_per_node) {
                                    packets_down[child]++;
                                    packets_down[parent]--;
                                    transmitting[child] = true; // Mark the child as busy (not actually transmitting but receiving from parent).
                                }

                                transmitting[parent] = true; // Mark the parent as transmitting. It is also the case when transmission success is false because it simulates a collision by the fact the child is busy.
                            }
                        }
                    }
                }
            }

            // Reset the transmitting and receiving status for the next step
            memset(transmitting, false, nodes_count);

            // Root node never holds an packet
            packets_up[0] = 0;

            // Store DOWN finish condition
            down_finished = !any_packet_missing(nodes_count, packets_per_node, packets_down);

            if (max_steps != -1 && steps > max_steps) {
                early_stop = true;
                break;
            }
        }

        avg_steps += steps;

        if (early_stop) {
            break;
        }
    }

    if (!early_stop) {
        return avg_steps / epoch_len;
    } else {
        return max_steps;
    }
}

// Non-recursive version using a stack.
static inline void dfs_tree(float **adj_matrix, int nodes_count, bool *visited, int start_node, std::vector<Edge> *tree_edges) {
    std::stack<int> stack;
    stack.push(start_node);
    visited[start_node] = true;

    while (!stack.empty()) {
        int current_node = stack.top();
        stack.pop();

        for (int neighbour = 0; neighbour < nodes_count; neighbour++) {
            if (adj_matrix[current_node][neighbour] > 0 && !visited[neighbour]) {
                // Mark as visited and add the edge to the tree edges
                visited[neighbour] = true;
                Edge edge = { .parent = current_node, .child = neighbour };
                tree_edges->push_back(edge);

                // Push the neighbor to the stack to continue DFS from this node later
                stack.push(neighbour);
            }
        }
    }
}

// Recursive version.
/*static void dfs_tree(float **adj_matrix, int nodes_count, bool *visited, int current_node, vector<Edge> *tree_edges) {
    visited[current_node] = true;
    for (int neighbour = 0; neighbour < nodes_count; neighbour++) {
        if (adj_matrix[current_node][neighbour] > 0 && !visited[neighbour]) {
            Edge edge = {.parent = current_node, .child = neighbour};
            tree_edges->push_back(edge);
            dfs_tree(adj_matrix, nodes_count, visited, neighbour, tree_edges);
        }
    }
}*/

static inline uint64_t combinations(int n_edges, int k_nodes) {
    mpz_t n_fact, k_fact, n_minus_k_fact, denom, result;
    mpz_init(n_fact);
    mpz_init(k_fact);
    mpz_init(n_minus_k_fact);
    mpz_init(denom);
    mpz_init(result);

    // Calculate factorials
    mpz_fac_ui(n_fact, n_edges);            // n_edges!
    mpz_fac_ui(k_fact, k_nodes);            // k_nodes!
    mpz_fac_ui(n_minus_k_fact, n_edges - k_nodes); // (n_edges - k_nodes)!

    // Calculate denominator = k_nodes! * (n_edges - k_nodes)!
    mpz_mul(denom, k_fact, n_minus_k_fact);

    // Calculate result = n_edges! / (k_nodes! * (n_edges - k_nodes)!)
    mpz_divexact(result, n_fact, denom);

    // Check if the result fits in 64-bit unsigned integer
    uint64_t total_combinations;
    if (mpz_fits_ulong_p(result)) {
        total_combinations = mpz_get_ui(result);
    } else {
        // Overflow: set to 0 as per requirements
        total_combinations = 0;
    }

    // Clear the GMP variables
    mpz_clear(n_fact);
    mpz_clear(k_fact);
    mpz_clear(n_minus_k_fact);
    mpz_clear(denom);
    mpz_clear(result);

    return total_combinations;
}

static inline bool all_values_true(bool *array, int array_len) {
    for (int i = 0; i < array_len; i++) {
        if (array[i] == false) {
            return false;
        }
    }
    return true;
}


Edge** generate_subset_dags(float **adj_matrix, int nodes_count, int *generated_dags_count, int skip_factor, bool test_mode, bool no_skip) {
#if LOG_TIMINGS > 0
    long start_time = get_microseconds();
#endif

    int nodes[MAX_NODES];
    for (int i = 0; i < nodes_count; i++) {
        nodes[i] = i;
    }

    bool visited[MAX_NODES];

    vector<Edge> all_edges;
    for (int i = 0; i < nodes_count; i++) {
        for (int j = i + 1; j < nodes_count; j++) {
            if (adj_matrix[i][j] > 0.0) {
                Edge edge = {.parent = i, .child = j};
                all_edges.push_back(edge);
            }
        }
    }

    list<vector<Edge>> all_possible_trees;

    // Seed the random number generator
    srand(time(NULL));

    int n_edges = all_edges.size();
    int k_nodes = nodes_count - 1;

    if (n_edges < k_nodes) {
#if VERBOSE == 1
        printf("Error: Not enough edges to connect every node. Skipping.\n");
#endif
        return NULL;
    }

    uint64_t total_combinations = combinations(n_edges, k_nodes);
    CombinationIterator comb_iter(all_edges, k_nodes);
#if LOG_TIMINGS > 0
    long end_of_combination_computation = get_microseconds();
    printf("C - Took %ld [us] to init and compute combinations\n", end_of_combination_computation - start_time);

#endif
#if VERBOSE == 1
    printf("total_combinations %ld n_edges %d k_nodes %d\n", total_combinations, n_edges, k_nodes);
#endif

    if (skip_factor == 0) {
        // automatic mode
        skip_factor = total_combinations / 30000000;    // This reduction factor has been verified statistically to not loose the best DAGs
        if (skip_factor < 1) {
            skip_factor = 1;
        }
#if VERBOSE == 1
        printf("Skip factor automatically adjusted to %d\n", skip_factor);
#endif
    }

#if LOG_TIMINGS > 0
    long start_of_tree_matrix_generation = get_microseconds();
#endif
    // Allocation done once for all
    float *tree_matrix[nodes_count];
    for (int i = 0; i < nodes_count; i++) {
        tree_matrix[i] = (float *)calloc(nodes_count, sizeof(float));
        if (tree_matrix[i] == NULL) {
            printf("Allocation error 1\n");
            return NULL;
        }
    }

#if LOG_TIMINGS > 0
    long end_of_tree_matrix_generation = get_microseconds();
    printf("C - Took %ld [us] to generate tree matrix\n", end_of_tree_matrix_generation - start_of_tree_matrix_generation);

#if LOG_TIMINGS == 2
    long step_0, step_1, step_2, step_3, step_4, step_5, step_6, step_7;
#endif
    long start_of_tree_computation = get_microseconds();
#endif

    vector<Edge> tree_edges;
    tree_edges.reserve(nodes_count - 1);  // Reserve memory once

    while (comb_iter.hasNext()) {
#if LOG_TIMINGS == 2
        step_0 = get_microseconds();
#endif
        std::vector<Edge> edges = comb_iter.next();
#if LOG_TIMINGS == 2
        step_1 = get_microseconds();
#endif

        // Equivalent of Python tree_matrix = np.zeros_like(adj_matrix)
        //memset(tree_matrix[0], 0, sizeof(float) * nodes_count * nodes_count);
        for (int i = 0; i < nodes_count; i++) {
            for (int j = 0; j < nodes_count; j++) {
                tree_matrix[i][j] = 0.0;
            }
        }

        for (const auto& edge : edges) {
            tree_matrix[edge.parent][edge.child] = 1.0;
            tree_matrix[edge.child][edge.parent] = 1.0;
        }
#if LOG_TIMINGS == 2
        step_2 = get_microseconds();
#endif

        memset(visited, false, nodes_count);

#if LOG_TIMINGS == 2
        step_3 = get_microseconds();
#endif

        // Check if the selected edges form a connected tree
        dfs_tree(tree_matrix, nodes_count, visited, 0, &tree_edges);
#if LOG_TIMINGS == 2
        step_4 = get_microseconds();
#endif
        if (tree_edges.size() == nodes_count - 1 && all_values_true(visited, nodes_count)) {
            all_possible_trees.push_back(tree_edges);
        }

        if (!no_skip){
            // Progress with iterator by a random jump (scale by the number of nodes and number of edges)
            int64_t total_skip = (rand() % (k_nodes * n_edges * skip_factor));
            if (test_mode) {
                // Deterministic behaviour
                total_skip = (int64_t)((k_nodes * n_edges * skip_factor) / 2);
            }
#if LOG_TIMINGS == 2
            step_5 = get_microseconds();
#endif
            comb_iter.skipCombinations(total_skip);
            if (!comb_iter.hasNext()) {
                break;
            }
        }

        tree_edges.clear();

#if LOG_TIMINGS == 2
        step_6 = get_microseconds();
        printf("Step 1 %ld [us], Step 2 %ld [us], Step 3 %ld [us], Step 4 %ld [us], Step 5 %ld [us], Step 6 %ld [us]\n", step_1 - step_0, step_2 - step_1, step_3 - step_2, step_4 - step_3, step_5 - step_4, step_6 - step_5);
#endif
    }

#if LOG_TIMINGS > 0
    long end_of_tree_computation = get_microseconds();
    printf("C - Took %ld [us] to generate all dags\n", end_of_tree_computation - start_of_tree_computation);

    long start_of_format_conversion = get_microseconds();
#endif

    for (int i = 0; i < nodes_count; i++) {
        free(tree_matrix[i]);
    }

    // Convert list<vector<Edge>> to Edge[][]
    *generated_dags_count = all_possible_trees.size();
    auto it = all_possible_trees.begin();
    Edge **all_possible_trees_c = (Edge **)calloc(all_possible_trees.size(), sizeof(Edge *));
    if (all_possible_trees_c == NULL) {
        printf("Allocation error 2\n");
        return NULL;
    }
    for (int i = 0; i < all_possible_trees.size(); i++) {
        all_possible_trees_c[i] = (Edge *)calloc(k_nodes, sizeof(Edge));
        if (all_possible_trees_c[i] == NULL) {
            printf("Allocation error 3\n");
        }
        auto edges_it = it->begin();
        for (int j = 0; j < k_nodes; j++) {
            memcpy(&all_possible_trees_c[i][j], &(*edges_it), sizeof(Edge));
            edges_it++;
        }
        it++;
    }

#if LOG_TIMINGS > 0
    long end_of_format_conversion = get_microseconds();
    printf("C - Took %ld [us] to convert format\n", end_of_format_conversion - start_of_format_conversion);
    printf("C - Total time %ld [us]\n", end_of_format_conversion - start_time);
#endif

    return all_possible_trees_c;
}

void free_all_possible_tree(Edge **all_possible_trees_c, int dags_count) {
    for (int i = 0; i < dags_count; i++) {
        free(all_possible_trees_c[i]);
    }
    free(all_possible_trees_c);
}