/**
 * @author Yann Charbon <yann.charbon@heig-vd.ch>
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

#ifndef MAX_NODES
#define MAX_NODES   50
#endif

#ifndef MAX_CHILDREN
#define MAX_CHILDREN   10
#endif

typedef struct {
    int parent;
    int child;
} Edge;


// Used in DOWN link simulation. Checks if there is any node which is not filled with the target amount of packets.
static bool any_packet_missing(int node_count, int packets_per_node, int *packets) {
    int i = node_count;
    while (packets[--i] >= packets_per_node && i > 0);
    return (bool)i;
}

static void shuffle(int *array, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

static int get_children(Edge *edges, int edges_count, int parent_node, int *children, int children_count) {
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

static int get_children_transmit_intents(int *children_transmit_intents, int children_transmit_intents_count, int *children, int children_count, bool *transmit_intent) {
    children_transmit_intents_count = 0;
    memset(children_transmit_intents, false, MAX_CHILDREN);

    for (int i = 0; i < children_count; i++) {
        if (transmit_intent[children[i]] == true) {
            children_transmit_intents[children_transmit_intents_count++] = children[i];
        }
    }

    return children_transmit_intents_count;
}

static bool transmission_success(float rssi) {
    if ((float)rand()/(float)RAND_MAX <= rssi) {
        return true;
    }

    return false;
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
                            float rssi = adj_matrix[parent][transmitting_child];

                            if (transmission_success(rssi)) {
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
                            float rssi = adj_matrix[parent][child];

                            if (transmission_success(rssi) && packets[child] < packets_per_node) {
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