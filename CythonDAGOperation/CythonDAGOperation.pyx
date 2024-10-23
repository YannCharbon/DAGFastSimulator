import networkx
import random
import time
import numpy as np
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
cpdef cython_evaluate_dag_performance_up(G, adj_matrix, int epoch_len=2, int packets_per_node=15, int max_steps=-1):
    cdef int n = len(G.nodes)
    #packets = {node: packets_per_node for node in G.nodes}
    #transmit_intent = {node: True for node in G.nodes}  # Transmission intentions
    #transmitting = {node: False for node in G.nodes}  # Track which nodes are currently transmitting
    
    cdef dict packets = dict(G.nodes)
    cdef dict transmit_intent = dict(G.nodes)
    cdef dict transmitting = dict(G.nodes)
    cdef int total_packets = (n - 1) * packets_per_node
    for node in G.nodes:
        packets[node] = packets_per_node
        transmit_intent[node] = True
        transmitting[node] = False
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
        while total_packets > 0:
            steps += 1
            
            transmit_intent = {node: packets[node] > 0 for node in G.nodes}  # Transmission intentions
            #for node in G.nodes:
            #    transmit_intent[node] = packets[node] > 0

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
                        if parent == 0:
                            total_packets -= 1

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
cpdef cython_evaluate_dag_performance_down(G, adj_matrix, epoch_len=2, packets_per_node=15, max_steps=-1):
    cdef int n = len(G.nodes)

    cdef dict packets = dict(G.nodes)
    cdef dict transmitting = dict(G.nodes)  # Track which nodes are currently transmitting
    cdef int total_packets = packets_per_node * len(G.nodes)
    cdef list all_nodes_full
    for node in G.nodes:
        packets[node] = 0
        transmitting[node] = False
    avg_steps = 0
    early_stop = False
    for epoch in range(epoch_len):
        if epoch != 0:
            for i in range(0, len(packets)):
                # This is done to optimize execution time
                packets[i] = 0
                transmitting[i] = False

        steps = 0
        all_nodes_not_full = [True * len(G.nodes)]
        while any(all_nodes_not_full):
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
            all_nodes_not_full = [packets[node] < packets_per_node for node in G.nodes] 

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