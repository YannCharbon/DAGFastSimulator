import DAGDatasetGenerator
import networkx as nx
import numpy as np
import random
import pandas as pd
from enum import Enum
from pathlib import Path
class SimTelemetry:
    def __init__(self):
        pass

    def load_topology(self, filepath: str, root_id:int = 0):
        self.filepath = filepath
        self.dag = nx.read_edgelist(filepath, delimiter=',', create_using=nx.DiGraph, nodetype=int)
        self.nodes_telemetry = {node_id: NodeTelemetry() if node_id != root_id else NodeTelemetry(EnumNodeType.BR) for node_id in self.dag.nodes}
        # Set telemetries data for nodes
        for node in self.dag.nodes:
            self.nodes_telemetry[node].rank = nx.shortest_path_length(self.dag, target=node, source=root_id) if node != root_id else 0
            parent = list(self.dag.predecessors(node))
            if len(parent) > 0:
                self.nodes_telemetry[node].rssi = self.dag.edges[(parent[0], node)]['rssi']
                self.nodes_telemetry[node].parent = parent[0]

    def incr_fail_tx(self, node_id: int):
        self.nodes_telemetry[node_id].tx_fail +=1

    def incr_success_tx(self, node_id: int):
        self.nodes_telemetry[node_id].tx_success +=1

    def incr_collision_avoided(self, node_id: int):
        self.nodes_telemetry[node_id].collision_avoided +=1

    def set_node_rank(self, node_id: int, rank: int):
        self.nodes_telemetry[node_id].rank = rank

    def run_simulation(self):
        self.simulate_uplink()
        self.simulate_downlink()

    def generate_report(self, filename: str):
        df_report = pd.DataFrame(columns=['node_id', 'node_type', 'rank', 'parent', 'rssi', 'tx_success', 'tx_failure', 'collision_avoided'])
        for node_id, node_tm in self.nodes_telemetry.items():
            df_report.loc[len(df_report)] = [node_id, node_tm.node_type, node_tm.rank, node_tm.parent, node_tm.rssi, node_tm.tx_success, node_tm.tx_fail, node_tm.collision_avoided]
        df_report.sort_values(by='node_id').to_csv(filename, index=None)

    def simulate_uplink(self, epoch_len=2, packets_per_node=15, max_steps=-1):
        n = len(self.dag.nodes)

        packets = dict(self.dag.nodes)
        transmit_intent = dict(self.dag.nodes)
        transmitting = dict(self.dag.nodes)
        total_packets = (n - 1) * packets_per_node
        for node in self.dag.nodes:
            packets[node] = packets_per_node
            transmit_intent[node] = True
            transmitting[node] = False
        avg_steps = 0

        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = packets_per_node
                    transmitting[i] = False
                    transmit_intent[i] = True

            steps = 0
            while total_packets > 0:
                steps += 1

                transmit_intent = {node: packets[node] > 0 for node in self.dag.nodes}  # Transmission intentions
                #for node in G.nodes:
                #    transmit_intent[node] = packets[node] > 0

                # Process packet transmission for all nodes
                r = list(range(0, len(self.dag.nodes)))
                np.random.shuffle(r)
                for i in r:
                    parent = list(self.dag.nodes)[i]
                    # TODO: Increment collision_avoided for a selected child node
                    if transmitting[parent] == True:
                        continue

                    children = list(self.dag.successors(parent))
                    transmitting_child = -1
                    children_transmit_intents = [child for child in children if transmit_intent[child] == True]
                    if children_transmit_intents != []:
                        transmitting_child = random.choice(children_transmit_intents)

                    if transmitting_child >= 0:
                        # Determine if transmission is successful based on RSSI
                        rssi = self.dag.edges[(parent,transmitting_child)]['rssi']  # Get RSSI value for the link
                        transmission_success = random.random() <= rssi

                        if transmission_success:
                            self.incr_success_tx(transmitting_child)
                            packets[parent] += 1
                            packets[transmitting_child] -= 1
                            if parent == 0:
                                total_packets -= 1
                        else:
                            self.incr_fail_tx(transmitting_child)

                        transmitting[transmitting_child] = True

                # Reset the transmitting and receiving status for the next step
                transmitting = {node: False for node in self.dag.nodes}
                # Root node never holds an packet
                packets[0] = 0

            avg_steps += steps
        return avg_steps / epoch_len


    def simulate_downlink(self, epoch_len=2, packets_per_node=15, max_steps=-1):
        packets = dict(self.dag.nodes)
        transmitting = dict(self.dag.nodes)  # Track which nodes are currently transmitting
        all_nodes_not_full = [True * len(self.dag.nodes)]
        for node in self.dag.nodes:
            packets[node] = 0
            transmitting[node] = False
        avg_steps = 0
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets)):
                    # This is done to optimize execution time
                    packets[i] = 0
                    transmitting[i] = False

            steps = 0
            while any(all_nodes_not_full):
                steps += 1

                # Root node inserts a packet into the network if it's not transmitting
                if packets[0] < packets_per_node and not transmitting[0]:
                    packets[0] += 1

                # Process packet transmission for all nodes
                r = list(range(0, len(self.dag.nodes)))
                np.random.shuffle(r)
                for i in r:
                    node = list(self.dag.nodes)[i]
                    if packets[node] > 0 and not transmitting[node]:  # Node has packets to send and is not already transmitting
                        children = list(self.dag.successors(node))
                        if children:
                            # Choose a child randomly to try to send the packet to
                            child = random.choice(children)

                            if not transmitting[child]:  # Check if the child is not currently transmitting
                                # Determine if transmission is successful based on RSSI
                                rssi = self.dag.edges[(node, child)]['rssi']  # Get RSSI value for the link
                                transmission_success = random.random() <= rssi

                                if transmission_success and packets[child] < packets_per_node:
                                    packets[child] += 1
                                    packets[node] -= 1
                                    transmitting[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision
                                    self.incr_success_tx(node)
                                elif packets[child] < packets_per_node: # We don't want to incremente failed tx if all packets have been received by the child node
                                    self.incr_fail_tx(node)
                            else:
                                # Collision avoided
                                self.incr_collision_avoided(node)

                # Reset the transmitting status for the next step
                transmitting = {node: False for node in self.dag.nodes}
                all_nodes_not_full = [packets[node] < packets_per_node for node in self.dag.nodes]

            avg_steps += steps
        return avg_steps / epoch_len


class EnumNodeType(str, Enum):
    BR = 'BR'
    NODE = 'NODE'

class NodeTelemetry:
    def __init__(self, nodeType : EnumNodeType = EnumNodeType.NODE):
        self.node_type = nodeType
        self.parent = -1
        self.rssi = -1
        self.rank = -1
        self.tx_fail = 0
        self.tx_success = 0
        self.collision_avoided = 0

if __name__ == '__main__':
    sim = SimTelemetry()
    sim.load_topology('topologies_perf_2024-10-21 15_00_42.782489.csv')
    sim.run_simulation()
    simulation_path = Path('simulation')
    simulation_path.mkdir(exist_ok=True)

    sim.generate_report(simulation_path / Path(sim.filepath.replace('.csv', '_simulation.csv')))