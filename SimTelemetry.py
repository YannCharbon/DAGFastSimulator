"""
Author: Eric Tran <eric.tran@heig-vd.ch>
Author: Yann Charbon <yann.charbon@heig-vd.ch>
"""
from concurrent.futures import ProcessPoolExecutor
import glob
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
        data:nx.MultiDiGraph = nx.read_edgelist(filepath, delimiter=',', create_using=nx.MultiDiGraph, nodetype=int)

        # Collect all edge_types except 'topology'
        dag_types = {
            attr['edge_type'] for _, _, attr in data.edges(data=True)
            if attr['edge_type'] != 'topology'
        }

        number_of_dags = len(dag_types)

        self.dags = [nx.DiGraph() for _ in range(number_of_dags)]

        self.tpg = nx.DiGraph()
        self.tpg.add_nodes_from(data.nodes)

        for u,v,k, labels in data.edges(keys=True, data=True): # k = 0 topology, k = 1 dag
            link_quality = labels.get('link_quality')
            if k == 0:
                self.tpg.add_edge(u, v, link_quality=link_quality)
            else:
                dag_idx = int(labels.get('edge_type').split('_')[1])
                self.dags[dag_idx].add_edge(u, v, link_quality=link_quality)

        self.dags_nodes_telemetry = []
        for i in range(0, number_of_dags):
            self.dags_nodes_telemetry.append({node_id: NodeTelemetry() if node_id != root_id else NodeTelemetry(EnumNodeType.BR) for node_id in self.dags[i].nodes})

            # Set telemetries data for nodes
            for node in self.dags[i].nodes:
                self.dags_nodes_telemetry[i][node].rank = nx.shortest_path_length(self.dags[i], target=node, source=root_id) if node != root_id else 0
                parent = list(self.dags[i].predecessors(node))
                if len(parent) > 0:
                    self.dags_nodes_telemetry[i][node].link_quality = self.dags[i].edges[(parent[0], node)]['link_quality']
                    self.dags_nodes_telemetry[i][node].parent = parent[0]

    def incr_fail_tx(self, dag_idx, node_id: int):
        self.dags_nodes_telemetry[dag_idx][node_id].tx_fail +=1

    def incr_success_tx(self, dag_idx, node_id: int):
        self.dags_nodes_telemetry[dag_idx][node_id].tx_success +=1

    def incr_retransmissions(self, dag_idx, node_id: int):
        self.dags_nodes_telemetry[dag_idx][node_id].retransmissions +=1

    def run_simulation(self):
        #self.simulate_uplink()
        #self.simulate_downlink()
        for i in range(0, len(self.dags)):
            self.simulate(i)

    def generate_report(self, filename: str):
        # Start with generic columns (node_id, node_type)
        df_report = pd.DataFrame(columns=['node_id', 'node_type'])

        # Iterate over all DAGs
        for dag_id, dag_nodes_telemetry in enumerate(self.dags_nodes_telemetry):
            # Add DAG-specific columns
            dag_columns = [
                f"rank_dag{dag_id}",
                f"parent_dag{dag_id}",
                f"link_quality_dag{dag_id}",
                f"tx_success_dag{dag_id}",
                f"tx_failure_dag{dag_id}",
                f"retransmissions_dag{dag_id}",
                f"neighbors_dag{dag_id}",
            ]
            for column in dag_columns:
                if column not in df_report.columns:
                    df_report[column] = None

            # Populate rows for each node in the current DAG
            for node_id, node_tm in dag_nodes_telemetry.items():
                # Ensure the node is added to the dataframe
                if node_id not in df_report['node_id'].values:
                    df_report = pd.concat(
                        [df_report, pd.DataFrame({'node_id': [node_id], 'node_type': [node_tm.node_type]})],
                        ignore_index=True,
                    )

                # Find the row corresponding to the node
                row_index = df_report[df_report['node_id'] == node_id].index[0]

                # Generate neighbor information
                neighbors = ""
                for ngh_id in self.tpg.neighbors(node_id):
                    link_quality = self.tpg.edges[(node_id, ngh_id)].get('link_quality')
                    neighbors += f"{ngh_id}:{link_quality};"
                neighbors = neighbors[:-1]  # Remove last separator

                # Update the row with DAG-specific data
                df_report.loc[row_index, f"rank_dag{dag_id}"] = node_tm.rank
                df_report.loc[row_index, f"parent_dag{dag_id}"] = node_tm.parent
                df_report.loc[row_index, f"link_quality_dag{dag_id}"] = node_tm.link_quality
                df_report.loc[row_index, f"tx_success_dag{dag_id}"] = node_tm.tx_success
                df_report.loc[row_index, f"tx_failure_dag{dag_id}"] = node_tm.tx_fail
                df_report.loc[row_index, f"retransmissions_dag{dag_id}"] = node_tm.retransmissions
                df_report.loc[row_index, f"neighbors_dag{dag_id}"] = neighbors

        # Sort by node_id and save to CSV
        df_report.sort_values(by='node_id').to_csv(filename, index=None)



    def simulate_uplink(self, dag_idx, epoch_len=2, packets_per_node=15, max_steps=-1):
        n = len(self.dags[dag_idx].nodes)

        packets = dict(self.dags[dag_idx].nodes)
        transmit_intent = dict(self.dags[dag_idx].nodes)
        transmitting = dict(self.dags[dag_idx].nodes)
        total_packets = (n - 1) * packets_per_node
        for node in self.dags[dag_idx].nodes:
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

                transmit_intent = {node: packets[node] > 0 for node in self.dags[dag_idx].nodes}  # Transmission intentions
                #for node in G.nodes:
                #    transmit_intent[node] = packets[node] > 0

                # Process packet transmission for all nodes
                r = list(range(0, len(self.dags[dag_idx].nodes)))
                np.random.shuffle(r)
                for i in r:
                    parent = list(self.dags[dag_idx].nodes)[i]

                    children = list(self.dags[dag_idx].successors(parent))
                    transmitting_child = -1
                    children_transmit_intents = [child for child in children if transmit_intent[child] == True]
                    if children_transmit_intents != []:
                        transmitting_child = random.choice(children_transmit_intents)

                    if transmitting_child >= 0:
                        if transmitting[parent] == True:
                            self.incr_retransmissions(dag_idx, transmitting_child)
                            continue
                        # Determine if transmission is successful based on Link Quality
                        link_quality = self.dags[dag_idx].edges[(parent,transmitting_child)]['link_quality']  # Get Link Quality value for the link
                        transmission_success = random.random() <= link_quality

                        if transmission_success:
                            self.incr_success_tx(dag_idx, transmitting_child)
                            packets[parent] += 1
                            packets[transmitting_child] -= 1
                            if parent == 0:
                                total_packets -= 1
                        else:
                            self.incr_fail_tx(dag_idx, transmitting_child)

                        transmitting[transmitting_child] = True

                # Reset the transmitting and receiving status for the next step
                transmitting = {node: False for node in self.dags[dag_idx].nodes}
                # Root node never holds an packet
                packets[0] = 0

            avg_steps += steps
        return avg_steps / epoch_len


    def simulate_downlink(self, dag_idx, epoch_len=10, packets_per_node=15, max_steps=-1):
        packets = dict(self.dags[dag_idx].nodes)
        transmitting = dict(self.dags[dag_idx].nodes)  # Track which nodes are currently transmitting
        all_nodes_not_full = [True * len(self.dags[dag_idx].nodes)]
        for node in self.dags[dag_idx].nodes:
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
                r = list(range(0, len(self.dags[dag_idx].nodes)))
                np.random.shuffle(r)
                for i in r:
                    node = list(self.dags[dag_idx].nodes)[i]
                    if packets[node] > 0 and not transmitting[node]:  # Node has packets to send and is not already transmitting
                        children = list(self.dags[dag_idx].successors(node))
                        if children:
                            # Choose a child randomly to try to send the packet to
                            child = random.choice(children)

                            if not transmitting[child]:  # Check if the child is not currently transmitting
                                # Determine if transmission is successful based on Link Quality
                                link_quality = self.dags[dag_idx].edges[(node, child)]['link_quality']  # Get Link Quality value for the link
                                transmission_success = random.random() <= link_quality

                                if transmission_success and packets[child] < packets_per_node:
                                    packets[child] += 1
                                    packets[node] -= 1
                                    transmitting[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision
                                    self.incr_success_tx(dag_idx, node)
                                elif packets[child] < packets_per_node: # We don't want to incremente failed tx if all packets have been received by the child node
                                    self.incr_fail_tx(dag_idx, node)
                            else:
                                # Retranssmissions
                                self.incr_retransmissions(dag_idx, node)

                # Reset the transmitting status for the next step
                transmitting = {node: False for node in self.dags[dag_idx].nodes}
                all_nodes_not_full = [packets[node] < packets_per_node for node in self.dags[dag_idx].nodes]

            avg_steps += steps
        return avg_steps / epoch_len

    """
    This method combines the UP and DOWN simulations into a single simulation
    """
    def simulate(self, dag_idx, epoch_len=10, packets_per_node=15, max_steps=-1):
        n = len(self.dags[dag_idx].nodes)

        packets_up = {node: packets_per_node for node in self.dags[dag_idx].nodes}
        packets_down = {node: 0 for node in self.dags[dag_idx].nodes}
        transmit_intent_up = {node: True for node in self.dags[dag_idx].nodes}  # Transmission intentions
        transmitting = {node: False for node in self.dags[dag_idx].nodes}  # Track which nodes are currently transmitting

        avg_steps = 0
        early_stop = False
        for epoch in range(epoch_len):
            if epoch != 0:
                for i in range(0, len(packets_up)):
                    # This is done to optimize execution time
                    packets_up[i] = packets_per_node
                    packets_down[i] = 0
                    transmitting[i] = False
                    transmit_intent_up[i] = True

            steps = 0

            up_finished = False
            down_finished = False
            while not up_finished or not down_finished:
                steps += 1

                # Root node Root node has a new packet to insert into the network. Does not send it yet.
                if packets_down[0] < packets_per_node:
                    packets_down[0] += 1

                transmit_intent_up = {node: packets_up[node] > 0 for node in self.dags[dag_idx].nodes}  # Transmission intentions

                # Process packet transmission for all nodes
                r = list(range(0, len(self.dags[dag_idx].nodes)))
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
                        parent = list(self.dags[dag_idx].nodes)[i]

                        children = list(self.dags[dag_idx].successors(parent))
                        transmitting_child = -1
                        children_transmit_intents = [child for child in children if transmit_intent_up[child] == True]
                        if children_transmit_intents != []:
                            transmitting_child = random.choice(children_transmit_intents)

                        if transmitting_child >= 0:
                            if transmitting[parent] == True:
                                self.incr_retransmissions(dag_idx, transmitting_child)
                                continue
                            # Determine if transmission is successful based on Link Quality
                            link_quality = self.dags[dag_idx].edges[(parent,transmitting_child)]['link_quality']  # Get Link Quality value for the link
                            transmission_success = random.random() <= link_quality

                            if transmission_success:
                                self.incr_success_tx(dag_idx, transmitting_child)
                                packets_up[parent] += 1
                                packets_up[transmitting_child] -= 1
                                transmit_intent_up[transmitting_child] = False
                                transmitting[parent] = True # Parent is not actually transmitting, but it is busy while receiving from child
                            else:
                                self.incr_fail_tx(dag_idx, transmitting_child)

                            transmitting[transmitting_child] = True

                    else:   # DOWN
                        node = list(self.dags[dag_idx].nodes)[i]
                        if packets_down[node] > 0 and not transmitting[node]:  # Node has packets to send and is not already transmitting
                            children = list(self.dags[dag_idx].successors(node))
                            if children:
                                # Choose a child randomly to try to send the packet to
                                child = random.choice(children)

                                if not transmitting[child]:  # Check if the child is not currently transmitting
                                    # Determine if transmission is successful based on Link Quality
                                    link_quality = self.dags[dag_idx].edges[(node, child)]['link_quality']  # Get Link Quality value for the link
                                    transmission_success = random.random() <= link_quality

                                    if transmission_success and packets_down[child] < packets_per_node:
                                        packets_down[child] += 1
                                        packets_down[node] -= 1
                                        transmitting[child] = True  # Mark the child as transmitting. It is also the case when transmission success is false because it simulates a collision
                                        self.incr_success_tx(dag_idx, node)

                                    elif packets_down[child] < packets_per_node: # We don't want to incremente failed tx if all packets have been received by the child node
                                        self.incr_fail_tx(dag_idx, node)

                                    transmitting[node] = True # Mark the parent as transmitting. It is also the case when transmission success is false because it simulates a collision by the fact the child is busy.
                                else:
                                    self.incr_retransmissions(dag_idx, node)


                # Reset the transmitting and receiving status for the next step
                transmitting = {node: False for node in self.dags[dag_idx].nodes}
                # Root node never holds an packet
                packets_up[0] = 0

                if max_steps != -1 and steps > max_steps:
                    early_stop = True
                    break

                # Update end conditions
                up_finished = not any(packets_up[node] > 0 for node in self.dags[dag_idx].nodes)
                down_finished = not any(packets_down[node] < packets_per_node for node in self.dags[dag_idx].nodes)

            avg_steps += steps

            if early_stop == True:
                break

        if not early_stop:
            return int(avg_steps / epoch_len)
        else:
            return max_steps

    @staticmethod
    def process_topology(args):
        topology, simulation_path = args
        sim = SimTelemetry()
        sim.load_topology(topology)
        simulation_path = Path(simulation_path)
        simulation_path.mkdir(exist_ok=True)
        output_file = simulation_path / Path(topology.split('/')[-1].replace('.csv', '_simulation.csv'))
        sim.run_simulation()
        sim.generate_report(output_file)

    @classmethod
    def run_parallel_simulations(cls, topology_folder: str = 'dags', output_sim_folder: str = 'simulation'):
        print("Run simulation with telemetry")

        # Get the list of all topology files
        topologies = glob.glob(f'{topology_folder}/*.csv')

        print(f"Running simulations for {len(topologies)} topologies in parallel...")

        # Prepare arguments for each topology
        args = [(topology, output_sim_folder) for topology in topologies]

        # Use ProcessPoolExecutor to parallelize the processing
        with ProcessPoolExecutor() as executor:
            executor.map(cls.process_topology, args)

        print("Simulations completed.")

class EnumNodeType(str, Enum):
    BR = 'BR'
    NODE = 'NODE'

class NodeTelemetry:
    def __init__(self, nodeType : EnumNodeType = EnumNodeType.NODE):
        self.node_type = nodeType
        self.parent = -1
        self.link_quality = -1
        self.rank = -1
        self.tx_fail = 0
        self.tx_success = 0
        self.retransmissions = 0

if __name__ == '__main__':
    sim = SimTelemetry()
    sim.load_topology('dags/topologies_2024-11-12 10_10_03.226982_best_dag.csv')
    sim.run_simulation()
    simulation_path = Path('simulation')
    simulation_path.mkdir(exist_ok=True)
    sim.generate_report(simulation_path / Path(sim.filepath.split('/')[-1].replace('.csv', '_simulation.csv')))