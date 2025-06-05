"""
Author: Yann Charbon <yann.charbon@heig-vd.ch>
Author: Eric Tran <eric.tran@heig-vd.ch>
"""

import os
from DAGDatasetGenerator import DAGDatasetGenerator
from SimTelemetry import SimTelemetry

times_generate_dag = list()
times_simulate_com = list()
if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()
    """
    telemetry = SimTelemetry()
    for n_nodes in range(5, 22):
        generator.run_double_flux(n_nodes, 1000, keep_dags_count=1, keep_random_dags=False, max_workers=int(os.cpu_count()), verbose=False, dags_folder_path="dags")
    telemetry.run_parallel_simulations("dags", "simulation")"
    """
    for n_nodes in range(23, 30):
        generator.run_double_flux(n_nodes, 1000, keep_dags_count=1, keep_random_dags=False, max_workers=int(os.cpu_count()), verbose=False, dags_folder_path="tmp")