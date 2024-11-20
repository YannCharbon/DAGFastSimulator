"""
Author: Yann Charbon <yann.charbon@heig-vd.ch>
Author: Eric Tran <eric.tran@heig-vd.ch>
"""

import os
from DAGDatasetGenerator import DAGDatasetGenerator
from SimTelemetry import SimTelemetry

if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()
    telemetry = SimTelemetry()

    generator.run_double_flux(20, 100, int(os.cpu_count()), verbose=False)
    telemetry.run_parallel_simulations("dags", "simulation")