from DAGDatasetGenerator import DAGDatasetGenerator
import matplotlib.pyplot as plt
import os
from SimTelemetry import SimTelemetry
from pathlib import Path
import glob

if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()
    telemetry = SimTelemetry()

    #best_dag, best_perf, adj_matrix = generator.run_once(10)
    #print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
    #
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)

    #generator.run(10, 10)
    #generator.run_parallel(10, 4, int(os.cpu_count()/2))
    #generator.run_parallel(20, 10, int(os.cpu_count()))
    #generator.run_parallel_pure_c(15, 200, int(os.cpu_count()))
    #generator.run_parallel_double_flux_pure_c(18, 50, int(os.cpu_count()))
    generator.run_parallel_double_flux_pure_c(20, 50, int(os.cpu_count()), verbose=False)
    # To keep plots open
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)
    #plt.ioff()
    #plt.show()

    telemetry.run_parallel_simulations("dags", "simulation")