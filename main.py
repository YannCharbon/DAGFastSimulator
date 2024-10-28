from DAGDatasetGenerator import DAGDatasetGenerator
import matplotlib.pyplot as plt
import os
from SimTelemetry import SimTelemetry
from pathlib import Path
import glob

if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()

    #best_dag, best_perf, adj_matrix = generator.run_once(10)
    #print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
    #
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)

    #generator.run(10, 10)
    generator.run_parallel(10, 4, int(os.cpu_count()/2))
    # To keep plots open
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)
    #plt.ioff()
    #plt.show()
    """
    print("Run simulation with telemetry")
    simulation_path = Path('simulation')
    simulation_path.mkdir(exist_ok=True)
    for topology in glob.glob('./dags/*.csv'):
        sim = SimTelemetry()
        sim.load_topology(topology)
        sim.run_simulation()
        sim.generate_report(simulation_path / Path(sim.filepath.split('/')[-1].replace('.csv', '_simulation.csv')))
    """