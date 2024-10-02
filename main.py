from DAGDatasetGenerator import DAGDatasetGenerator
import numpy as np
import matplotlib.pyplot as plt
import datetime

if __name__ == '__main__':
    print("Starting")
    generator = DAGDatasetGenerator()

    #best_dag, best_perf, adj_matrix = generator.run_once(10)
    #print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
#
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)

    generator.run(6, 10000)

    # To keep plots open
    plt.ioff()
    plt.show()