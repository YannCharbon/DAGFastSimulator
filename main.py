from DAGDatasetGenerator import DAGDatasetGenerator
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()

    #best_dag, best_perf, adj_matrix = generator.run_once(10)
    #print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
    #
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)

    #generator.run(10, 10)
    generator.run_parallel(15, 10, int(os.cpu_count() )) 
    #generator.run_once_with_adaptive_steps(15)
    #generator.run_parallel_cython(10, 10, int(os.cpu_count() / 2)) 
    #generator.run_once_cython(10) 
    # To keep plots open
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)
    #plt.ioff()
    #plt.show()