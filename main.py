from DAGDatasetGenerator import DAGDatasetGenerator
import numpy as np
import matplotlib.pyplot as plt
import datetime
import concurrent.futures
import time
import os

def run_parallel(generator, n, count):
    start_time = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(generator.run_once, n) for i in range(count)}
        i = 0
        filename = "topologies_perf_{}.txt".format(datetime.datetime.now()).replace(":", "_")
        f = open(filename, "a")
        for future in concurrent.futures.as_completed(futures):
            best_dag, best_perf, adj_matrix = future.result()
            futures.remove(future)
            i += 1
            print(f"End of run : {i}")
            f.write(str((adj_matrix, best_dag.edges())) + '\n')
        f.close()

    end_time = time.time()
    print(f"Total runtime : {end_time - start_time}")

if __name__ == '__main__':

    print("Starting")
    generator = DAGDatasetGenerator()

    #best_dag, best_perf, adj_matrix = generator.run_once(10)
    #print("best dag is {} perf = {}".format(best_dag.edges, best_perf))
    #
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)

    #generator.run(6, 10000)
    run_parallel(generator, 6, 100) 
    
    # To keep plots open
    #generator.draw_network(adj_matrix)
    #generator.draw_dag(best_dag, adj_matrix)
    #plt.ioff()
    #plt.show()