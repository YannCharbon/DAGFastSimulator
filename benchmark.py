from DAGDatasetGenerator import DAGDatasetGenerator
import time
import numpy as np
import timeit
import random


def begin_banner(fn_name):
    print("------------------------------------------")
    print(f"Begin {fn_name}")

def end_banner():
    print("------------------------------------------")


def benchmark_dags_generations(num_nodes, max_iter):
    begin_banner(benchmark_dags_generations.__name__)
    generator = DAGDatasetGenerator()
    total_time = 0.
    total_time_partial= 0.
    total_time_partial_pure_c = 0.
    for i in range(max_iter):
        print(f"Iteration {i}")
        adj_matrix, density_factor = generator.generate_random_adj_matrix(num_nodes)
        # Don't compute using basic function because not suited for bigger topologies
        #start_time = time.time()
        #generator.generate_dags(adj_matrix)
        #end_time = time.time()
        #total_time += end_time - start_time
        #print(f"Run time dag generation : {total_time}")
        # Test with Generation of subset DAGs
        start_time = time.time()
        generator.generate_subset_dags(adj_matrix)
        end_time = time.time()
        total_time_partial += end_time - start_time
        print(f"Run time subset dag generation : {total_time_partial}")
        # Test with Generation of subset DAGs and pure C implementation
        start_time = time.time()
        generator.generate_subset_dags_pure_c(adj_matrix)
        end_time = time.time()
        total_time_partial_pure_c += end_time - start_time
        print(f"Run time subset dag generation and pure C: {total_time_partial_pure_c}")

    average_time = total_time / (max_iter * max_iter)
    average_time_partial = total_time_partial / (max_iter * max_iter)
    average_time_partial_pure_c = total_time_partial_pure_c / (max_iter * max_iter)

    partial_improvement_percent = (average_time - average_time_partial) / average_time * 100
    partial_pure_c_improvement_percent = (average_time_partial - average_time_partial_pure_c) / average_time_partial * 100

    print(f"Average processing time : {average_time}")
    print(f"Average processing time with partial generation: {average_time_partial}, improvement: {partial_improvement_percent}%")
    print(f"Average processing time with partial generation and pure C: {average_time_partial_pure_c}, improvement: {partial_pure_c_improvement_percent}%")
    end_banner()

def benchmark_updown(num_nodes, max_iter):
    begin_banner(benchmark_updown.__name__)
    generator = DAGDatasetGenerator()
    total_time = 0.
    total_time_cython= 0.
    nb_iter = 0
    for i in range(max_iter):
        adj_matrix, density_factor = generator.generate_random_adj_matrix(num_nodes)
        all_dags = generator.generate_subset_dags(adj_matrix)
        if not len(all_dags):
            i -= 1
            continue
        for j in range(max_iter):
            nb_iter += 1
            print(f"Iteration {nb_iter}")
            start_time = time.time()
            generator.get_best_dag_parallel_up_down(all_dags, adj_matrix)
            end_time = time.time()
            total_time += end_time - start_time
            print(f"Run time to find best DAG : {end_time - start_time}")
            # Test with Generation of subset DAGs
            start_time = time.time()
            generator.cython_get_best_dag_parallel_up_down(all_dags, adj_matrix)
            end_time = time.time()
            total_time_cython+= end_time - start_time
            print(f"Run time to find best DAG with cython : {end_time - start_time}")
    print(f"Average processing time : {total_time / (max_iter*max_iter)}")
    print(f"Average processing time with cython: {total_time_cython/ (max_iter*max_iter)}")
    end_banner()

def benchmark_updown_with_adaptative_steps(num_nodes, max_iter):
    begin_banner(benchmark_updown_with_adaptative_steps.__name__)
    generator = DAGDatasetGenerator()
    total_time = 0.
    total_time_adaptative = 0.
    nb_iter = 0
    for i in range(max_iter):
        adj_matrix, density_factor = generator.generate_random_adj_matrix(num_nodes)
        all_dags = generator.generate_subset_dags(adj_matrix)
        if not len(all_dags):
            i -= 1
            continue
        for j in range(max_iter):
            nb_iter += 1
            print(f"Iteration {nb_iter}")
            start_time = time.time()
            generator.get_best_dag_parallel_up_down(all_dags, adj_matrix)
            end_time = time.time()
            total_time += end_time - start_time
            print(f"Run time to find best DAG : {end_time - start_time}")
            # Test with Generation of subset DAGs
            start_time = time.time()
            generator.get_best_dag_parallel_with_adaptative_steps(all_dags, adj_matrix)
            end_time = time.time()
            total_time_adaptative+= end_time - start_time
            print(f"Run time to find best DAG with cython : {end_time - start_time}")
    print(f"Average processing time : {total_time / (max_iter * max_iter)}")
    print(f"Average processing time with adaptative steps: {total_time_adaptative/ (max_iter * max_iter)}")
    end_banner()

def benchmark_updown_with_adaptative_steps_pure_c(num_nodes, max_iter):
    begin_banner(benchmark_updown_with_adaptative_steps_pure_c.__name__)
    generator = DAGDatasetGenerator()
    total_time = 0.
    total_time_adaptative = 0.
    total_time_adaptative_pure_c = 0.
    nb_iter = 0
    for i in range(max_iter):
        adj_matrix, density_factor = generator.generate_random_adj_matrix(num_nodes)
        all_dags = generator.generate_subset_dags(adj_matrix)
        if not len(all_dags):
            i -= 1
            continue
        for j in range(max_iter):
            nb_iter += 1
            print(f"Iteration {nb_iter}")
            start_time = time.time()
            generator.get_best_dag_parallel_up_down(all_dags, adj_matrix)
            end_time = time.time()
            total_time += end_time - start_time
            print(f"Run time to find best DAG : {end_time - start_time}")

            # Test with Generation of subset DAGs
            start_time = time.time()
            generator.get_best_dag_parallel_with_adaptative_steps(all_dags, adj_matrix)
            end_time = time.time()
            total_time_adaptative += end_time - start_time
            print(f"Run time to find best DAG with cython : {end_time - start_time}")

            start_time = time.time()
            generator.get_best_dag_parallel_with_adaptative_steps_pure_c(all_dags, adj_matrix)
            end_time = time.time()
            total_time_adaptative_pure_c += end_time - start_time
            print(f"Run time to find best DAG with pure C : {end_time - start_time}")

    average_time = total_time / (max_iter * max_iter)
    average_time_adaptative = total_time_adaptative / (max_iter * max_iter)
    average_time_adaptative_pure_c = total_time_adaptative_pure_c / (max_iter * max_iter)

    adaptative_improvement_percent = (average_time - average_time_adaptative) / average_time * 100
    adaptative_pure_c_improvement_percent = (average_time_adaptative - average_time_adaptative_pure_c) / average_time_adaptative * 100

    print(f"Average processing time : {average_time}")
    print(f"Average processing time with adaptative steps: {average_time_adaptative}, improvement: {adaptative_improvement_percent}%")
    print(f"Average processing time with adaptative steps and pure C: {average_time_adaptative_pure_c}, improvement: {adaptative_pure_c_improvement_percent}%")
    end_banner()

## Benchmark numpy
def benchmark_shuffle(num_nodes, max_iter):
    begin_banner(benchmark_shuffle.__name__)
    list_test = list(range(num_nodes))
    random_total_time = 0.
    np_total_time = 0.
    for i in range(max_iter):
        start_time = time.time()
        random.shuffle(list_test)
        end_time = time.time()
        random_total_time += end_time - start_time
        print(f"Time for random shuffle : {end_time - start_time}")
    for i in range(max_iter):
        start_time = time.time()
        np.random.shuffle(list_test)
        end_time = time.time()
        np_total_time += end_time - start_time
        print(f"Time for numpy random shuffle : {end_time - start_time}")

    print(f"Average random shuffle : {random_total_time / max_iter}")
    print(f"Average numpy random shuffle : {np_total_time / max_iter}")
    end_banner()

def benchmark_choice(num_nodes, max_iter):
    begin_banner(benchmark_choice.__name__)
    list_test = list(range(num_nodes))
    random_total_time = 0.
    np_total_time = 0.
    for i in range(max_iter):
        start_time = time.time()
        random.choice(list_test)
        end_time = time.time()
        random_total_time += end_time - start_time
        print(f"Time for random choice: {end_time - start_time}")
    for i in range(max_iter):
        start_time = time.time()
        np.random.choice(list_test)
        end_time = time.time()
        np_total_time += end_time - start_time
        print(f"Time for numpy random choice : {end_time - start_time}")

    print(f"Average random choice : {random_total_time / max_iter}")
    print(f"Average numpy random choice : {np_total_time / max_iter}")
    end_banner()

def benchmark_random_values(max_iter):
    begin_banner(benchmark_random_values.__name__)
    random_total_time = 0.
    np_total_time = 0.
    for i in range(max_iter):
        start_time = time.time()
        random.random()
        end_time = time.time()
        random_total_time += end_time - start_time
        print(f"Time for random value generation : {end_time - start_time}")
    for i in range(max_iter):
        start_time = time.time()
        np.random.random()
        end_time = time.time()
        np_total_time += end_time - start_time
        print(f"Time for numpy random generation : {end_time - start_time}")

    print(f"Total random generation : {random_total_time}")
    print(f"Total numpy random generation : {np_total_time}")
    end_banner()

if __name__ == '__main__':
    generator = DAGDatasetGenerator()
    MAX_ITER = 5
    NUM_NODES = 15

    # Benchmark DAG generator class
    benchmark_dags_generations(NUM_NODES, MAX_ITER)
    #benchmark_updown(NUM_NODES, MAX_ITER)
    #benchmark_updown_with_adaptative_steps(NUM_NODES, MAX_ITER)
    #benchmark_updown_with_adaptative_steps_pure_c(NUM_NODES, MAX_ITER)

    # Benchmark numpy vs random package
    #benchmark_shuffle(NUM_NODES, MAX_ITER)
    #benchmark_choice(NUM_NODES, MAX_ITER)
    #benchmark_random_values(MAX_ITER)
