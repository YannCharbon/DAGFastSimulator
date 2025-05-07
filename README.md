# DAG Fast Simulator and dataset generator

This repository contains a fast mesh network simulator which is able to take a topology (adjacency matrix) as an input and compute one of the best performing DAG structure  in terms of routing, packet delivery ratio and bottlenecks.

It can be used to produce datasets with telemetry information.

# Installation

First of all, you must compile the C/C++ core of the simulator. The compilation must be performed on the target machine in order for the compiler to optimize the code for it. This core is written in C/C++ to maximum execution speed compred to python equivalent.

Before first compilation, please install `libgmp` once for all with `sudo apt install libgmp-dev`. Then compile the code as a library. You should adjust `DMAX_NODES` to match the maximum number of nodes of your topologies and `DMAX_CHILDREN` to match the maximum number of children for each node that can be handled by the simulator. These values are set at compilation time to avoid having to deal with internal dynamic memory allocation which is succeptible to slow down the execution speed.

```bash
cd CDAGOperation/
g++ -DMAX_NODES=33 -DMAX_CHILDREN=5 -DLOG_TIMINGS=0 -DVERBOSE=0 -shared -o libCDAGOperation.so -fPIC -Ofast -flto CDAGOperation.cpp  -lgmp
```

After this compilation step has been performed, the fast simulator and dataset generator are ready to be used.

# Usage examples
## Base features

The `DAGDatasetGenerator` can be used for two distinguished tasks: generating all/a subset of all spanning trees that can be formed within a given topology (adjacency matrix), and simulating the network traffic within a spanning tree. Each method of this class are described in the `DAGDatasetGenerator.py` file.

There is an additional class called `SimTelemetry` which can be used to process the output of the `DAGDatasetGenerator` and which computes additional metrics for the generated spanning trees.

## Manually compute the performance of each DAG of a provided topology

This codes computes the performance score for all the DAGs that can be formed within a single topology (provided through the adjacency matrix).
The adjacency matrix indicates the link quality between the nodes (`0.0` means that there is no link, `1.0` means that the link is perfect, values in between are the probability of successful packet transmission).

```python
from DAGDatasetGenerator import DAGDatasetGenerator

if __name__ == '__main__':
    generator = DAGDatasetGenerator()

    adj_matrix =    [[0.00, 0.40, 0.19, 0.00, 0.00, 0.00, 0.10],
                    [0.40, 0.00, 0.12, 0.00, 0.58, 0.00, 0.00],
                    [0.19, 0.12, 0.00, 0.00, 0.00, 0.31, 0.00],
                    [0.00, 0.00, 0.00, 0.09, 0.15, 0.02, 0.00],
                    [0.00, 0.58, 0.00, 0.15, 0.00, 0.22, 0.00],
                    [0.00, 0.00, 0.31, 0.02, 0.22, 0.00, 0.19],
                    [0.10, 0.00, 0.00, 0.00, 0.00, 0.19, 0.00]]

    # Generates all the possible valid tree from the topology as "no_skip" is set to True
    all_dags = generator.generate_subset_dags(adj_matrix, no_skip=True)

    perfs = []
    for dag in all_dags:
        # Simulates the network traffic through the spanning tree
        _, perf = generator.evaluate_dag_performance_double_flux(dag, adj_matrix, epoch_len=3)
        perfs.append(perf)

    print(perfs)
```

## Automatically generate a dataset for a given network size

This will compute the best DAG for 100 topologies that have 20 nodes. Detailled telemetry information can be computed in post-treatment (such as retransmissions count, TX success/failure counts, rank, etc. for each node).
In this example, the generated DAG are saved in CSV files in the "dags" directory.

```python
import os
from DAGDatasetGenerator import DAGDatasetGenerator
from SimTelemetry import SimTelemetry

if __name__ == '__main__':
    generator = DAGDatasetGenerator()
    telemetry = SimTelemetry()

    generator.run_double_flux(20, 100, int(os.cpu_count()), verbose=False, dags_folder_path="dags")
    telemetry.run_parallel_simulations("dags", "simulation")
```

