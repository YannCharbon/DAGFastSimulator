# DAG Fast Simulator and dataset generator

This repository contains a fast mesh network simulator which is able to take a topology (adjacency matrix) as an input and compute one of the best performing DAG structure  in terms of routing, packet delivery ratio and bottlenecks.

It can be used to produce datasets with telemetry information.

# Usage examples
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

    all_dags = generator.generate_subset_dags(adj_matrix, no_skip=True)

    perfs = []
    for dag in all_dags:
        _, perf = generator.evaluate_dag_performance_double_flux(dag, adj_matrix, epoch_len=3)
        perfs.append(perf)

    print(perfs)
```

## Automatically generate a dataset for a given network size

This will compute the best DAG for 100 topologies that have 20 nodes. Detailled telemetry information can be computed in post-treatment (such as retransmissions count, TX success/failure counts, rank, etc. for each node).

```python
import os
from DAGDatasetGenerator import DAGDatasetGenerator
from SimTelemetry import SimTelemetry

if __name__ == '__main__':
    generator = DAGDatasetGenerator()
    telemetry = SimTelemetry()

    generator.run_double_flux(20, 100, int(os.cpu_count()), verbose=False)
    telemetry.run_parallel_simulations("dags", "simulation")
```

