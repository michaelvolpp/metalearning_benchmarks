from typing import Optional, Tuple

import numpy as np

from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark


def collate_benchmark(benchmark: MetaLearningBenchmark):
    x = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x),
        dtype=np.float32,
    )
    y = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y),
        dtype=np.float32,
    )
    for l, task in enumerate(benchmark):
        x[l] = task.x
        y[l] = task.y

    return x, y
