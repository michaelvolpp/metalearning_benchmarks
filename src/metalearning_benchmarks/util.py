from typing import Optional, Tuple

import numpy as np

from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark


def collate_benchmark(
    benchmark: MetaLearningBenchmark, add_noise: Optional[bool] = True
):
    x = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x),
        dtype=np.float32,
    )
    y = np.zeros(
        (benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y),
        dtype=np.float32,
    )

    if add_noise:
        for l, task in enumerate(benchmark):
            x[l] = task.x
            y[l] = task.y
    else:
        for l, task in enumerate(benchmark._iter_without_noise()):
            x[l] = task.x
            y[l] = task.y

    return x, y
