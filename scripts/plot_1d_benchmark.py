from typing import List

import matplotlib.pyplot as plt
import numpy as np

from metalearning_benchmarks import (
    RBFGPBenchmark,
    WeaklyPeriodicGPBenchmark,
    Matern52GPBenchmark,
    Quadratic1D,
    Linear1D,
    Affine1D,
    Sinusoid,
)
from metalearning_benchmarks.base_benchmark import MetaLearningBenchmark
from metalearning_benchmarks.util import normalize_benchmark


def plot_benchmark(benchmarks: List[MetaLearningBenchmark]):
    fig, axes = plt.subplots(
        nrows=1,
        ncols=len(benchmarks),
        sharey=False,
        sharex=False,
        figsize=(len(benchmarks) * 5, 5),
    )
    for i, benchmark in enumerate(benchmarks):
        assert benchmark.d_x == benchmark.d_y == 1
        ax = axes[i] if len(benchmarks) > 1 else axes
        for task in benchmark:
            sort_idx = np.argsort(task.x.squeeze())
            x, y = task.x.squeeze()[sort_idx], task.y.squeeze()[sort_idx]
            ax.plot(x, y)
            ax.grid()
            ax.set_title(type(benchmark).__name__)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    n_task = 5
    n_datapoints_per_task = 128
    benchmarks = []
    benchmarks.append(
        Sinusoid(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=0.1,
            seed_noise=1234,
            seed_task=2234,
            seed_x=3234,
        )
    )
    benchmarks.append(
        Affine1D(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=0.1,
            seed_noise=1234,
            seed_task=2234,
            seed_x=3234,
        ),
    )
    benchmarks.append(
        RBFGPBenchmark(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=0.1,
            seed_noise=1234,
            seed_task=2234,
            seed_x=3234,
        ),
    )
    plot_benchmark(benchmarks=benchmarks)
