from typing import Optional, Tuple

import numpy as np

from metalearning_benchmarks.base_benchmark import (
    MetaLearningTask,
    MetaLearningBenchmark,
)
from metalearning_benchmarks.list_of_tasks_benchmark import (
    ListOfTasksBenchmark,
)


def split_task(
    task: MetaLearningTask,
    n_context: int,
    rng: Optional[np.random.RandomState] = None,
) -> Tuple[MetaLearningTask, MetaLearningTask]:
    assert 0 <= n_context <= task.n_datapoints

    idx = np.arange(task.n_datapoints)
    if rng is not None:
        rng.shuffle(idx)
    idx_context = idx[:n_context]
    idx_target = idx[n_context:]

    task_context = MetaLearningTask(
        x=task.x[idx_context], y=task.y[idx_context], param=task.param
    )
    task_target = MetaLearningTask(
        x=task.x[idx_target], y=task.y[idx_target], param=task.param
    )

    return task_context, task_target


def _normalize_task(task: MetaLearningTask, normalizers: dict) -> MetaLearningTask:
    x_norm = task.x - normalizers["mean_x"][None, :]
    if not (normalizers["std_x"] == 0.0).any():
        x_norm = x_norm / normalizers["std_x"][None, :]

    y_norm = task.y - normalizers["mean_y"][None, :]
    if not (normalizers["std_y"] == 0.0).any():
        y_norm = y_norm / normalizers["std_y"][None, :]

    return MetaLearningTask(x=x_norm, y=y_norm)


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


def normalize_benchmark(benchmark: MetaLearningBenchmark) -> MetaLearningBenchmark:
    ## compute normalizers
    x = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_x))
    y = np.zeros((benchmark.n_task, benchmark.n_datapoints_per_task, benchmark.d_y))
    for i, task in enumerate(benchmark._iter_without_noise()):
        x[i], y[i] = task.x, task.y
    normalizers = {
        "mean_x": x.mean(axis=(0, 1)),
        "mean_y": y.mean(axis=(0, 1)),
        "std_x": x.std(axis=(0, 1)),
        "std_y": y.std(axis=(0, 1)),
    }

    ## normalize tasks
    norm_tasks = []
    for i, task in enumerate(benchmark._iter_without_noise()):
        norm_tasks.append(_normalize_task(task=task, normalizers=normalizers))

    ## normalize x-bounds
    x_bounds = benchmark.x_bounds
    x_bounds = x_bounds - normalizers["mean_x"][:, None]
    if not (normalizers["std_x"] == 0.0).any():
        x_bounds = x_bounds / normalizers["std_x"][:, None]

    ## generate new benchmark with normalized tasks
    return ListOfTasksBenchmark(
        task_list=norm_tasks,
        output_noise=benchmark.output_noise,
        x_bounds=x_bounds,
        seed_task=benchmark.seed_task,
        seed_noise=benchmark.seed_noise,
    )
