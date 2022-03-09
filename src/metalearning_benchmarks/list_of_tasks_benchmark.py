from metalearning_benchmarks.base_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)
from typing import List, Optional
import numpy as np


def _is_consistent(task_list: List[MetaLearningTask]) -> bool:
    assert task_list  # not empty
    for task in task_list:
        assert task.d_x == task_list[0].d_x
        assert task.d_y == task_list[0].d_y
        assert task.n_datapoints == task_list[0].n_datapoints


def _get_x_bounds(task_list: List[MetaLearningTask]) -> np.ndarray:
    d_x = task_list[0].d_x
    min_x = np.array([np.inf] * d_x)
    max_x = np.array([-np.inf] * d_x)
    for task in task_list:
        for i in range(d_x):
            cur_min_x = min(task.x[i])
            cur_max_x = max(task.x[i])
            if cur_min_x < min_x[i]:
                min_x[i] = cur_min_x
            if cur_max_x > max_x[i]:
                max_x[i] = cur_max_x
    return np.stack((min_x, max_x), axis=1)


class ListOfTasksBenchmark(MetaLearningBenchmark):
    def __init__(
        self,
        task_list: List[MetaLearningBenchmark],
        output_noise: float,
        x_bounds: np.ndarray,
        seed_task: int,
        seed_noise: int,
    ):
        _is_consistent(task_list=task_list)
        super().__init__(
            n_task=len(task_list),
            n_datapoints_per_task=task_list[0].n_datapoints,
            output_noise=output_noise,
            seed_x=None,
            seed_task=seed_task,
            seed_noise=seed_noise,
        )
        self._task_list = task_list
        self._x_bounds = x_bounds

    @property
    def d_x(self) -> int:
        return self._task_list[0].d_x

    @property
    def d_y(self) -> int:
        return self._task_list[0].d_y

    @property
    def d_param(self) -> Optional[int]:
        return None

    @property
    def x_bounds(self) -> np.ndarray:
        return self._x_bounds

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return self._task_list[task_index]
