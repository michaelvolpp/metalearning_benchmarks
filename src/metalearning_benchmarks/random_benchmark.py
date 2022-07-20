from abc import abstractmethod

from metalearning_benchmarks.metalearning_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)
import numpy as np


class RandomBenchmark(MetaLearningBenchmark):
    is_dynamical_system = False

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )
        self.x = (
            self.rng_x.rand(n_task, n_datapoints_per_task, self.d_x)
            * (self.x_bounds[:, 1] - self.x_bounds[:, 0])
            + self.x_bounds[:, 0]
        )
        self.y = self.rng_task.rand(n_task, n_datapoints_per_task, self.d_y)

    @property
    @abstractmethod
    def d_x(self) -> int:
        pass

    @property
    @abstractmethod
    def d_y(self) -> int:
        pass

    @property
    def x_bounds(self) -> np.ndarray:
        return np.array([[-1.0, 1.0]] * self.d_x)

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return MetaLearningTask(x=self.x[task_index], y=self.y[task_index])


class RandomBenchmarkDx3Dy2(RandomBenchmark):
    d_x = 3
    d_y = 2

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )


class RandomBenchmarkDx1Dy1(RandomBenchmark):
    d_x = 1
    d_y = 1

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )
