from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from metalearning_benchmarks.metalearning_task import MetaLearningTask


class MetaLearningBenchmark(ABC):
    """
    An abstract base class for a metalearning benchmark, which is a collection of tasks.
    This should be used for nonparametric benchmarks, such as those containing samples from
    a Gaussian process.

    For a parametrised benchmark, see base_parametric_benchmark.py.
    """

    def __init__(
        self,
        n_task: int,
        n_datapoints_per_task: int,
        output_noise: float,
        seed_x: Optional[int],
        seed_task: int,
        seed_noise: int,
    ):
        self.n_task = n_task
        self.n_datapoints_per_task = n_datapoints_per_task
        self.output_noise = output_noise
        self.seed_x = seed_x
        self.seed_task = seed_task
        self.seed_noise = seed_noise
        self._noise_vectors = [None] * self.n_task

        if self.seed_x is not None:
            self.rng_x = np.random.RandomState(seed=self.seed_x)
        else:
            self.rng_x = None
        self.rng_task = np.random.RandomState(seed=self.seed_task)
        self.rng_noise = np.random.RandomState(seed=self.seed_noise)

    @property
    @abstractmethod
    def d_x(self) -> int:
        """
        Has to be defined as a class property.
        :return: The input dimensionality.
        """
        pass

    @property
    @abstractmethod
    def d_y(self) -> int:
        """
        Has to be defined as a class property.
        Return the output dimensionality.
        """
        pass


    @property
    @abstractmethod
    def is_dynamical_system(self) -> bool:
        pass

    @property
    @abstractmethod
    def x_bounds(self) -> np.ndarray:
        """
        Return the bounds of the inputs. (d_x, 2)-np.ndarray. First column is lower
        bounds, second column is the upper bounds.
        """
        pass

    @abstractmethod
    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        """
        Return the task with index task_index.
        """
        pass

    @property
    # for backwards compatibility
    def n_points_per_task(self) -> int:
        return self.n_datapoints_per_task

    @property
    # for backwards compatibility
    def rng_param(self):
        return self.rng_task

    def _generate_noise_vector_for_task(
        self, task: MetaLearningTask, task_index: int
    ) -> np.ndarray:
        if self._noise_vectors[task_index] is None:
            self._noise_vectors[task_index] = self.output_noise * self.rng_noise.randn(
                *task.y.shape
            )

        return self._noise_vectors[task_index]

    def _add_noise_to_task(
        self, task: MetaLearningTask, task_index: int
    ) -> MetaLearningTask:
        noisy_y = task.y + self._generate_noise_vector_for_task(
            task=task, task_index=task_index
        )
        return MetaLearningTask(x=task.x, y=noisy_y, param=task.param)

    def get_task_by_index(self, task_index: int) -> MetaLearningTask:
        task = self._add_noise_to_task(
            self._get_task_by_index_without_noise(task_index=task_index),
            task_index=task_index,
        )
        return task

    def get_random_task(self) -> MetaLearningTask:
        idx = int(self.rng_task.randint(low=0, high=self.n_task, size=1))
        task = self.get_task_by_index(task_index=idx)
        return task

    def __iter__(self) -> MetaLearningTask:
        for task_idx in range(self.n_task):
            yield self.get_task_by_index(task_index=task_idx)

    def _iter_without_noise(self) -> MetaLearningTask:
        for task_idx in range(self.n_task):
            yield self._get_task_by_index_without_noise(task_index=task_idx)

    def __len__(self) -> int:
        return self.n_task
