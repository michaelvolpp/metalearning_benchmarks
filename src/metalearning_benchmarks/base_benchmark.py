from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np


class MetaLearningTask(ABC):
    """
    This is a simple container for two arrays and (not necessarily) a parameter vector,

    x : (n_datapoints_per_task, d_x)
    y : (n_datapoints_per_task, d_y),
    param : (d_param,)

    which can be accessed as the tasks attributes.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param: Optional[np.ndarray] = None,
    ):
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        if param is not None:
            assert param.ndim == 1

        self._x = x
        self._y = y
        self._param = param

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def param(self) -> Optional[np.ndarray]:
        return self._param

    @property
    def n_datapoints(self) -> int:
        return self.x.shape[0]

    @property
    # for backwards compatibility
    def n_points(self) -> int:
        return self.n_datapoints

    @property
    def d_x(self) -> int:
        return self.x.shape[1]

    @property
    def d_y(self) -> int:
        return self.y.shape[1]

    @property
    def d_param(self) -> Optional[int]:
        return self.param.shape[0] if self.param is not None else None


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
        :return: The input dimensionality.
        """
        pass

    @property
    @abstractmethod
    def d_y(self) -> int:
        """
        Return the output dimensionality.
        """
        pass

    @property
    @abstractmethod
    def d_param(self) -> Optional[int]:
        """
        If the benchmark is parametric, return the dimensionality of the parameter
        vector.
        """
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

    @property
    def is_nonparametric(self) -> bool:
        return self.d_param is None

    def _add_noise_to_task(self, task: MetaLearningTask) -> MetaLearningTask:
        noisy_y = task.y + self.output_noise * self.rng_noise.randn(*task.y.shape)
        return MetaLearningTask(x=task.x, y=noisy_y, param=task.param)

    def get_task_by_index(self, task_index: int) -> MetaLearningTask:
        task = self._add_noise_to_task(
            self._get_task_by_index_without_noise(task_index=task_index)
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
