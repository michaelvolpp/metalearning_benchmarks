from abc import abstractmethod

import numpy as np

from metalearning_benchmarks.metalearning_benchmark import \
    MetaLearningBenchmark
from metalearning_benchmarks.metalearning_task import MetaLearningTask


class ParametricBenchmark(MetaLearningBenchmark):
    """
    Serves as an ABC for creating a collection of tasks that are generated by a parametric function.
    For example, Quadratic1D draws n_task tuples (a_n,b_n,c_n) and creates a benchmark with elements
    benchmark[n] = y(x) = (a_n * (x + b_n)) ** 2 + c_n.

    By providing a parametrization and a prescription for sampling a given parameter set, a custom
    parametric benchmark can be created.

    For a more general base class, see base_benchmark.py.
    """

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
        self.params = (
            self.rng_task.rand(n_task, self.d_param)
            * (self.param_bounds[:, 1] - self.param_bounds[:, 0])
            + self.param_bounds[:, 0]
        )
        self.x = (
            self.rng_x.rand(n_task, n_datapoints_per_task, self.d_x)
            * (self.x_bounds[:, 1] - self.x_bounds[:, 0])
            + self.x_bounds[:, 0]
        )
        self.y = np.zeros((self.x.shape[0], self.x.shape[1], self.d_y))
        for i in range(self.n_task):
            self.y[i] = self(params=self.params[i], x=self.x[i])

    @property
    @abstractmethod
    def d_x(self) -> int:
        """
        Provide the dimension for the x-values of the benchmark.
        For Quadratic3D, d_x = 3.
        """
        pass

    @property
    @abstractmethod
    def d_y(self) -> int:
        """
        The output dimension of the benchmark.
        """
        pass

    @property
    @abstractmethod
    def d_param(self) -> int:
        """
        The parameter dimension of the benchmark.
        """
        pass

    @property
    @abstractmethod
    def x_bounds(self) -> np.ndarray:
        """
        The bounds of the input values of the benchmark.
        return value shape: (d_x, 2)
        """
        pass

    @property
    @abstractmethod
    def param_bounds(self) -> np.ndarray:
        """
        The bounds of the parameter values of the benchmark.
        return value shape: (d_param, 2)
        """
        pass

    @abstractmethod
    def __call__(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the function at given x and parameters.

        Parameters
        ----------
        params : (d_params,)
        x : (..., d_x)

        Returns
        -------
        y : (..., d_y)
        """
        pass

    def _get_task_by_index_without_noise(self, task_index):
        return MetaLearningTask(
            x=self.x[task_index], y=self.y[task_index], param=self.params[task_index]
        )