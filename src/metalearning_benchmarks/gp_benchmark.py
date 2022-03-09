from abc import abstractmethod

import numpy as np
import scipy.linalg
import scipy.spatial

from metalearning_benchmarks.base_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)


class GPBenchmark(MetaLearningBenchmark):
    d_param = None

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

        y = [self.generate_one_task(task_x) for task_x in self.x]
        self.y = np.array(y)

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
        return np.array([[-2.0, 2.0]] * self.d_x)

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return MetaLearningTask(x=self.x[task_index], y=self.y[task_index], param=None)

    def kernel(self, r):
        raise NotImplementedError

    def gram_matrix(self, x):
        distances = scipy.spatial.distance.pdist(x)
        gram_matrix_triu = self.kernel(distances)
        gram_matrix_diag = self.kernel(0.0) * np.eye(x.shape[0])

        gram_matrix = np.zeros((x.shape[0], x.shape[0]))
        triu_idx = np.triu_indices(x.shape[0], k=1)  # without diagonal
        gram_matrix[triu_idx[0], triu_idx[1]] = gram_matrix_triu
        gram_matrix = gram_matrix + gram_matrix.T + gram_matrix_diag

        return gram_matrix

    def generate_one_task(self, x):
        K = self.gram_matrix(x)
        # add noise to diagonal to make cholesky stable
        K = K + 1e-5 * np.eye(x.shape[0])
        cholesky = scipy.linalg.cholesky(K, lower=True)
        y = cholesky.dot(self.rng_task.randn(*(x.shape[0], 1)))

        return y


class RBFGPBenchmark(GPBenchmark):
    d_x = 1
    d_y = 1

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    def kernel(self, dist, lengthscale=1.0, signal_var=1.0):
        kernel_val = signal_var * np.exp(-1 / 2 * dist ** 2 / lengthscale ** 2)
        return kernel_val


class Matern52GPBenchmark(GPBenchmark):
    d_x = 1
    d_y = 1

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    def kernel(self, dist, lengthscale=0.25):
        kernel_val = (
            1 + np.sqrt(5) * dist / lengthscale + 5 * dist ** 2 / (3 * lengthscale ** 2)
        ) * np.exp(-np.sqrt(5) * dist / lengthscale)
        return kernel_val


class WeaklyPeriodicGPBenchmark(GPBenchmark):
    d_x = 1
    d_y = 1

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    def kernel(self, dist):
        kernel_val = np.exp(-2 * np.sin(1 / 2 * dist) ** 2 - 1 / 8 * dist ** 2)
        return kernel_val
