from abc import abstractmethod

import numpy as np
import scipy.linalg
import scipy.spatial

from metalearning_benchmarks.metalearning_benchmark import (
    MetaLearningBenchmark,
    MetaLearningTask,
)


class GPBenchmark(MetaLearningBenchmark):
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
        self._generate_hyperparams()

        y = [
            self.generate_one_task(id=id, x=task_x) for id, task_x in enumerate(self.x)
        ]
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

    @abstractmethod
    def _generate_hyperparams(self):
        pass

    @abstractmethod
    def kernel(self, id, distances, **kwargs):
        raise NotImplementedError

    def gram_matrix(self, id, x):
        distances = scipy.spatial.distance.pdist(x)
        gram_matrix_triu = self.kernel(id=id, distances=distances)
        gram_matrix_diag = self.kernel(id=id, distances=0.0) * np.eye(x.shape[0])

        gram_matrix = np.zeros((x.shape[0], x.shape[0]))
        triu_idx = np.triu_indices(x.shape[0], k=1)  # without diagonal
        gram_matrix[triu_idx[0], triu_idx[1]] = gram_matrix_triu
        gram_matrix = gram_matrix + gram_matrix.T + gram_matrix_diag

        return gram_matrix

    def generate_one_task(self, id, x):
        K = self.gram_matrix(id=id, x=x)
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

    def _generate_hyperparams(self):
        pass # hyperparameters are fixed

    def kernel(self, id, distances, lengthscale=1.0, signal_var=1.0):
        kernel_val = signal_var * np.exp(-1 / 2 * distances**2 / lengthscale**2)
        return kernel_val


class RBFGPVBenchmark(GPBenchmark):
    """
    RBFGP with varying hyperparameters according to
    Kim et al., "Attentive Neural Processes".
    """

    d_x = 1
    d_y = 1
    d_hyperparam = 2
    lengthscale_bounds = np.array([0.1, 0.6])
    signal_scale_bounds = np.array([0.1, 1.0])
    hyperparam_bounds = np.array([lengthscale_bounds, signal_scale_bounds])

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
    
    def _generate_hyperparams(self):
        self.hyperparams = (
            self.rng_task.rand(self.n_task, self.d_hyperparam)
            * (self.hyperparam_bounds[:, 1] - self.hyperparam_bounds[:, 0])
            + self.hyperparam_bounds[:, 0]
        )

    def kernel(self, id, distances):
        lengthscale = self.hyperparams[id, 0]
        signal_var = self.hyperparams[id, 1] ** 2
        kernel_val = signal_var * np.exp(-1 / 2 * distances**2 / lengthscale**2)
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

    def _generate_hyperparams(self):
        pass # hyperparameters are fixed

    def kernel(self, id, distances, lengthscale=0.25):
        kernel_val = (
            1
            + np.sqrt(5) * distances / lengthscale
            + 5 * distances**2 / (3 * lengthscale**2)
        ) * np.exp(-np.sqrt(5) * distances / lengthscale)
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

    def _generate_hyperparams(self):
        pass # hyperparameters are fixed

    def kernel(self, id, distances):
        kernel_val = np.exp(
            -2 * np.sin(1 / 2 * distances) ** 2 - 1 / 8 * distances**2
        )
        return kernel_val
