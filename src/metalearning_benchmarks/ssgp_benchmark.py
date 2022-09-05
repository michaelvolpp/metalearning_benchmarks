from abc import abstractmethod
from typing import Tuple

import numpy as np

from metalearning_benchmarks import (
    RBFGPBenchmark,
    RBFGPVBenchmark,
    Matern52GPBenchmark,
    ObjectiveFunctionBenchmark,
)


class SparseSpectrumGPBenchmark(ObjectiveFunctionBenchmark):
    is_dynamical_system = False
    noise_var_ssgp = 1e-3  # how far the sample deviates from the training data
    n_features = 100
    jitter = 1e-10

    # we use the parameter as a float task_id
    d_param = 1
    param_bounds = np.array([[0.0, 1.0]])

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        # is filled in super().__init__ through self._pre_call_hook
        self._phi_handle = dict()
        self._theta = dict()
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    @property
    @abstractmethod
    def _data_generating_gp(self):
        pass

    @abstractmethod
    def _compute_phi_handle(self, task_id: int):
        pass

    def _x_min(self, param: np.ndarray):
        return None  # minimum not known

    def _pre_call_hook(self):
        """
        Pre-compute the theta samples and phi handles for each task.
        """
        for l, param in enumerate(self.params):
            key = param[0]  # float
            task = self._data_generating_gp.get_task_by_index(l)
            self._phi_handle[key] = self._compute_phi_handle(l)
            self._theta[key] = self._train(x=task.x, y=task.y, param=param)

    def _train(self, x: np.ndarray, y: np.ndarray, param: np.ndarray) -> np.ndarray:
        """
        Compute theta samples given training data (x, y).
        """
        # evaluate phi handle
        key = param[0]
        phi = self._phi_handle[key](x=x)

        # compute mean of Gaussian
        a = phi.T @ phi + self.noise_var_ssgp * np.eye(self.n_features)
        a_inv = np.linalg.inv(a)
        mu = a_inv @ phi.T @ y

        # compute Cholesky of Gaussian
        var = self.noise_var_ssgp * a_inv
        var = var + self.jitter * np.eye(var.shape[0])
        var = (var + var.T) / 2
        chol = np.linalg.cholesky(var)

        # sample Gaussian
        theta = mu + chol @ self.rng_task.randn(self.n_features, 1)
        return theta

    def __call__(self, x: np.ndarray, param: np.ndarray) -> np.ndarray:
        key = param[0]
        y = self._theta[key].T @ self._phi_handle[key](x).T
        return y.T


class RBFSparseSpectrumGPBenchmark(SparseSpectrumGPBenchmark):
    d_x = RBFGPBenchmark.d_x
    d_y = RBFGPBenchmark.d_y
    x_bounds = RBFGPBenchmark.x_bounds
    lengthscale = RBFGPBenchmark.lengthscale
    signal_var = RBFGPBenchmark.signal_var

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        self._data_gp = RBFGPBenchmark(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    @property
    def _data_generating_gp(self):
        return self._data_gp

    def _compute_phi_handle(self, task_id):
        """
        Compute a new phi handle.
        """
        w = self.rng_task.randn(self.n_features, self.d_x) / self.lengthscale
        b = self.rng_task.uniform(0, 2 * np.pi, size=self.n_features)
        phi_handle = lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(
            x @ w.T + b
        )
        return phi_handle


class RBFSparseSpectrumGPVBenchmark(SparseSpectrumGPBenchmark):
    d_x = RBFGPVBenchmark.d_x
    d_y = RBFGPVBenchmark.d_y
    x_bounds = RBFGPVBenchmark.x_bounds

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        self._data_gp = RBFGPVBenchmark(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    @property
    def _data_generating_gp(self):
        return self._data_gp

    def _compute_phi_handle(self, task_id):
        """
        Compute a new phi handle.
        """
        lengthscale, signal_scale = self._data_generating_gp._hyperparams[task_id]
        signal_var = signal_scale**2
        w = self.rng_task.randn(self.n_features, self.d_x) / lengthscale
        b = self.rng_task.uniform(0, 2 * np.pi, size=self.n_features)
        phi_handle = lambda x: np.sqrt(2 * signal_var / self.n_features) * np.cos(
            x @ w.T + b
        )
        return phi_handle


class Matern52SparseSpectrumGPBenchmark(SparseSpectrumGPBenchmark):
    d_x = Matern52GPBenchmark.d_x
    d_y = Matern52GPBenchmark.d_y
    x_bounds = Matern52GPBenchmark.x_bounds
    lengthscale = Matern52GPBenchmark.lengthscale
    signal_var = Matern52GPBenchmark.signal_var

    def __init__(
        self, n_task, n_datapoints_per_task, output_noise, seed_task, seed_x, seed_noise
    ):
        self._data_gp = Matern52GPBenchmark(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise,
            seed_task=seed_task,
            seed_x=seed_x,
            seed_noise=seed_noise,
        )
        super().__init__(
            n_task,
            n_datapoints_per_task,
            output_noise,
            seed_task,
            seed_x,
            seed_noise,
        )

    @property
    def _data_generating_gp(self):
        return self._data_gp

    def _compute_phi_handle(self, task_id):
        """
        Compute a new phi handle.
        """
        w = self.rng_task.standard_t(5, (self.n_features, self.d_x)) / self.lengthscale
        b = self.rng_task.uniform(0, 2 * np.pi, size=self.n_features)
        phi_handle = lambda x: np.sqrt(2 * self.signal_var / self.n_features) * np.cos(
            x @ w.T + b
        )
        return phi_handle


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    n_task = 16
    ssgp = Matern52SparseSpectrumGPBenchmark(
        n_task=n_task,
        n_datapoints_per_task=16,
        output_noise=0.0,
        seed_task=1234,
        seed_x=2234,
        seed_noise=3234,
    )

    fig, ax = plt.subplots()
    for task_id in range(n_task):
        # get data from the SSGP
        task = ssgp.get_task_by_index(task_id)
        x, y = task.x, task.y
        # evaluate the SSGP
        x_plot = np.linspace(-2.0, 2.0, 1000).reshape(1000, 1)
        y_plot = ssgp.call_task_by_index_without_noise(x=x_plot, task_index=task_id)
        # get data from underlying (non-SS)GP
        task_data = ssgp._data_generating_gp.get_task_by_index(task_id)
        x_data, y_data = task_data.x, task_data.y
        # plot
        color = next(ax._get_lines.prop_cycler)["color"]
        ax.plot(x_plot, y_plot, color=color)
        ax.scatter(x, y, marker="x", color=color)
        ax.scatter(x_data, y_data, marker="o", color=color)
    plt.show()
