import numpy as np

from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
)


class Quadratic3D(ParametricBenchmark):
    # cf. ABLR paper
    d_param = 3
    d_x = 3
    d_y = 1
    is_dynamical_system = False

    a_bounds = np.array([0.1, 10.0])
    b_bounds = np.array([0.1, 10.0])
    c_bounds = np.array([0.1, 10.0])
    param_bounds = np.array([a_bounds, b_bounds, c_bounds])
    x_bounds = np.array([[-1.0, 1.0]] * d_x)

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

    def __call__(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        a, b, c = params
        y = (
            0.5 * a * np.linalg.norm(x, axis=1, keepdims=True) ** 2
            + b * np.matmul(x, np.ones((self.d_x, 1)))
            + 3 * c
        )
        return y
