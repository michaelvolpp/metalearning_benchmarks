import numpy as np

from metalearning_benchmarks.parametric_benchmark import ParametricBenchmark


class Affine1D(ParametricBenchmark):
    d_param = 2
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    m_bounds = np.array([0.3, 8.0])
    b_bounds = np.array([-8.0, -3.0])
    param_bounds = np.array([m_bounds, b_bounds])
    x_bounds = np.array([[-1.0, 1.0]])

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

    def __call__(self, x: np.ndarray, param: np.ndarray):
        m, b = param
        y = m * x + b
        return y
