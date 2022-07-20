import numpy as np

from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
)


class Linear1D(ParametricBenchmark):
    d_param = 1
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    m_bounds = np.array([3.0, 8.0])
    param_bounds = np.array([m_bounds])
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

    def __call__(self, params, x):
        m = params
        y = m * x
        return y
