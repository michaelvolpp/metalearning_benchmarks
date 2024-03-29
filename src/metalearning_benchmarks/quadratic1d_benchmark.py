import numpy as np

from metalearning_benchmarks.parametric_benchmark import ObjectiveFunctionBenchmark


class Quadratic1D(ObjectiveFunctionBenchmark):
    # cf. our BA-paper @ ICLR
    d_param = 3
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    a_bounds = np.array([0.5, 1.5])
    b_bounds = np.array([-0.9, 0.9])
    c_bounds = np.array([-1.0, 1.0])
    param_bounds = np.array([a_bounds, b_bounds, c_bounds])
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
        a, b, c = param
        y = (a * (x + b)) ** 2 + c
        return y

    def _x_min(self, param: np.ndarray):
        a, b, _ = param
        assert a > 0.0
        return -b
