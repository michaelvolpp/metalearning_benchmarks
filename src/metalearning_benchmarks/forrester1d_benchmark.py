import numpy as np

from metalearning_benchmarks.parametric_benchmark import ObjectiveFunctionBenchmark


class Forrester1D(ObjectiveFunctionBenchmark):
    # https://www.sfu.ca/~ssurjano/forretal08.html
    # https://mf2.readthedocs.io/en/latest/functions/forrester.html
    d_param = 3
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    a_bounds = np.array([0.5, 2.0])
    b_bounds = np.array([0.1, 20.0])
    c_bounds = np.array([-5.0, 5.0])
    param_bounds = np.array([a_bounds, b_bounds, c_bounds])
    x_bounds = np.array([[0.0, 1.0]])

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
        assert param.shape == (self.d_param,)
        a, b, c = param

        forrester = (6 * x - 2) ** 2 * np.sin(12 * x - 4)
        y = a * forrester + b * (x - 0.5) - c

        return y

    def _x_min(self, param: np.ndarray):
        return None
