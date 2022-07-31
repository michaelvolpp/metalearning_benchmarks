import numpy as np

from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
)


class Sinusoid1D(ParametricBenchmark):
    # cf. MAML paper, section "5.1. Regression"
    d_param = 2
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    amplitude_bounds = np.array([0.1, 5])
    phase_bounds = np.array([0.0, np.pi])
    param_bounds = np.array([amplitude_bounds, phase_bounds])
    x_bounds = np.array([[-5.0, 5.0]])

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

    def __call__(self, x: np.ndarray, param: np.ndarray) -> np.ndarray:
        amplitude, phase = param
        y = amplitude * np.sin(x + phase)
        return y
