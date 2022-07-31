import numpy as np

from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
)


class LineSine1D(ParametricBenchmark):
    # cf. Probabilistic MAML paper, section 5 "Illustrative 5-shot Regression"
    d_param = 5
    d_x = 1
    d_y = 1
    is_dynamical_system = False

    choice_bounds = np.array([0.0, 1.0])
    slope_bounds = np.array([-3.0, 3.0])
    intercept_bounds = np.array([-3.0, 3.0])
    amplitude_bounds = np.array([0.1, 5])
    phase_bounds = np.array([0.0, np.pi])
    param_bounds = np.array(
        [choice_bounds, slope_bounds, intercept_bounds, amplitude_bounds, phase_bounds]
    )
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
        choice, slope, intercept, amplitude, phase = param
        if choice < 0.5:
            y = amplitude * np.sin(x + phase)
        else:
            y = slope * x + intercept
        return y
