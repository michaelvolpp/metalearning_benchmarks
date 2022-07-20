from abc import abstractmethod

import numpy as np
from numpy.polynomial.polynomial import Polynomial

from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
)


class PolynomialBenchmark(ParametricBenchmark):
    d_x = 1
    d_y = 1
    x_bounds = np.array([[-1.0, 1.0]])
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

    @property
    @abstractmethod
    def d_param(self) -> int:
        pass

    @property
    def param_bounds(self) -> np.ndarray:
        return np.array([[-10.0, 10.0]] * self.d_param)

    def __call__(self, params: np.ndarray, x: np.ndarray) -> np.ndarray:
        return Polynomial(coef=params)(x)


class PolynomialDeg0(PolynomialBenchmark):
    degree = 0
    d_param = degree + 1

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


class PolynomialDeg1(PolynomialBenchmark):
    degree = 1
    d_param = degree + 1

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


class PolynomialDeg2(PolynomialBenchmark):
    degree = 2
    d_param = degree + 1

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


class PolynomialDeg5(PolynomialBenchmark):
    degree = 5
    d_param = degree + 1

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


class PolynomialDeg10(PolynomialBenchmark):
    degree = 10
    d_param = degree + 1

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
