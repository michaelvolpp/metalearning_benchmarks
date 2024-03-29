from metalearning_benchmarks.metalearning_task import MetaLearningTask
from metalearning_benchmarks.metalearning_benchmark import MetaLearningBenchmark
from metalearning_benchmarks.parametric_benchmark import (
    ParametricBenchmark,
    ObjectiveFunctionBenchmark,
)
from metalearning_benchmarks.image_completion_benchmark import ImageCompletionBenchmark
from metalearning_benchmarks.gp_benchmark import (
    RBFGPBenchmark,
    RBFGPVBenchmark,
    RBFGPV2Benchmark,
    Matern52GPBenchmark,
    WeaklyPeriodicGPBenchmark,
)
from metalearning_benchmarks.ssgp_benchmark import (
    RBFSparseSpectrumGPBenchmark,
    RBFSparseSpectrumGPVBenchmark,
    RBFSparseSpectrumGPV2Benchmark,
    Matern52SparseSpectrumGPBenchmark,
)
from metalearning_benchmarks.polynomial_benchmark import (
    PolynomialDeg0,
    PolynomialDeg1,
    PolynomialDeg10,
    PolynomialDeg2,
    PolynomialDeg5,
)
from metalearning_benchmarks.quadratic1d_benchmark import Quadratic1D
from metalearning_benchmarks.quadratic3d_benchmark import Quadratic3D
from metalearning_benchmarks.random_benchmark import (
    RandomBenchmarkDx1Dy1,
    RandomBenchmarkDx3Dy2,
)
from metalearning_benchmarks.sinusoid1d_benchmark import Sinusoid1D
from metalearning_benchmarks.linear1d_benchmark import Linear1D
from metalearning_benchmarks.affine1d_benchmark import Affine1D
from metalearning_benchmarks.line_sine1d_benchmark import LineSine1D
from metalearning_benchmarks.furuta_benchmark import FreePointMassFurutaBenchmark
from metalearning_benchmarks.forrester1d_benchmark import Forrester1D
from metalearning_benchmarks.branin2d_benchmark import Branin2D
from metalearning_benchmarks.hartmann3d_benchmark import Hartmann3D
from metalearning_benchmarks.mnist_benchmark import (
    MNIST_TrainBenchmark,
    MNIST_TestBenchmark,
)

benchmark_dict = {
    "PolynomialDeg0": PolynomialDeg0,
    "PolynomialDeg1": PolynomialDeg1,
    "PolynomialDeg2": PolynomialDeg2,
    "PolynomialDeg5": PolynomialDeg5,
    "PolynomialDeg10": PolynomialDeg10,
    "RandomBenchmarkDx1Dy1": RandomBenchmarkDx1Dy1,
    "RandomBenchmarkDx3Dy2": RandomBenchmarkDx3Dy2,
    "Sinusoid1D": Sinusoid1D,
    "Linear1D": Linear1D,
    "Affine1D": Affine1D,
    "LineSine1D": LineSine1D,
    "Quadratic1D": Quadratic1D,
    "Quadratic3D": Quadratic3D,
    "RBFGPBenchmark": RBFGPBenchmark,
    "RBFGPVBenchmark": RBFGPVBenchmark,
    "RBFGPV2Benchmark": RBFGPV2Benchmark,
    "Matern52GPBenchmark": Matern52GPBenchmark,
    "WeaklyPeriodicGPBenchmark": WeaklyPeriodicGPBenchmark,
    "RBFSSGPBenchmark": RBFSparseSpectrumGPBenchmark,
    "RBFSSGPVBenchmark": RBFSparseSpectrumGPVBenchmark,
    "RBFSSGPV2Benchmark": RBFSparseSpectrumGPV2Benchmark,
    "Matern52SSGPBenchmark": Matern52SparseSpectrumGPBenchmark,
    "FreePointMassFuruta": FreePointMassFurutaBenchmark,
    "Forrester1D": Forrester1D,
    "Branin2D": Branin2D,
    "Hartmann3D": Hartmann3D,
    "MNIST_TrainBenchmark": MNIST_TrainBenchmark,
    "MNIST_TestBenchmark": MNIST_TestBenchmark,
}
