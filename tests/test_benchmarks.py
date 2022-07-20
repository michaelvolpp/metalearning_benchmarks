import pytest

from metalearning_benchmarks import benchmark_dict
from metalearning_benchmarks.metalearning_benchmark import MetaLearningTask
import numpy as np
from metalearning_benchmarks.util import normalize_benchmark


@pytest.fixture()
def all_benchmarks():
    return list(benchmark_dict.values())


def test_static_attributes(all_benchmarks):
    print("Testing static attributes of {:d} benchmarks".format(len(all_benchmarks)))

    # TODO: check that d_x, d_y, and d_param are static properties for all benchmarks
    for bm in all_benchmarks:
        assert isinstance(bm.d_x, int)
        assert isinstance(bm.d_y, int)
        assert isinstance(bm.d_param, int) or bm.d_param is None


def test_shapes(all_benchmarks):
    perform_shape_test(all_benchmarks, normalize=False)
    perform_shape_test(all_benchmarks, normalize=True)


def test_determinism(all_benchmarks):
    perform_determinism_test(all_benchmarks, normalize=False)
    perform_determinism_test(all_benchmarks, normalize=True)


def test_noise(all_benchmarks):
    perform_noise_test(all_benchmarks, normalize=False)
    perform_noise_test(all_benchmarks, normalize=True)


def test_normalize(all_benchmarks):
    print("Testing normalization for {:d} benchmarks".format(len(all_benchmarks)))

    n_task = 3
    n_datapoints_per_task = 8
    output_noise = 1e-2
    seed_x = 1234
    seed_task = 1235
    seed_noise = 1236

    for bm in all_benchmarks:
        bm_instance = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise,
        )
        bm_instance = normalize_benchmark(bm_instance)

        ## check that normalization works
        x = np.zeros((n_task, n_datapoints_per_task, bm_instance.d_x))
        y = np.zeros((n_task, n_datapoints_per_task, bm_instance.d_y))
        for i, task in enumerate(bm_instance):
            x[i], y[i] = task.x, task.y
        normalizers = {
            "mean_x": x.mean(axis=(0, 1)),
            "mean_y": y.mean(axis=(0, 1)),
            "std_x": x.std(axis=(0, 1)),
            "std_y": y.std(axis=(0, 1)),
        }
        assert normalizers["mean_x"] == pytest.approx(
            np.zeros((bm_instance.d_x)), abs=1e-2
        )
        assert normalizers["mean_y"] == pytest.approx(
            np.zeros((bm_instance.d_y)), abs=1e-2
        )
        if not (normalizers["std_x"] == 0.0).any():
            assert normalizers["std_x"] == pytest.approx(
                np.ones((bm_instance.d_x)), abs=1e-2
            )
        if not (normalizers["std_y"] != 0.0).any():
            assert normalizers["std_y"] == pytest.approx(
                np.ones((bm_instance.d_y)), abs=1e-2
            )


def perform_shape_test(all_benchmarks, normalize):
    print("Testing shapes of {:d} benchmarks".format(len(all_benchmarks)))

    n_task = 3
    n_datapoints_per_task = 8
    output_noise = 1e-2
    seed_x = 1234
    seed_task = 1235
    seed_noise = 1236

    for bm in all_benchmarks:
        bm_instance = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise,
        )
        if normalize:
            bm_instance = normalize_benchmark(bm_instance)

        task = bm_instance.get_random_task()
        assert bm_instance.x_bounds.shape == (bm_instance.d_x, 2)
        assert (bm_instance.x_bounds[:, 0] < bm_instance.x_bounds[:, 1]).all()
        assert len(list(bm_instance)) == n_task
        assert isinstance(task, MetaLearningTask)
        assert task.x.shape == (n_datapoints_per_task, bm_instance.d_x)
        assert task.y.shape == (n_datapoints_per_task, bm_instance.d_y)
        if not bm_instance.is_nonparametric:
            assert task.param.shape == (bm_instance.d_param,)


def perform_determinism_test(all_benchmarks, normalize):
    print("Testing determinisim of {:d} benchmarks".format(len(all_benchmarks)))

    n_task = 3
    n_datapoints_per_task = 8
    output_noise = 1e-2
    seed_x_1, seed_x_2 = 1234, 2234
    seed_task_1, seed_task_2 = 1235, 2235
    seed_noise = 1236  # we check noise in a separate test

    for bm in all_benchmarks:
        # generate some tasks with various random seeds
        bm_instance_1_1 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_1,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_1_2 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_1,
            seed_task=seed_task_2,
            seed_noise=seed_noise,
        )
        bm_instance_1_3 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_1,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_2_1 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_2,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_2_2 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_2,
            seed_task=seed_task_2,
            seed_noise=seed_noise,
        )
        bm_instance_2_3 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise
            if not getattr(bm, "is_dynamical_system", False)
            else 0,
            seed_x=seed_x_2,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        if normalize:
            bm_instance_1_1 = normalize_benchmark(bm_instance_1_1)
            bm_instance_1_2 = normalize_benchmark(bm_instance_1_2)
            bm_instance_1_3 = normalize_benchmark(bm_instance_1_3)
            bm_instance_2_1 = normalize_benchmark(bm_instance_2_1)
            bm_instance_2_2 = normalize_benchmark(bm_instance_2_2)
            bm_instance_2_3 = normalize_benchmark(bm_instance_2_3)

        task_1_1 = bm_instance_1_1.get_task_by_index(task_index=0)
        task_1_2 = bm_instance_1_2.get_task_by_index(task_index=0)
        task_1_3 = bm_instance_1_3.get_task_by_index(task_index=0)
        task_2_1 = bm_instance_2_1.get_task_by_index(task_index=0)
        task_2_2 = bm_instance_2_2.get_task_by_index(task_index=0)
        task_2_3 = bm_instance_2_3.get_task_by_index(task_index=0)

        if not getattr(bm, "is_dynamical_system", False):
            # same x-seed -> same x?
            assert (task_1_1.x == task_1_2.x).all()
            assert (task_1_1.x == task_1_3.x).all()
            assert (task_2_1.x == task_2_2.x).all()
            assert (task_2_1.x == task_2_3.x).all()
        else:
            # same x-seed and same task-seed -> same x?
            assert (task_1_1.x == task_1_3.x).all()
            assert (task_2_1.x == task_2_3.x).all()

        # different x-seed -> different x?
        assert (task_1_1.x != task_2_1.x).any()
        assert (task_1_1.x != task_2_2.x).any()
        assert (task_1_1.x != task_2_3.x).any()
        assert (task_1_2.x != task_2_1.x).any()
        assert (task_1_2.x != task_2_2.x).any()
        assert (task_1_2.x != task_2_3.x).any()
        assert (task_1_3.x != task_2_1.x).any()
        assert (task_1_3.x != task_2_2.x).any()
        assert (task_1_3.x != task_2_3.x).any()

        # same param-seed -> same params?
        if not bm_instance_1_1.is_nonparametric:
            assert (task_1_1.param == task_2_1.param).all()
            assert (task_1_2.param == task_2_2.param).all()

        # different param-seed -> different params?
        if not bm_instance_1_1.is_nonparametric:
            assert (task_1_1.param != task_1_2.param).any()
            assert (task_2_1.param != task_2_2.param).any()


def perform_noise_test(all_benchmarks, normalize):
    print("Testing noise behaviour of {:d} benchmarks".format(len(all_benchmarks)))

    n_task = 3
    n_datapoints_per_task = 8
    seed_x = 1234
    seed_task = 2234
    seed_noise_1, seed_noise_2 = 3236, 3237
    output_noise_0, output_noise_1, output_noise_2 = 0.0, 0.1, 0.2

    for bm in all_benchmarks:
        if getattr(bm, "is_dynamical_system", False):
            continue  # dynamical systems do not use output noise

        # generate benchmarks with and without noise
        bm_instance_seed_1_noise_0 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise_0,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise_1,
        )
        bm_instance_seed_1_noise_1 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise_1,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise_1,
        )
        bm_instance_seed_1_noise_2 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise_2,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise_1,
        )
        bm_instance_seed_2_noise_1 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise_1,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise_2,
        )
        if normalize:
            bm_instance_seed_1_noise_0 = normalize_benchmark(bm_instance_seed_1_noise_0)
            bm_instance_seed_1_noise_1 = normalize_benchmark(bm_instance_seed_1_noise_1)
            bm_instance_seed_1_noise_2 = normalize_benchmark(bm_instance_seed_1_noise_2)
            bm_instance_seed_2_noise_1 = normalize_benchmark(bm_instance_seed_2_noise_1)

        task_noise_seed_1_noise_0 = bm_instance_seed_1_noise_0.get_task_by_index(
            task_index=0
        )
        task_noise_seed_1_no_noise = (
            bm_instance_seed_1_noise_1._get_task_by_index_without_noise(task_index=0)
        )
        task_noise_seed_1_noise_1 = bm_instance_seed_1_noise_1.get_task_by_index(
            task_index=0
        )
        task_noise_seed_1_noise_1_2 = bm_instance_seed_1_noise_1.get_task_by_index(
            task_index=0
        )
        task_noise_seed_1_noise_2 = bm_instance_seed_1_noise_2.get_task_by_index(
            task_index=0
        )
        task_noise_seed_2_no_noise = (
            bm_instance_seed_2_noise_1._get_task_by_index_without_noise(task_index=0)
        )
        task_noise_seed_2_noise_1 = bm_instance_seed_2_noise_1.get_task_by_index(
            task_index=0
        )

        # check output_noise == 0 produces same data as _get_task_by_index_without_noise
        assert (task_noise_seed_1_no_noise.y == task_noise_seed_1_no_noise.y).all()
        assert (task_noise_seed_1_no_noise.y == task_noise_seed_2_no_noise.y).all()
        # check that adding noise changes data compared to output_noise == 0
        assert (task_noise_seed_1_noise_0.y != task_noise_seed_1_noise_1.y).any()
        # check that different noise levels generate different data
        assert (task_noise_seed_2_noise_1.y != task_noise_seed_1_noise_2.y).any()
        # check that different seed_noise with same noise level generate different data
        assert (task_noise_seed_1_noise_1.y != task_noise_seed_2_noise_1.y).any()
        # check that retrieving the same task twice yields the same data
        assert (task_noise_seed_1_noise_1.y == task_noise_seed_1_noise_1_2.y).all()
