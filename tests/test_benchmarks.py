import pytest

from metalearning_benchmarks import benchmark_dict
from metalearning_benchmarks import MetaLearningTask, ParametricBenchmark
import numpy as np


@pytest.fixture()
def all_benchmarks():
    return list(benchmark_dict.values())


def test_static_attributes(all_benchmarks):
    print("Testing static attributes of {:d} benchmarks".format(len(all_benchmarks)))

    # TODO: check that d_x, d_y, and d_param are static properties for all benchmarks
    for bm in all_benchmarks:
        assert isinstance(bm.d_x, int)
        assert isinstance(bm.d_y, int)
        if issubclass(bm, ParametricBenchmark):
            assert isinstance(bm.d_param, int) 
        assert isinstance(bm.is_dynamical_system, bool)


def test_shapes(all_benchmarks):
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
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x,
            seed_task=seed_task,
            seed_noise=seed_noise,
        )

        task = bm_instance.get_random_task()
        assert bm_instance.x_bounds.shape == (bm_instance.d_x, 2)
        assert (bm_instance.x_bounds[:, 0] < bm_instance.x_bounds[:, 1]).all()
        assert len(list(bm_instance)) == n_task
        assert isinstance(task, MetaLearningTask)
        assert task.x.shape == (n_datapoints_per_task, bm_instance.d_x)
        assert task.y.shape == (n_datapoints_per_task, bm_instance.d_y)
        if issubclass(bm, ParametricBenchmark):
            assert task.param.shape == (bm_instance.d_param,)


def test_determinism(all_benchmarks):
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
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_1,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_1_2 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_1,
            seed_task=seed_task_2,
            seed_noise=seed_noise,
        )
        bm_instance_1_3 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_1,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_2_1 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_2,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        bm_instance_2_2 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_2,
            seed_task=seed_task_2,
            seed_noise=seed_noise,
        )
        bm_instance_2_3 = bm(
            n_task=n_task,
            n_datapoints_per_task=n_datapoints_per_task,
            output_noise=output_noise if not bm.is_dynamical_system else 0.0,
            seed_x=seed_x_2,
            seed_task=seed_task_1,
            seed_noise=seed_noise,
        )
        task_1_1 = bm_instance_1_1.get_task_by_index(task_index=0)
        task_1_2 = bm_instance_1_2.get_task_by_index(task_index=0)
        task_1_3 = bm_instance_1_3.get_task_by_index(task_index=0)
        task_2_1 = bm_instance_2_1.get_task_by_index(task_index=0)
        task_2_2 = bm_instance_2_2.get_task_by_index(task_index=0)
        task_2_3 = bm_instance_2_3.get_task_by_index(task_index=0)

        if not bm.is_dynamical_system:
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

        if issubclass(bm, ParametricBenchmark):
            # same param-seed -> same params?
            assert (task_1_1.param == task_1_3.param).all()
            assert (task_1_1.param == task_2_1.param).all()
            assert (task_1_1.param == task_2_3.param).all()

            # different param-seed -> different params?
            assert (task_1_1.param != task_1_2.param).any()
            assert (task_1_3.param != task_1_2.param).any()
            assert (task_2_1.param != task_2_2.param).any()
            assert (task_2_3.param != task_2_2.param).any()


def test_noise(all_benchmarks):
    print("Testing noise behaviour of {:d} benchmarks".format(len(all_benchmarks)))

    n_task = 3
    n_datapoints_per_task = 8
    seed_x = 1234
    seed_task = 2234
    seed_noise_1, seed_noise_2 = 3236, 3237
    output_noise_0, output_noise_1, output_noise_2 = 0.0, 0.1, 0.2

    for bm in all_benchmarks:
        if bm.is_dynamical_system:
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
        assert (task_noise_seed_1_no_noise.y == task_noise_seed_1_noise_0.y).all()
        assert (task_noise_seed_1_no_noise.y == task_noise_seed_2_no_noise.y).all()
        # check that adding noise changes data compared to output_noise == 0
        assert (task_noise_seed_1_noise_0.y != task_noise_seed_1_noise_1.y).any()
        # check that different noise levels generate different data
        assert (task_noise_seed_2_noise_1.y != task_noise_seed_1_noise_2.y).any()
        # check that different seed_noise with same noise level generate different data
        assert (task_noise_seed_1_noise_1.y != task_noise_seed_2_noise_1.y).any()
        # check that retrieving the same task twice yields the same data
        assert (task_noise_seed_1_noise_1.y == task_noise_seed_1_noise_1_2.y).all()
