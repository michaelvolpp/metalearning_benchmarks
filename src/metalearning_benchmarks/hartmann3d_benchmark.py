import numpy as np

from metalearning_benchmarks.parametric_benchmark import ObjectiveFunctionBenchmark


class Hartmann3D(ObjectiveFunctionBenchmark):
    # https://www.sfu.ca/~ssurjano/hart3.html
    # https://github.com/boschresearch/MetaBO/blob/master/metabo/environment/objectives.py
    d_param = 4  # 3D-translation, 1D-scaling
    d_x = 3
    d_y = 1
    is_dynamical_system = False

    t1_bounds = np.array([-0.25, 0.25])
    t2_bounds = np.array([-0.25, 0.25])
    t3_bounds = np.array([-0.25, 0.25])
    s_bounds = np.array([0.75, 1.25])
    param_bounds = np.array([t1_bounds, t2_bounds, t3_bounds, s_bounds])
    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

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
        t = param[0:3]
        s = param[3]

        y = self._hartmann3_translated_scaled(x=x, t=t, s=s)

        return y

    @staticmethod
    def _hartmann3(x: np.ndarray) -> np.ndarray:
        # check inputs
        n_points = x.shape[0]
        assert x.shape == (n_points, 3)

        # define parameters
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = 1e-4 * np.array(
            [
                [3689, 1170, 2673],
                [4699, 4387, 7470],
                [1091, 8732, 5547],
                [381, 5743, 8828],
            ]
        )

        # compute Hartmann-3
        x = x.reshape(x.shape[0], 1, -1)
        B = x - P
        B = B**2
        exponent = A * B
        exponent = np.einsum("ijk->ij", exponent)
        C = np.exp(-exponent)
        hm3 = -np.einsum("i, ki", alpha, C)

        # normalize
        mean = -0.93
        std = 0.95
        hm3 = 1 / std * (hm3 - mean)

        # reshape
        hm3 = hm3.reshape(n_points, 1)

        return hm3

    def _hartmann3_translated_scaled(
        self, x: np.ndarray, t: np.ndarray, s: float
    ) -> np.ndarray:
        # check input
        n_points = x.shape[0]
        assert x.shape == (n_points, 3)
        assert t.shape == (3,)

        # compute translated and scaled branin
        x_new = x - t
        hm3 = self._hartmann3(x_new)
        hm3 = s * hm3

        return hm3

    def _x_min(self, param: np.ndarray):
        return None


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    hm3 = Hartmann3D(
        n_task=16,
        n_datapoints_per_task=16,
        output_noise=0.1,
        seed_task=1235,
        seed_x=2234,
        seed_noise=3234,
    )

    def mesh_to_rows(X1, X2):
        x = np.vstack([X1.ravel(), X2.ravel()]).T
        return x

    def rows_to_mesh(x, z, n_x1, n_x2):
        rows = np.hstack((x, z)).T
        rows = rows.reshape(3, n_x1, n_x2)
        X1, X2, Z = rows[0, :], rows[1, :], rows[2, :]
        return X1, X2, Z

    # plot slices with fixed x3 = x3_val 
    x3_vals = [0.0, 0.25, 0.5, 0.75, 1.0]
    for x3_val in x3_vals:
        task_ids = (0, 1, 2, 3)
        nrows, ncols = int(np.sqrt(len(task_ids))), int(np.sqrt(len(task_ids)))
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(8, 8),
            squeeze=False,
            subplot_kw={"projection": "3d"},
        )
        for i, task_id in enumerate(task_ids):
            row, col = i // ncols, i % ncols
            ax = axes[row, col]
            task = hm3.get_task_by_index(task_id)
            x, y = task.x, task.y
            n_x1, n_x2 = 100, 100
            x1_plt = np.linspace(Hartmann3D.x_bounds[0, 0], Hartmann3D.x_bounds[0, 1], n_x1)
            x2_plt = np.linspace(Hartmann3D.x_bounds[1, 0], Hartmann3D.x_bounds[1, 1], n_x2)
            X1, X2 = np.meshgrid(x1_plt, x2_plt)
            x_plt = mesh_to_rows(X1=X1, X2=X2)
            x_plt_with_x3 = np.hstack([x_plt, x3_val * np.ones((x_plt.shape[0],1))])
            z_plt_var = hm3.call_task_by_index_with_noise(
                x=x_plt_with_x3, task_index=task_id
            )
            X1, X2, Z_plt_var = rows_to_mesh(x=x_plt, z=z_plt_var, n_x1=n_x1, n_x2=n_x2)
            ax.plot_surface(X1, X2, Z_plt_var, alpha=0.5)
            ax.scatter(x[:, 0], x[:, 1], y, color="r")
            ax.set_xlabel("x_1")
            ax.set_ylabel("x_2")
        fig.suptitle(f"x_3 = {x3_val:.2f}")
        fig.tight_layout()
    plt.show()
