import numpy as np

from metalearning_benchmarks.parametric_benchmark import ObjectiveFunctionBenchmark


class Branin2D(ObjectiveFunctionBenchmark):
    # https://www.sfu.ca/~ssurjano/branin.html
    # https://github.com/boschresearch/MetaBO/blob/master/metabo/environment/objectives.py
    d_param = 3  # 2D-translation, 1D-scaling
    d_x = 2
    d_y = 1
    is_dynamical_system = False

    t1_bounds = np.array([-0.25, 0.25])
    t2_bounds = np.array([-0.25, 0.25])
    s_bounds = np.array([0.75, 1.25])
    param_bounds = np.array([t1_bounds, t2_bounds, s_bounds])
    x_bounds = np.array([[0.0, 1.0], [0.0, 1.0]])

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
        t = param[0:2]
        s = param[2]

        y = self._branin_translated_scaled(x=x, t=t, s=s)

        return y

    @staticmethod
    def _branin(x: np.ndarray) -> np.ndarray:
        # check inputs
        n_points = x.shape[0]
        assert x.shape == (n_points, 2)
        x1, x2 = x[:, 0], x[:, 1]

        # scale x \in [0, 1]**2 to domain of Branin
        x1 = x1 * 15.0
        x1 = x1 - 5.0
        x2 = x2 * 15.0

        # define parameters
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)

        # compute Branin
        bra = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s

        # normalize function value
        mean = 54.44
        std = 51.44
        bra = 1 / std * (bra - mean)

        # reshape
        bra = bra.reshape(n_points, 1)

        return bra

    def _branin_translated_scaled(
        self, x: np.ndarray, t: np.ndarray, s: float
    ) -> np.ndarray:
        # check input
        n_points = x.shape[0]
        assert x.shape == (n_points, 2)
        assert t.shape == (2,)

        # compute translated and scaled branin
        x_new = x - t
        bra = self._branin(x_new)
        bra = s * bra

        return bra

    def _x_min(self, param: np.ndarray):
        return None


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    bra = Branin2D(
        n_task=16,
        n_datapoints_per_task=16,
        output_noise=0.25,
        seed_task=1234,
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

    # # plot 1: contour
    # nrows, ncols = 4, 4
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), squeeze=False)
    # for i, task_id in enumerate(range(bra.n_task)):
    #     row, col = i // ncols, i % ncols
    #     ax = axes[row, col]
    #     task = bra.get_task_by_index(task_id)
    #     x, y = task.x, task.y
    #     n_x1, n_x2 = 100, 100
    #     x1_plt = np.linspace(Branin2D.x_bounds[0, 0], Branin2D.x_bounds[0, 1], n_x1)
    #     x2_plt = np.linspace(Branin2D.x_bounds[1, 0], Branin2D.x_bounds[1, 1], n_x2)
    #     X1, X2 = np.meshgrid(x1_plt, x2_plt)
    #     x_plt = mesh_to_rows(X1=X1, X2=X2)
    #     z_plt_var = bra.call_task_by_index_with_noise(x=x_plt, task_index=task_id)
    #     X1, X2, Z_plt_var = rows_to_mesh(x=x_plt, z=z_plt_var, n_x1=n_x1, n_x2=n_x2)
    #     if i == 0:
    #         cs = ax.contour(X1, X2, Z_plt_var, levels=25)
    #     else:
    #         cs = ax.contour(X1, X2, Z_plt_var, levels=cs.levels)
    #     ax.scatter(x[:, 0], x[:, 1])
    # fig.tight_layout()

    # plot 2: 3D
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
        task = bra.get_task_by_index(task_id)
        x, y = task.x, task.y
        n_x1, n_x2 = 100, 100
        x1_plt = np.linspace(Branin2D.x_bounds[0, 0], Branin2D.x_bounds[0, 1], n_x1)
        x2_plt = np.linspace(Branin2D.x_bounds[1, 0], Branin2D.x_bounds[1, 1], n_x2)
        X1, X2 = np.meshgrid(x1_plt, x2_plt)
        x_plt = mesh_to_rows(X1=X1, X2=X2)
        z_plt_var = bra.call_task_by_index_with_noise(x=x_plt, task_index=task_id)
        X1, X2, Z_plt_var = rows_to_mesh(x=x_plt, z=z_plt_var, n_x1=n_x1, n_x2=n_x2)
        ax.plot_surface(X1, X2, Z_plt_var, alpha=.5)
        ax.scatter(x[:, 0], x[:, 1], y, color="r")
    fig.tight_layout()
    plt.show()
