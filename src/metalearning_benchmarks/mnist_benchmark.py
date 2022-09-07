import os
import pathlib

import numpy as np

import mnist
from metalearning_benchmarks import ImageCompletionBenchmark


class MNISTBenchmark(ImageCompletionBenchmark):

    # mandatory static properties
    d_y = 1  # n_channels
    n_px_x1 = 28
    n_px_x2 = 28
    n_images_total = 60000 + 10000  # train + test

    min_px_val = 0
    max_px_val = 255

    x_bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    min_y = -0.5
    max_y = 0.5

    download_dir = os.path.join(pathlib.Path(__file__).parent.resolve(), "mnist")

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

    def _load_all_images(self):
        # load images
        os.makedirs(self.download_dir, exist_ok=True)
        mnist.temporary_dir = lambda: self.download_dir
        train_images, test_images = mnist.train_images(), mnist.test_images()

        # concatenate images
        all_images = np.concatenate((train_images, test_images), axis=0)
        all_images = all_images[:, :, :, None]  # add channel dim
        return all_images


if __name__ == "__main__":
    import time

    from matplotlib import pyplot as plt

    from metalearning_benchmarks.util import collate_benchmark

    ## generate benchmark
    now = time.time()
    mnist_bm = MNISTBenchmark(
        n_task=10 * 10,
        n_datapoints_per_task=MNISTBenchmark.n_px_x1 * MNISTBenchmark.n_px_x2,
        output_noise=0.25,
        seed_task=1237,
        seed_x=2238,
        seed_noise=3235,
    )
    print(f"Took {time.time() - now:.2f}s")

    ## plot some images
    # choose context
    n_ctx = mnist_bm.n_px//8
    x, y = collate_benchmark(mnist_bm)
    x_ctx, y_ctx = x[:, :n_ctx], y[:, :n_ctx]
    image_ctx = mnist_bm.x_y_to_images(x=x_ctx, y=y_ctx)
    # plot
    nrows = int(np.sqrt(mnist_bm.n_task))
    ncols = nrows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(8, 8), squeeze=False)
    for i in range(image_ctx.shape[0]):
        row, col = i // ncols, i % ncols
        ax = axes[row, col]
        ax.imshow(
            image_ctx[i],
            vmin=mnist_bm.min_px_val,
            vmax=mnist_bm.max_px_val,
            cmap="Greys" if mnist_bm.n_channels == 1 else "viridis",
        )
    fig.tight_layout()
    plt.show()
