from abc import abstractmethod
from typing import Tuple

import numpy as np

from metalearning_benchmarks import MetaLearningBenchmark, MetaLearningTask


def mesh_to_rows(X1, X2):
    x = np.vstack([X1.ravel(), X2.ravel()]).T
    return x


class ImageCompletionBenchmark(MetaLearningBenchmark):
    is_dynamical_system = False
    d_x = 2

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
        if n_datapoints_per_task != self.n_px_total:
            print("Warning! You are not using all pixels available!")
        if self.n_channels != 1:
            raise NotImplementedError  # double check code first

        # load images
        print(" *** Loading all Images ***")
        _all_images = self.load_all_images()

        # transform images to MetaLearningBenchmark
        print(" *** Converting Images to MetaLearningBenchmark ***")
        self._image_ids = self.rng_task.choice(self.n_images_total, size=self.n_task)
        self.x, self.y = self.images_to_x_y(_all_images[self._image_ids])

        # sanity check
        assert self.x.shape == (self.n_task, self.n_px, 2)
        assert self.y.shape == (self.n_task, self.n_px, self.n_channels)

    @property
    @abstractmethod
    def n_px_x1(self) -> int:
        """
        Number of pixels to the right.
        """
        pass

    @property
    @abstractmethod
    def n_px_x2(self) -> int:
        """
        Number of pixels up.
        """
        pass

    @property
    @abstractmethod
    def min_y(self) -> float:
        """
        The lower bound of y values in resulting meta-learning dataset.
        """
        pass

    @property
    @abstractmethod
    def max_y(self) -> float:
        """
        The upper bound of y values in resulting meta-learning dataset.
        """
        pass

    @property
    @abstractmethod
    def min_px_val(self) -> float:
        """
        Minimum pixel value of images.
        """
        pass

    @property
    @abstractmethod
    def max_px_val(self) -> float:
        """
        Maximum pixel value of images.
        """
        pass

    @property
    @abstractmethod
    def n_images_total(self):
        """
        Total number of images available in the dataset.
        """
        pass

    @abstractmethod
    def _load_all_images(self) -> np.ndarray:
        """
        Load all images into an array of shape
        (n_images_total, n_pixels_x1, n_pixels_x2, n_channels).
        """
        pass

    @property
    def n_px_total(self):
        return self.n_px_x1 * self.n_px_x2

    @property
    def n_px(self):
        return self.n_datapoints_per_task

    @property
    def n_channels(self):
        return self.d_y

    @property
    def _x_template(self):
        x1 = np.linspace(self.x_bounds[0, 0], self.x_bounds[0, 1], self.n_px_x1)
        x2 = np.linspace(self.x_bounds[0, 0], self.x_bounds[0, 1], self.n_px_x2)
        X1, X2 = np.meshgrid(x1, x2)
        x = mesh_to_rows(X1, X2)
        return x

    def load_all_images(self):
        images = self._load_all_images()

        # check output
        assert images.shape == (
            self.n_images_total,
            self.n_px_x1,
            self.n_px_x2,
            self.n_channels,
        )
        return images

    def images_to_x_y(self, images: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ## check input
        n_images = images.shape[0]
        assert images.shape == (n_images, self.n_px_x1, self.n_px_x2, self.n_channels)

        ## define x by repeating the template
        x = np.repeat(self._x_template[None], repeats=n_images, axis=0)

        ## define y
        # reshape
        y = images.reshape(n_images, self.n_px_total, self.n_channels)
        # scale y to [min_y, max_y]
        s = (self.max_y - self.min_y) / (self.max_px_val - self.min_px_val)
        y = y * s - self.min_px_val * s + self.min_y

        ## shuffle
        x, y = self._shuffle_x_y(x=x, y=y)

        ## choose n_datapoints_per_task pixels
        x, y = x[:, : self.n_datapoints_per_task], y[:, : self.n_datapoints_per_task]

        ## check output
        assert x.shape == (n_images, self.n_px, 2)
        assert y.shape == (n_images, self.n_px, self.n_channels)
        assert (y >= self.min_y).all()
        assert (y <= self.max_y).all()
        return x, y

    def x_y_to_images(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # check input
        n_images = x.shape[0]
        n_px_present = x.shape[1]  # not nec. self.n_px (context set)
        assert x.shape == (n_images, n_px_present, 2)
        assert y.shape == (n_images, n_px_present, self.n_channels)

        # define full image (use min_y as default, i.e. as the value of non-present px)
        images = self.min_y * np.ones((n_images, self.n_px_total, self.n_channels))

        # fill present indices
        closest_idx = self._get_closest_template_indices(x)
        for n in range(n_images):
            images[n, closest_idx[n], :] = y[n]

        # clip y values to [min_y, max_y] (could lie outside due to noise)
        images = np.clip(images, a_min=self.min_y, a_max=self.max_y)

        # scale pixel values to [min_pixel_value, max_pixel_value]
        s = (self.max_y - self.min_y) / (self.max_px_val - self.min_px_val)
        images = 1 / s * (images + self.min_px_val * s - self.min_y)

        # reshape
        images = np.reshape(
            images, (n_images, self.n_px_x1, self.n_px_x2, self.n_channels)
        )

        # check output
        assert images.shape == (n_images, self.n_px_x1, self.n_px_x2, self.n_channels)
        assert (images >= self.min_px_val).all()
        assert (images <= self.max_px_val).all()
        return images

    def _get_closest_template_indices(self, x: np.ndarray) -> np.ndarray:
        # check input
        n_images = x.shape[0]
        n_px_present = x.shape[1]
        assert x.shape == (n_images, n_px_present, 2)

        # compute closest idx w.r.t. Eucledian distance
        closest_idx = np.argmin(
            ((self._x_template[None, :, None, :] - x[:, None, :, :]) ** 2).sum(-1),
            axis=1,
        )

        # check output
        assert closest_idx.shape == (n_images, n_px_present)
        return closest_idx

    def _shuffle_x_y(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Shuffle x, y along pixel axis
        """
        ## check input
        n_images = x.shape[0]
        n_px = x.shape[1]
        assert x.shape == (n_images, n_px, 2)
        assert y.shape == (n_images, n_px, self.n_channels)

        ## shuffle
        idx = np.arange(n_px)
        for n in range(n_images):  # TODO: vectorize this
            # use rng_x, as then we can generate the same images but with different
            # pixel subsets (if n_datapoints_per_task < n_px_total)
            idx = self.rng_x.permutation(idx) 
            x[n] = x[n, idx]
            y[n] = y[n, idx]

        return x, y

    def _get_task_by_index_without_noise(self, task_index: int) -> MetaLearningTask:
        return MetaLearningTask(x=self.x[task_index], y=self.y[task_index], param=None)
