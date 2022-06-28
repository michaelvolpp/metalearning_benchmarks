import numpy as np
from typing import Optional


class MetaLearningTask:
    """
    This is a simple container for two arrays and (not necessarily) a parameter vector,

    x : (n_datapoints_per_task, d_x)
    y : (n_datapoints_per_task, d_y),
    param : (d_param,)

    which can be accessed as the tasks attributes.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        param: Optional[np.ndarray] = None,
    ):
        assert x.ndim == y.ndim == 2
        assert x.shape[0] == y.shape[0]
        if param is not None:
            assert param.ndim == 1

        self._x = x
        self._y = y
        self._param = param

    @property
    def x(self) -> np.ndarray:
        return self._x

    @property
    def y(self) -> np.ndarray:
        return self._y

    @property
    def param(self) -> Optional[np.ndarray]:
        return self._param

    @property
    def n_datapoints(self) -> int:
        return self.x.shape[0]

    @property
    # for backwards compatibility
    def n_points(self) -> int:
        return self.n_datapoints

    @property
    def d_x(self) -> int:
        return self.x.shape[1]

    @property
    def d_y(self) -> int:
        return self.y.shape[1]

    @property
    def d_param(self) -> Optional[int]:
        return self.param.shape[0] if self.param is not None else None
