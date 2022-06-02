from itertools import cycle
from typing import Optional, Tuple, Union

import numpy as np
from scipy.special import comb

from metalearning_benchmarks.base_benchmark import MetaLearningBenchmark


class MetaLearningDataset:
    """
    TLDR:
    -----
    - Extends MetaLearningBenchmark by functionality to
        - Split MetaLearningBenchmarks into training and validation data.
            --> task = [ train         | val ]
        - Optionally mask out context sets from the training data.
            --> task = [ x o x x o x x | val ], o = context points

    General information:
    --------------------
    - MetaLearningDataset provides functionality to split each task contained in
      the MetaLearningBenchmark into training and validation data. This is
      useful if you want to check intra-task generalization, i.e., how well the
      model generalizes to data coming from the same task it was trained on.
    - To check inter-task generalization, use different instances of
      MetaLearningDataset, initialized with different MetaLearningBenchmarks for
      meta-training and meta-testing.

    Use cases:
    ----------
    - Train-validation split:
        - Meta-training:
            - Use training data to train the model. Don't confuse training data
              with context data here. Depending on the model, some points in the
              training data could be labeled as context data. Other models might
              not employ such a split.
            - Use validation data to check intra-task generalization on the
              meta-training data.
        - Meta-testing:
            - Use training data to adapt the model. Typically, you might want to
              employ only parts of the training data (i.e., a context set) for
              adaptation, to check how well generalization works for a given
              context set size.
            - Use validation data to check intra-task generalization on the
              meta-testing data.
    - Context masking:
        - Meta-training:
            - Some models (typically those that amortize inference) will split
              their training data into context and target sets themselves. In
              this case you can use the full training data to train the model.
            - Non-amortizing models view different context set sizes generated
              from the training data of the same task formally as different
              tasks. These "subtasks" have to be fixed a priori (i.e., before
              the training). Thus, to train such models, use the context-masking
              functionality to pre-determine the subtasks.
        - Meta-testing:
            - As described above, typically, you will want to check intra-task
              generalizaiton on the meta-testing data for different sizes of the
              context set (aka different numbers of shots).
            - To this end, also use the functionality described above to mask
              out subtasks a priori from the training data.
            - Always evaluate the performance on the validation data (it is not
              altered by the masking).
            - Note that the non-context points in the training set won't be used
              at all (the model is always evaluated on the validation data and
              adapted on the context data).
            - This functionality allows to make sure to test different models on
              the same context splits.


    Details:
    --------
    - Train-validation split:
        - Each task (x_l, y_l) contained in the MetaLearningBenchmark is split
          as (x_train_l | x_val_l, y_train_l | y_val_l). The splits are
          determined by the parameter n_train. Thus, n_val =
          benchmark.n_datapoints_per_task - n_train.
        - A model is supposed to be meta-trained or meta-tested on the data
          returned by get_train_data() or get_train_dataset().The train data can
          be returned by get_train_data() or get_train_dataset().
    - Context-masking:
        - If model does not use amortization, pre-determine context/query splits
          a priori by providing a list of ctx_sizes upon intialization.
        - get_train_data() or the train_dataset now additionally return a
          boolean mask for each task which marks context data points as True.
        - Furthermore, all data now comes with a task_id s.t. the model knows
          which task is currently presented.
    """

    def __init__(
        self,
        metalearning_benchmark: MetaLearningBenchmark,
        seed: int,
        n_train: Optional[int] = None,
        ctx_size_range: Optional[Tuple[int]] = None,
        n_ctx_sets_per_task: Optional[int] = None,
    ):
        # set attributes
        self.bm = metalearning_benchmark
        self.rng = np.random.default_rng(seed=seed)
        self.n_train = self.bm.n_datapoints_per_task if n_train is None else n_train
        self.n_val = self.bm.n_datapoints_per_task - self.n_train

        # check ctx_size_range
        assert not (n_ctx_sets_per_task is None and ctx_size_range is not None)
        assert not (n_ctx_sets_per_task is not None and ctx_size_range is None)
        if ctx_size_range is None:
            ctx_size_range = (self.n_train, self.n_train)
            n_ctx_sets_per_task = 1
        self.n_ctx_sets_per_task = n_ctx_sets_per_task
        assert ctx_size_range[0] <= ctx_size_range[1]
        assert self.n_train >= ctx_size_range[1]
        self.ctx_size_range = ctx_size_range

        (
            self.x_train,
            self.y_train,
            self.ctx_mask,
            self.ctx_size,
            self.x_val,
            self.y_val,
            self.subtask_ids,
            self.task_id,
        ) = self.generate_dataset()

    @property
    def n_subtasks(self):
        return self.x_train.shape[0]

    def get_training_data(self):
        return (
            self.x_train,
            self.y_train,
            self.ctx_mask,
            self.subtask_ids,
        )

    def get_subtasks_by_subtask_ids(self, subtask_ids: Union[int, np.ndarray]):
        # subtask_id is the id of the task in the dataset 
        return (
            self.x_train[subtask_ids],
            self.y_train[subtask_ids],
            self.ctx_mask[subtask_ids],
            self.x_val[subtask_ids],
            self.y_val[subtask_ids],
            self.subtask_ids[subtask_ids],
        )

    def get_subtasks_by_task_id(self, task_id: int):
        # task_id is the id of the task in the benchmark
        idx = self.task_id == task_id
        if not idx.any():
            raise ValueError(f"Task id {task_id:d} not found!")
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def get_subtasks_by_ctx_size(self, ctx_size: int):
        idx = self.ctx_size == ctx_size
        if not idx.any():
            raise ValueError(f"Task size {ctx_size:d} not available!")
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def get_subtasks_by_task_id_and_ctx_size(self, full_task_id: int, ctx_size: int):
        idx = np.logical_and(
            self.task_id == full_task_id, self.ctx_size == ctx_size
        )
        if not idx.any():
            raise ValueError(
                f"Task id {full_task_id:d} has no subtask of size {ctx_size:d}!"
            )
        return self.get_subtasks_by_subtask_ids(subtask_ids=self.subtask_ids[idx])

    def iter_ctx_sizes(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size

    def iter_subtasks_by_ctx_size(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size, self.get_subtasks_by_ctx_size(ctx_size)

    def iter_subtask_ids_by_ctx_size(self):
        for ctx_size in np.unique(self.ctx_size):
            yield ctx_size, self.get_subtasks_by_ctx_size(ctx_size)[-1]

    def generate_dataset(self):
        # generate subtasks (TODO: improve efficiency!)
        def get_next_ctx_size():
            ct = 0
            for k in ctx_size_iterator:
                if size_counts[k] < comb(self.n_train, k):
                    size_counts[k] += 1
                    return k

                # assure that function returns
                ct += 1
                if ct >= self.n_ctx_sets_per_task:
                    raise ValueError(
                        f"There are not enough distinct context sets available!"
                    )

        def get_next_mask_idx(ctx_size, assure_unique=True):
            # TODO: is there a better way than rejection sampling
            while True:
                mask_idx_cand = sorted(
                    self.rng.choice(range(self.n_train), size=ctx_size, replace=False)
                )
                if assure_unique:
                    if tuple(mask_idx_cand) not in used_mask_idx[ctx_size]:
                        used_mask_idx[ctx_size].append(tuple(mask_idx_cand))
                        return mask_idx_cand
                else:
                    return mask_idx_cand

        # initialize arrays
        n_subtask = self.bm.n_task * self.n_ctx_sets_per_task
        x_train = np.zeros((n_subtask, self.n_train, self.bm.d_x), dtype=np.float32)
        y_train = np.zeros((n_subtask, self.n_train, self.bm.d_y), dtype=np.float32)
        x_val = np.zeros((n_subtask, self.n_val, self.bm.d_x), dtype=np.float32)
        y_val = np.zeros((n_subtask, self.n_val, self.bm.d_y), dtype=np.float32)
        ctx_mask = np.zeros((n_subtask, self.n_train), dtype=np.int32)
        ctx_size = np.zeros((n_subtask,), dtype=np.int32)
        task_ids = np.zeros((n_subtask,), dtype=np.int32)
        full_task_ids = np.zeros((n_subtask,), dtype=np.int32)

        # keep track of how many context sets of size k we already added to
        # avoid obvious redundancy
        ctx_size_iterator = cycle(
            np.unique(
                np.linspace(
                    self.ctx_size_range[0],
                    self.ctx_size_range[1],
                    num=self.n_ctx_sets_per_task,
                    dtype=np.int32,
                )
            )
        )
        size_counts = {
            k: 0 for k in range(self.ctx_size_range[0], self.ctx_size_range[1] + 1)
        }
        used_mask_idx = {
            k: [] for k in range(self.ctx_size_range[0], self.ctx_size_range[1] + 1)
        }

        # TODO: improve efficiency
        task_id = -1
        for _ in range(self.n_ctx_sets_per_task):
            cur_ctx_size = get_next_ctx_size()
            cur_mask_idx = get_next_mask_idx(ctx_size=cur_ctx_size)
            for l, task in enumerate(self.bm):
                task_id += 1
                x_all, y_all = task.x, task.y
                x_train[task_id], y_train[task_id] = (
                    x_all[: self.n_train, :],
                    y_all[: self.n_train, :],
                )
                x_val[task_id], y_val[task_id] = (
                    x_all[self.n_train :, :],
                    y_all[self.n_train :, :],
                )
                ctx_size[task_id] = cur_ctx_size
                ctx_mask[task_id][cur_mask_idx] = 1
                task_ids[task_id] = task_id
                full_task_ids[task_id] = l

        # convert to boolean ctx masks
        ctx_mask = ctx_mask.astype(bool)

        return (
            x_train,
            y_train,
            ctx_mask,
            ctx_size,
            x_val,
            y_val,
            task_ids,
            full_task_ids,
        )
