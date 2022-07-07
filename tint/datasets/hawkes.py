import pickle as pkl
import numpy as np
import os
import torch as th

from torch.nn.utils.rnn import pad_sequence

from .dataset import DataModule


try:
    from tick.hawkes import SimuHawkes, HawkesKernelExp
except ImportError:
    SimuHawkes = None
    HawkesKernelExp = None


file_dir = os.path.dirname(__file__)


class Hawkes(DataModule):
    """
    Hawkes dataset.

    Args:
        data_dir (str): Where to download files.
        batch_size (int): Batch size. Default to 32
        prop_val (float): Proportion of validation. Default to .2
        num_workers (int): Number of workers for the loaders. Default to 0
        seed (int): For the random split. Default to 42
    """

    def __init__(
        self,
        data_dir: str = os.path.join(
            os.path.split(file_dir)[0],
            "data",
            "hawkes",
        ),
        batch_size: int = 32,
        prop_val: float = 0.2,
        num_workers: int = 0,
        seed: int = 42,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            prop_val=prop_val,
            num_workers=num_workers,
            seed=seed,
        )

        self.mu = [0.05, 0.05]
        self.alpha = [[0.1, 0.2], [0.2, 0.1]]
        self.decay = [[1.0, 1.0], [1.0, 1.0]]
        self.window = 1000

    def download(self, split: str = "train"):
        assert (
            SimuHawkes is not None
        ), "You must install tick to generate hawkes data."
        file = os.path.join(self.data_dir, f"{split}.npz")

        if split == "train":
            idx = range(1000)
        elif split == "test":
            idx = range(1000, 1100)
        else:
            raise NotImplementedError

        points = [
            self.generate_points(
                mu=self.mu,
                alpha=self.alpha,
                decay=self.decay,
                window=self.window,
                seed=i,
            )
            for i in idx
        ]

        with open(file, "wb") as fp:
            pkl.dump(obj=points, file=fp)

    def preprocess(self, split: str = "train") -> (th.Tensor, th.Tensor):
        file = os.path.join(self.data_dir, f"{split}.npz")

        # Load data
        with open(file, "rb") as fp:
            data = pkl.load(file=fp)

        # Create features
        features = pad_sequence(
            [self.get_features(x) for x in data],
            batch_first=True,
        ).unsqueeze(-1)

        # Create labels
        labels = pad_sequence(
            [self.get_labels(x) for x in data],
            batch_first=True,
        ).unsqueeze(-1)

        return features, labels

    @staticmethod
    def generate_points(
        mu: list,
        alpha: list,
        decay: list,
        window: int,
        seed: int,
        dt: float = 0.01,
    ):
        """
        Generates points of an marked Hawkes processes using the tick library.

        Args:
            mu (list): Hawkes baseline.
            alpha (list): Event parameter.
            decay (list): Decay parameter.
            window (int): The window of the simulated process.
            seed (int): The random seed.
            dt (float): Granularity. Default to 0.01
        """
        hawkes = SimuHawkes(
            n_nodes=len(mu), end_time=window, verbose=False, seed=seed
        )
        for i in range(len(mu)):
            for j in range(len(mu)):
                hawkes.set_kernel(
                    i=i,
                    j=j,
                    kernel=HawkesKernelExp(
                        intensity=alpha[i][j] / decay[i][j], decay=decay[i][j]
                    ),
                )
            hawkes.set_baseline(i, mu[i])

        hawkes.track_intensity(dt)
        hawkes.simulate()
        return hawkes.timestamps

    @staticmethod
    def get_features(point: list) -> th.Tensor:
        """
        Create features and labels from a hawkes process.

        Args:
            point (list): A hawkes process.
        """
        times = np.concatenate(point)
        sort_idx = np.argsort(times)
        times = times[sort_idx]
        return th.from_numpy(times)

    @staticmethod
    def get_labels(point: list) -> th.Tensor:
        """
        Create features and labels from a hawkes process.

        Args:
            point (list): A hawkes process.
        """
        times = np.concatenate(point)
        labels = np.concatenate([[i] * len(x) for i, x in enumerate(point)])
        sort_idx = np.argsort(times)
        labels = labels[sort_idx]
        return th.from_numpy(labels)
