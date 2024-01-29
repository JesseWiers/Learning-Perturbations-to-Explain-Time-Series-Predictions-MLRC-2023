import torch as th
import torch.nn as nn

from torchmetrics import Accuracy, Precision, Recall, AUROC
from typing import Callable, Union

from tint.models import Net

from mamba import Mamba, ModelArgs


class MambaStateClassifier(nn.Module):
    def __init__(self, feature_size: int, n_state: int, dropout: float = 0.5, regres: bool = True):
        super().__init__()
        self.n_state = n_state
        self.regres = regres
        
        # Initialize Mamba model
        mamba_args = ModelArgs(
            d_model=feature_size,
            n_layer=4,  # Number of layers in Mamba
            vocab_size=n_state  # Vocab size for Mamba
            # You can add other Mamba-specific arguments here
        )
        self.mamba = Mamba(mamba_args)

        # Regressor layer
        if self.regres:
            self.regressor = nn.Sequential(
                nn.BatchNorm1d(num_features=feature_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(feature_size, n_state),
            )

    def forward(self, x, return_all: bool = False):
        mamba_output = self.mamba(x)

        if self.regres:
            if return_all:
                # Handling sequence output for regressor
                return self.regressor(mamba_output)
            return self.regressor(mamba_output[:, -1, :])
        return mamba_output[:, -1, :]


class MambaClassifierNet(Net):
    def __init__(self, feature_size: int, n_state: int, dropout: float = 0.5, regres: bool = True, loss: Union[str, Callable] = "mse", optim: str = "adam", lr: float = 0.001, lr_scheduler: Union[dict, str] = None, lr_scheduler_args: dict = None, l2: float = 0.0):
        classifier = MambaStateClassifier(
            feature_size=feature_size,
            n_state=n_state,
            dropout=dropout,
            regres=regres,
        )

        super().__init__(
            layers=classifier,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )
        self.save_hyperparameters()

        for stage in ["train", "val", "test"]:
            setattr(self, stage + "_acc", Accuracy(task="binary"))
            setattr(self, stage + "_pre", Precision(task="binary"))
            setattr(self, stage + "_rec", Recall(task="binary"))
            setattr(self, stage + "_auroc", AUROC(task="binary"))
            
    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        t = th.randint(batch[1].shape[-1], (1,)).item()
        x, y = batch
        x = x[:, : t + 1]
        y = y[:, t]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        for metric in ["acc", "pre", "rec", "auroc"]:
            getattr(self, stage + "_" + metric)(y_hat[:, 1], y.long())
            self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x)