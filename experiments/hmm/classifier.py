import torch as th
import torch.nn as nn

from torchmetrics import Accuracy, Precision, Recall, AUROC
from typing import Callable, Union

from tint.models import Net


class StateClassifier(nn.Module):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_state = n_state
        self.rnn_type = rnn
        self.regres = regres
        # Input to torch LSTM should be of size (batch, seq_len, input_size)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )
        else:
            self.rnn = nn.LSTM(
                feature_size,
                self.hidden_size,
                bidirectional=bidirectional,
                batch_first=True,
            )

        self.regressor = nn.Sequential(
            nn.BatchNorm1d(num_features=self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.n_state),
        )

    def forward(self, x, return_all: bool = False):
        if self.rnn_type == "GRU":
            all_encodings, encoding = self.rnn(x)
        else:
            all_encodings, (encoding, state) = self.rnn(x)

        if self.regres:
            if return_all:
                reshaped_encodings = all_encodings.reshape(
                    all_encodings.shape[0] * all_encodings.shape[1], -1
                )
                return self.regressor(reshaped_encodings).reshape(
                    all_encodings.shape[0], all_encodings.shape[1], -1
                )
            return self.regressor(encoding.reshape(encoding.shape[1], -1))
        return encoding.reshape(encoding.shape[1], -1)


class StateClassifierNet(Net):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        classifier = StateClassifier(
            feature_size=feature_size,
            n_state=n_state,
            hidden_size=hidden_size,
            rnn=rnn,
            dropout=dropout,
            regres=regres,
            bidirectional=bidirectional,
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
