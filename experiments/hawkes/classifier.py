import torch as th

from torchmetrics import Accuracy, Precision, Recall, AUROC
from typing import Callable, Union

from tint.models import Net

from experiments.hmm.classifier import StateClassifier


class HawkesClassifierNet(Net):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        hidden_size: int,
        rnn: str = "GRU",
        dropout: float = 0.5,
        regres: bool = True,
        bidirectional: bool = False,
        window: int = 1000,
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

        self.window = window

        for stage in ["train", "val", "test"]:
            setattr(self, stage + "_acc", Accuracy())
            setattr(self, stage + "_pre", Precision())
            setattr(self, stage + "_rec", Recall())
            setattr(self, stage + "_auroc", AUROC())

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        x, y = batch

        idx = (x > 0).sum(1, keepdim=True)
        window = self.window * th.ones_like(x, device=x.device)
        t = th.randint(1, idx.min().item(), (1,)).item()

        x = th.cat(
            [x, th.zeros_like(x[:, 0, :].unsqueeze(1), device=x.device)], dim=1
        )
        x.scatter_(1, idx, window)
        x = th.cat([x[:, :t], x[:, 1 : t + 1], y[:, :t].unsqueeze(-1)], dim=-1)
        y = y[:, t]
        y_hat = self(x)
        loss = self.loss(y_hat, y)

        for metric in ["acc", "pre", "rec", "auroc"]:
            getattr(self, stage + "_" + metric)(y_hat[:, 1], y.long())
            self.log(stage + "_" + metric, getattr(self, stage + "_" + metric))

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        idx = (x > 0).sum(1, keepdim=True)
        window = self.window * th.ones_like(x, device=x.device)

        x = th.cat(
            [x, th.zeros_like(x[:, 0, :].unsqueeze(1), device=x.device)], dim=1
        )
        x.scatter_(1, idx, window)
        x = th.cat([x[:, :-1], x[:, 1:], y.unsqueeze(-1)], dim=-1)
        return self(x)
