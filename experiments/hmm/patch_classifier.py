import torch as th
import torch.nn as nn
from tint.models import Net
import torch
import torch.nn as nn
from transformers import PatchTSTModel, PatchTSTConfig
from torchmetrics import Accuracy, Precision, Recall, AUROC
from typing import Callable, Union
import pytorch_lightning as pl


class PatchClassifier(nn.Module):
    def __init__(self, feature_size: int, n_state: int, dropout: float = 0.5):
        super().__init__()
        self.n_state = n_state
        
        # # Configure the PatchTST model
        # config = PatchTSTConfig(
        #     num_input_channels=feature_size,
        #     context_length=177,
        #     patch_length=12,
        #     stride=12,
        #     use_cls_token=True,
        #     # Other configurations as needed
        # )
        config = PatchTSTConfig(
            num_input_channels=feature_size,  # feature_size should match the number of features in your data
            context_length=47,  # Adjusted to match input sequence length
            patch_length=12,
            stride=12,
            use_cls_token=True,
            # Other configurations as needed
        )

        self.patch_tst = PatchTSTModel(config)

        # Placeholder for BatchNorm1d, will be initialized after knowing the number of features
        self.bn = None
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(config.d_model, n_state)

    def forward(self, x):
        x = self.patch_tst(x).last_hidden_state
        cls_output = x[:, 0]

        # Dynamically initialize BatchNorm1d with the correct number of features
        if self.bn is None:
            self.bn = nn.BatchNorm1d(cls_output.size(1))
            self.bn = self.bn.to(cls_output.device)  # Move to the same device as the input

        cls_output = self.bn(cls_output)
        cls_output = self.dropout(cls_output)
        return self.linear(cls_output)

    #     # Configure the PatchTST model
    #     config = PatchTSTConfig(
    #         num_input_channels=feature_size,
    #         context_length=77,  # Adjust based on your sequence length
    #         patch_length=12,
    #         stride=12,
    #         use_cls_token=True,
    #         # Other configurations as needed
    #     )
    #     self.patch_tst = PatchTSTModel(config)

    #     # Regressor or classification head
    #     self.classifier = nn.Sequential(
    #         nn.BatchNorm1d(num_features=config.d_model),
    #         nn.GELU(),
    #         nn.Dropout(dropout),
    #         nn.Linear(config.d_model, n_state)
    #     )

    # def forward(self, x):
    #     x = self.patch_tst(x).last_hidden_state
    #     # Assuming we use the output corresponding to the CLS token
    #     cls_output = x[:, 0]
    #     return self.classifier(cls_output)

class PatchClassifierNet(pl.LightningModule):
    def __init__(
        self,
        feature_size: int,
        n_state: int,
        dropout: float = 0.5,
        loss: Union[str, Callable] = "cross_entropy",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        super().__init__()
        self.classifier = PatchClassifier(
            feature_size=feature_size,
            n_state=n_state,
            dropout=dropout
        )
        # Store optimizer and other parameters as instance attributes
        self._optim = optim  # Assigning the passed optimizer parameter to an instance attribute
        self.lr = lr
        self._lr_scheduler = lr_scheduler
        self._lr_scheduler_args = lr_scheduler_args
        self.l2 = l2
        
        for stage in ["train", "val", "test"]:
            setattr(self, stage + "_acc", Accuracy(task="binary"))
            setattr(self, stage + "_pre", Precision(task="binary"))
            setattr(self, stage + "_rec", Recall(task="binary"))
            setattr(self, stage + "_auroc", AUROC(task="binary"))
            
    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.classifier(*args, **kwargs)
    
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
    
    
    def training_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="train")
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="val")
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self.step(batch=batch, batch_idx=batch_idx, stage="test")
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        return self(x.float())
    
    def configure_optimizers(self):
        if self._optim == "adam":
            optim = th.optim.Adam(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                self.parameters(),
                lr=self.lr,
                weight_decay=self.l2,
                momentum=0.9,
                nesterov=True,
            )
        else:
            raise NotImplementedError

        lr_scheduler = self._lr_scheduler
        if lr_scheduler is not None:
            lr_scheduler = lr_scheduler.copy()
            lr_scheduler["scheduler"] = lr_scheduler["scheduler"](
                optim, **self._lr_scheduler_args
            )
            return {"optimizer": optim, "lr_scheduler": lr_scheduler}

        return {"optimizer": optim}