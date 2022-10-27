import torch as th
import torch.nn as nn

from captum._utils.common import _run_forward

from typing import Callable, Union

from tint.models import Net


class ExtremalMask(nn.Module):
    """
    Extremal Mask model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
    """

    def __init__(
        self,
        forward_func: Callable,
        model: nn.Module = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "forward_func", forward_func)
        self.model = model
        self.batch_size = batch_size

        self.input_size = None
        self.register_parameter("mask", None)

    def init(self, input_size: tuple, batch_size: int = 32) -> None:
        self.input_size = input_size
        self.batch_size = batch_size

        self.mask = nn.Parameter(th.Tensor(*input_size))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.mask.data.fill_(0.5)

    def forward(
        self,
        x: th.Tensor,
        batch_idx,
        baselines,
        target,
        *additional_forward_args,
    ) -> (th.Tensor, th.Tensor):
        mask = self.mask

        # Subset sample to current batch
        mask = mask[
            self.batch_size * batch_idx : self.batch_size * (batch_idx + 1)
        ]

        # We forward model if provided
        if self.model is not None:
            mask = mask + self.model(x - baselines)

        # We clamp the mask
        mask = mask.clamp(0, 1)

        # Mask data according to samples
        # We eventually cut samples up to x time dimension
        # x1 represents inputs with important features masked.
        # x2 represents inputs with unimportant features masked.
        mask = mask[:, : x.shape[1], ...]
        x1 = x * mask + baselines * (1.0 - mask)
        x2 = x * (1.0 - mask) + baselines * mask

        # Return f(perturbed x)
        return (
            _run_forward(
                forward_func=self.forward_func,
                inputs=x1,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
            _run_forward(
                forward_func=self.forward_func,
                inputs=x2,
                target=target,
                additional_forward_args=additional_forward_args,
            ),
        )

    def representation(self, data, baselines):
        if self.model is None:
            return self.mask.detach().cpu().clamp(0, 1)
        return (
            (self.mask + self.model(data - baselines))
            .detach()
            .cpu()
            .clamp(0, 1)
        )


class ExtremalMaskNet(Net):
    """
    Extremal mask model as a Pytorch Lightning model.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        comp_loss (bool): Whether to include comprehensiveness loss.
            Default to ``False``
        model (nnn.Module): A model used to recreate the original
            predictions, in addition to the mask. Default to ``None``
        batch_size (int): Batch size of the model. Default to 32
        loss (str, callable): Which loss to use. Default to ``'mse'``
        optim (str): Which optimizer to use. Default to ``'adam'``
        lr (float): Learning rate. Default to 1e-3
        lr_scheduler (dict, str): Learning rate scheduler. Either a dict
            (custom scheduler) or a string. Default to ``None``
        lr_scheduler_args (dict): Additional args for the scheduler.
            Default to ``None``
        l2 (float): L2 regularisation. Default to 0.0

    Examples:
        >>> import torch as th
        >>> from tint.attr.models import ExtremalMaskNet
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> mask = ExtremalMaskNet(
        ...     forward_func=mlp,
        ...     optim="adam",
        ...     lr=0.01,
        ... )
    """

    def __init__(
        self,
        forward_func: Callable,
        comp_loss: bool = False,
        model: nn.Module = None,
        batch_size: int = 32,
        loss: Union[str, Callable] = "mse",
        optim: str = "adam",
        lr: float = 0.001,
        lr_scheduler: Union[dict, str] = None,
        lr_scheduler_args: dict = None,
        l2: float = 0.0,
    ):
        mask = ExtremalMask(
            forward_func=forward_func,
            model=model,
            batch_size=batch_size,
        )

        super().__init__(
            layers=mask,
            loss=loss,
            optim=optim,
            lr=lr,
            lr_scheduler=lr_scheduler,
            lr_scheduler_args=lr_scheduler_args,
            l2=l2,
        )

        self.comp_loss = comp_loss

    def forward(self, *args, **kwargs) -> th.Tensor:
        return self.net(*args, **kwargs)

    def step(self, batch, batch_idx, stage):
        # x is the data to be perturbed
        # y is the same data without perturbation
        x, y, baselines, target, *additional_forward_args = batch

        # If additional_forward_args is only one None,
        # set it to None
        if additional_forward_args == [None]:
            additional_forward_args = None

        # Get perturbed output
        # y_hat1 is computed by masking important features
        # y_hat2 is computed by masking unimportant features
        if additional_forward_args is None:
            y_hat1, y_hat2 = self(x.float(), batch_idx, baselines, target)
        else:
            y_hat1, y_hat2 = self(
                x.float(),
                batch_idx,
                baselines,
                target,
                *additional_forward_args,
            )

        # Get unperturbed output for inputs and baselines
        y_target1 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=y,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )
        y_target2 = _run_forward(
            forward_func=self.net.forward_func,
            inputs=th.zeros_like(y) + baselines,
            target=target,
            additional_forward_args=tuple(additional_forward_args)
            if additional_forward_args is not None
            else None,
        )

        # Add L1 loss
        mask_ = self.net.mask
        if self.net.model is not None:
            mask_ = mask_[
                self.net.batch_size
                * batch_idx : self.net.batch_size
                * (batch_idx + 1)
            ]
            mask_ = mask_ + self.net.model(x - baselines)
        reg_loss = mask_.abs().mean()

        # Compute and return loss
        if self.comp_loss:
            return (
                self.loss(y_hat1, y_target1)
                + self.loss(y_hat2, y_target2)
                + reg_loss
            )
        return self.loss(y_hat1, y_target1) + reg_loss

    def configure_optimizers(self):
        params = [{"params": self.net.mask}]

        if self.net.model is not None:
            params += [{"params": self.net.model.parameters()}]

        if self._optim == "adam":
            optim = th.optim.Adam(
                params=params,
                lr=self.lr,
                weight_decay=self.l2,
            )
        elif self._optim == "sgd":
            optim = th.optim.SGD(
                params=params,
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
