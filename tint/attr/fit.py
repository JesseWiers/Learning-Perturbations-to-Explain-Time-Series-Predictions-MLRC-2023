import copy
import torch as th
import torch.nn.functional as F

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_inputs,
    _format_output,
    _is_tuple,
    _run_forward,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from pytorch_lightning import LightningDataModule, Trainer
from torch.utils.data import DataLoader, TensorDataset
from typing import Any, Callable, Tuple

from tint.utils import get_progress_bars, _add_temporal_mask, _slice_to_time

from .models import JointFeatureGeneratorNet


def kl_multilabel(p1, p2, reduction="none"):
    # treats each column as separate class and calculates KL over the class,
    # sums it up and sends batched
    n_classes = p1.shape[1]
    total_kl = th.zeros(p1.shape).to(p1.device)

    for n in range(n_classes):
        p2_tensor = th.stack([p2[:, n], 1 - p2[:, n]], dim=1)
        p1_tensor = th.stack([p1[:, n], 1 - p1[:, n]], dim=1)
        kl = F.kl_div(th.log(p2_tensor), p1_tensor, reduction=reduction)
        total_kl[:, n] = th.sum(kl, dim=1)

    return total_kl


class Fit(PerturbationAttribution):
    """
    Feature Importance in Time.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.
        generator (JointFeatureGeneratorNet): Conditional generator model to
            predict future observations as a Pytorch Lightning module.
            If not provided, a default generator is created.
            Default to ``None``
        datamodule (LightningDataModule): A Pytorch Lightning data
            module to train the generator. If not provided, you must provide
            features. Default to ``None``
        features (th.Tensor): A tensor of features to train the generator.
            If not provided, you must provide a datamodule.
            Default to ``None``
        trainer (Trainer): A Pytorch Lightning trainer to train the generator.
            If not provided, a default trainer is created. Default to ``None``
        batch_size (int): Batch size for generator training. Default to 32

    References:
        `What went wrong and when? Instance-wise Feature Importance for Time-series Models <https://arxiv.org/abs/2003.02821>`_

    Examples:
        >>> import torch as th
        >>> from tint.attr import Fit
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = Fit(mlp, features=data)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(
        self,
        forward_func: Callable,
        generator: JointFeatureGeneratorNet = None,
        datamodule: LightningDataModule = None,
        features: th.Tensor = None,
        trainer: Trainer = None,
        batch_size: int = 32,
    ) -> None:
        super().__init__(forward_func=forward_func)

        # Create dataloader if not provided
        dataloader = None
        if datamodule is None:
            assert (
                features is not None
            ), "You must provide either a datamodule or features"

            dataloader = DataLoader(
                TensorDataset(features),
                batch_size=batch_size,
            )

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=300)
        else:
            trainer = copy.deepcopy(trainer)

        # Create generator if not provided
        if generator is None:
            generator = JointFeatureGeneratorNet()
        else:
            generator = copy.deepcopy(generator)

        # Init generator with feature size
        if features is None:
            shape = next(iter(datamodule.train_dataloader()))[0].shape
        else:
            shape = features.shape
        generator.net.init(feature_size=shape[-1])

        # Train generator
        trainer.fit(
            generator, train_dataloaders=dataloader, datamodule=datamodule
        )

        # Set to eval mode
        generator.eval()

        # Extract generator model from pytorch lightning model
        self.generator = generator.net

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Any = None,
        n_samples: int = 10,
        distance_metric: str = "kl",
        multilabel: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        attribute method.

        Args:
            inputs (tensor or tuple of tensors):  Input for which integrated
                gradients are computed. If forward_func takes a single
                tensor as input, a single input tensor should be provided.
                If forward_func takes multiple tensors as input, a tuple
                of the input tensors should be provided. It is assumed
                that for all given input tensors, dimension 0 corresponds
                to the number of examples, and if multiple input tensors
                are provided, the examples must be aligned appropriately.
            additional_forward_args (any, optional): If the forward function
                requires additional arguments other than the inputs for
                which attributions should not be computed, this argument
                can be provided. It must be either a single additional
                argument of a Tensor or arbitrary (non-tuple) type or a
                tuple containing multiple additional arguments including
                tensors or any arbitrary python types. These arguments
                are provided to forward_func in order following the
                arguments in inputs.
                For a tensor, the first dimension of the tensor must
                correspond to the number of examples. It will be
                repeated for each of `n_steps` along the integrated
                path. For all other types, the given argument is used
                for all forward evaluations.
                Note that attributions are not computed with respect
                to these arguments.
                Default: None
            n_samples (int): Number of Monte-Carlo samples.
                Default: 10
            distance_metric (str): Distance metric.
                Default to 'kl'
            multilabel (bool): Whether the task is single or multi-labeled.
                Default: False
            temporal_additional_forward_args (tuple): Set each
                additional forward arg which is temporal.
                Only used with return_temporal_attributions.
                Default: None
            return_temporal_attributions (bool): Whether to return
                attributions for all times or not.
                Default: False
            show_progress (bool, optional): Displays the progress of computation.
                It will try to use tqdm if available for advanced features
                (e.g. time estimation). Otherwise, it will fallback to
                a simple output of progress.
                Default: False

        Returns:
            - **attributions** (*tensor* or tuple of *tensors*):
                The attributions with respect to each input feature.
                Attributions will always be
                the same size as the provided inputs, with each value
                providing the attribution of the corresponding input index.
                If a single tensor is provided as inputs, a single tensor is
                returned. If a tuple is provided for inputs, a tuple of
                corresponding sized tensors is returned.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_inputs(inputs)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"
        data = inputs[0]

        # Set generator to device
        self.generator.to(data.device)

        if return_temporal_attributions:
            data, additional_forward_args, _ = _add_temporal_mask(
                inputs=data,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )

        attributions = (
            self.representation(
                inputs=data,
                additional_forward_args=additional_forward_args,
                n_samples=n_samples,
                distance_metric=distance_metric,
                multilabel=multilabel,
                temporal_additional_forward_args=temporal_additional_forward_args,
                show_progress=show_progress,
            ),
        )

        if return_temporal_attributions:
            attributions = (
                attributions[0].reshape((-1, data.shape[1]) + data.shape[1:]),
            )

        return _format_output(is_inputs_tuple, attributions)

    def representation(
        self,
        inputs: th.Tensor,
        additional_forward_args: Any = None,
        n_samples: int = 10,
        distance_metric: str = "kl",
        multilabel: bool = False,
        temporal_additional_forward_args: Tuple[bool] = None,
        show_progress: bool = False,
    ):
        """
        Get representations based on a generator and inputs.

        Args:
            inputs (th.Tensor): Input data.
            additional_forward_args (Any): Optional additional args to be
                passed into the model. Default to ``None``
            n_samples (int): Number of Monte-Carlo samples. Default to 10
            distance_metric (str): Distance metric. Default to ``'kl'``
            multilabel (bool): Whether the task is single or multi-labeled.
                Default to ``False``
            temporal_additional_forward_args (tuple): Set each
                additional forward arg which is temporal.
                Only used with return_temporal_attributions.
                Default to ``None``
            show_progress (bool): Displays the progress of computation.
                Default to False

        Returns:
            th.Tensor: attributions.
        """
        assert distance_metric in [
            "kl",
            "mean_divergence",
            "LHS",
            "RHS",
        ], "Unrecognised distance metric."

        _, t_len, n_features = inputs.shape
        score = th.zeros(inputs.shape).to(inputs.device)

        if multilabel:
            activation = F.sigmoid
        else:
            activation = lambda x: F.softmax(x, -1)

        t_range = range(1, t_len)
        if show_progress:
            t_range = get_progress_bars()(
                t_range,
                desc=f"{self.get_name()} attribution",
            )

        for t in t_range:
            partial_inputs, kwargs_copy = _slice_to_time(
                inputs=inputs,
                time=t + 1,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )
            p_y_t = activation(
                _run_forward(
                    forward_func=self.forward_func,
                    inputs=partial_inputs,
                    additional_forward_args=kwargs_copy[
                        "additional_forward_args"
                    ],
                )
            )

            partial_inputs, kwargs_copy = _slice_to_time(
                inputs=inputs,
                time=t,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )
            p_tm1 = activation(
                _run_forward(
                    forward_func=self.forward_func,
                    inputs=partial_inputs,
                    additional_forward_args=kwargs_copy[
                        "additional_forward_args"
                    ],
                )
            )

            for i in range(n_features):
                x_hat, kwargs_copy = _slice_to_time(
                    inputs=inputs,
                    time=t + 1,
                    additional_forward_args=additional_forward_args,
                    temporal_additional_forward_args=temporal_additional_forward_args,
                )
                div_all = []

                for _ in range(n_samples):
                    x_hat_t, _ = self.generator.forward_conditional(
                        inputs[:, :t, :], inputs[:, t, :], [i]
                    )
                    x_hat[:, t, :] = x_hat_t
                    y_hat_t = activation(
                        _run_forward(
                            forward_func=self.forward_func,
                            inputs=x_hat,
                            additional_forward_args=kwargs_copy[
                                "additional_forward_args"
                            ],
                        )
                    )

                    if distance_metric == "kl":
                        if not multilabel:
                            div = th.sum(
                                F.kl_div(
                                    th.log(p_tm1), p_y_t, reduction="none"
                                ),
                                -1,
                            ) - th.sum(
                                F.kl_div(
                                    th.log(y_hat_t), p_y_t, reduction="none"
                                ),
                                -1,
                            )
                        else:
                            t1 = kl_multilabel(p_y_t, p_tm1)
                            t2 = kl_multilabel(p_y_t, y_hat_t)
                            div, _ = th.max(t1 - t2, dim=1)
                        div_all.append(div.cpu().detach())

                    elif distance_metric == "mean_divergence":
                        div = th.abs(y_hat_t - p_y_t)
                        div_all.append(th.mean(div.detach().cpu(), -1))

                    elif distance_metric == "LHS":
                        div = th.sum(
                            F.kl_div(th.log(p_tm1), p_y_t, reduction="none"),
                            -1,
                        )
                        div_all.append(div.cpu().detach())

                    elif distance_metric == "RHS":
                        div = th.sum(
                            F.kl_div(th.log(y_hat_t), p_y_t, reduction="none"),
                            -1,
                        )
                        div_all.append(div.cpu().detach())
                    else:
                        raise NotImplementedError

                e_div = th.stack(div_all).mean(0)
                if distance_metric == "kl":
                    score[:, t, i] = 2.0 / (1 + th.exp(-5 * e_div)) - 1
                elif distance_metric == "mean_divergence":
                    score[:, t, i] = 1 - e_div
                else:
                    score[:, t, i] = e_div

        return score
