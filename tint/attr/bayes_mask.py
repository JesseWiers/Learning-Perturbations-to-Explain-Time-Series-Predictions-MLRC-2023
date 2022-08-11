import copy
import torch as th

from captum.attr._utils.attribution import PerturbationAttribution
from captum.log import log_usage
from captum._utils.common import (
    _format_input,
    _format_output,
    _is_tuple,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from typing import Any, Callable, Tuple

from tint.utils import TensorDataset, _add_temporal_mask, default_collate
from .models import BayesMaskNet


class BayesMask(PerturbationAttribution):
    """
    Bayes masks method.

    Args:
        forward_func (callable): The forward function of the model or any
            modification of it.

    Examples:
        >>> import torch as th
        >>> from tint.attr import BayesMask
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> data = th.rand(32, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = BayesMask(mlp)
        >>> attr = explainer.attribute(inputs)
    """

    def __init__(self, forward_func: Callable) -> None:
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        additional_forward_args: Any = None,
        trainer: Trainer = None,
        mask_net: BayesMaskNet = None,
        batch_size: int = 32,
        temporal_additional_forward_args: Tuple[bool] = None,
        return_temporal_attributions: bool = False,
        return_covariance: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        """
        Attribute method.

        Args:
            inputs (tuple, th.Tensor): Input data.
            additional_forward_args (Any): Any additional argument passed
                to the model. Default to ``None``
            trainer (Trainer): Pytorch Lightning trainer. If ``None``, a
                default trainer will be provided. Default to ``None``
            mask_net (BayesMaskNet): A Mask model. If ``None``, a default model
                will be provided. Default to ``None``
            batch_size (int): Batch size for Mask training. Default to 32
            temporal_additional_forward_args (tuple): Set each
                additional forward arg which is temporal.
                Only used with return_temporal_attributions.
                Default to ``None``
            return_temporal_attributions (bool): Whether to return
                attributions for all times or not. Default to ``False``
            return_covariance (bool): Whether to return the covariance of the
                bayes mask network or not. Default to ``False``

        Returns:
            (th.Tensor, tuple): Attributions.
        """
        # Keeps track whether original input is a tuple or not before
        # converting it into a tuple.
        is_inputs_tuple = _is_tuple(inputs)
        inputs = _format_input(inputs)

        # Init trainer if not provided
        if trainer is None:
            trainer = Trainer(max_epochs=100)
        else:
            trainer = copy.deepcopy(trainer)

        # Assert only one input, as the Retain only accepts one
        assert (
            len(inputs) == 1
        ), "Multiple inputs are not accepted for this method"
        data = inputs[0]

        # If return temporal attr, we expand the input data
        # and multiply it with a lower triangular mask
        if return_temporal_attributions:
            data, additional_forward_args, _ = _add_temporal_mask(
                inputs=data,
                additional_forward_args=additional_forward_args,
                temporal_additional_forward_args=temporal_additional_forward_args,
            )

        # Init MaskNet if not provided
        if mask_net is None:
            mask_net = BayesMaskNet(forward_func=self.forward_func)
        else:
            mask_net = copy.deepcopy(mask_net)

        # Init model
        mask_net.net.init(input_size=data.shape, batch_size=batch_size)

        # Prepare data
        dataloader = DataLoader(
            TensorDataset(
                *(data, data, *additional_forward_args)
                if additional_forward_args is not None
                else (data, data, None)
            ),
            batch_size=batch_size,
            collate_fn=default_collate,
        )

        # Fit model
        trainer.fit(mask_net, train_dataloaders=dataloader)

        # Set model to eval mode
        mask_net.eval()

        # Get attributions as mask representation
        attributions = mask_net.net.representation()

        # Reshape representation if temporal attributions
        if return_temporal_attributions:
            attributions = attributions.reshape(
                (-1, data.shape[1]) + data.shape[1:]
            )

        # Reshape as a tuple
        attributions = (attributions,)

        if return_covariance:
            covariance = mask_net.net.covariance()

            # Reshape representation if temporal attributions
            if return_temporal_attributions:
                covariance = covariance.reshape(
                    (-1, data.shape[1]) + data.shape[1:] + (data.shape[-1],)
                )

            # Reshape as a tuple
            covariance = (covariance,)

            return (
                _format_output(is_inputs_tuple, attributions),
                _format_output(is_inputs_tuple, covariance),
            )

        # Format attributions and return
        return _format_output(is_inputs_tuple, attributions)
