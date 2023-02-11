import torch

from captum.log import log_usage
from captum._utils.common import _format_inputs
from captum._utils.typing import (
    TensorOrTupleOfTensorsGeneric,
    BaselineType,
    TargetType,
)

from torch import Tensor
from typing import Any, Callable, Tuple, Union

from .occlusion import Occlusion


class TemporalOcclusion(Occlusion):
    """
    Temporal Occlusion.

    This method modifies the original occlusion by only perturbing the last
    time, leaving the previous times unchanged. It can be used together with
    ``time_forward_tunnel`` to compute attributions on time series.

    Args:
        forward_func (callable): The forward function of the model or
            any modification of it.

    Examples:
        >>> import torch as th
        >>> from tint.attr import TemporalOcclusion
        >>> from tint.models import MLP
        <BLANKLINE>
        >>> inputs = th.rand(8, 7, 5)
        >>> mlp = MLP([5, 3, 1])
        <BLANKLINE>
        >>> explainer = TemporalOcclusion(mlp)
        >>> attr = explainer.attribute(inputs, (1,))
    """

    def __init__(self, forward_func: Callable):
        super().__init__(forward_func=forward_func)

    @log_usage()
    def attribute(  # type: ignore
        self,
        inputs: TensorOrTupleOfTensorsGeneric,
        sliding_window_shapes: Union[
            Tuple[int, ...], Tuple[Tuple[int, ...], ...]
        ],
        strides: Union[
            None, int, Tuple[int, ...], Tuple[Union[int, Tuple[int, ...]], ...]
        ] = None,
        baselines: BaselineType = None,
        target: TargetType = None,
        additional_forward_args: Any = None,
        perturbations_per_eval: int = 1,
        attributions_fn: Callable = None,
        show_progress: bool = False,
    ) -> TensorOrTupleOfTensorsGeneric:
        r"""
        Args:

                inputs (tensor or tuple of tensors):  Input for which occlusion
                            attributions are computed. If forward_func takes a single
                            tensor as input, a single input tensor should be provided.
                            If forward_func takes multiple tensors as input, a tuple
                            of the input tensors should be provided. It is assumed
                            that for all given input tensors, dimension 0 corresponds
                            to the number of examples (aka batch size), and if
                            multiple input tensors are provided, the examples must
                            be aligned appropriately.
                sliding_window_shapes (tuple or tuple of tuples): Shape of patch
                            (hyperrectangle) to occlude each input. For a single
                            input tensor, this must be a tuple of length equal to the
                            number of dimensions of the input tensor - 2, defining
                            the dimensions of the patch. If the input tensor is 2-d,
                            this should be an empty tuple. For multiple input tensors,
                            this must be a tuple containing one tuple for each input
                            tensor defining the dimensions of the patch for that
                            input tensor, as described for the single tensor case.
                strides (int or tuple or tuple of ints or tuple of tuples, optional):
                            This defines the step by which the occlusion hyperrectangle
                            should be shifted by in each direction for each iteration.
                            For a single tensor input, this can be either a single
                            integer, which is used as the step size in each direction,
                            or a tuple of integers matching the number of dimensions
                            in the occlusion shape, defining the step size in the
                            corresponding dimension. For multiple tensor inputs, this
                            can be either a tuple of integers, one for each input
                            tensor (used for all dimensions of the corresponding
                            tensor), or a tuple of tuples, providing the stride per
                            dimension for each tensor.
                            To ensure that all inputs are covered by at least one
                            sliding window, the stride for any dimension must be
                            <= the corresponding sliding window dimension if the
                            sliding window dimension is less than the input
                            dimension.
                            If None is provided, a stride of 1 is used for each
                            dimension of each input tensor.
                            Default: None
                baselines (scalar, tensor, tuple of scalars or tensors, optional):
                            Baselines define reference value which replaces each
                            feature when occluded.
                            Baselines can be provided as:

                            - a single tensor, if inputs is a single tensor, with
                              exactly the same dimensions as inputs or
                              broadcastable to match the dimensions of inputs

                            - a single scalar, if inputs is a single tensor, which will
                              be broadcasted for each input value in input tensor.

                            - a tuple of tensors or scalars, the baseline corresponding
                              to each tensor in the inputs' tuple can be:

                              - either a tensor with matching dimensions to
                                corresponding tensor in the inputs' tuple
                                or the first dimension is one and the remaining
                                dimensions match with the corresponding
                                input tensor.

                              - or a scalar, corresponding to a tensor in the
                                inputs' tuple. This scalar value is broadcasted
                                for corresponding input tensor.

                            In the cases when `baselines` is not provided, we internally
                            use zero scalar corresponding to each input tensor.
                            Default: None
                target (int, tuple, tensor or list, optional):  Output indices for
                            which difference is computed (for classification cases,
                            this is usually the target class).
                            If the network returns a scalar value per example,
                            no target index is necessary.
                            For general 2D outputs, targets can be either:

                            - a single integer or a tensor containing a single
                              integer, which is applied to all input examples

                            - a list of integers or a 1D tensor, with length matching
                              the number of examples in inputs (dim 0). Each integer
                              is applied as the target for the corresponding example.

                            For outputs with > 2 dimensions, targets can be either:

                            - A single tuple, which contains #output_dims - 1
                              elements. This target index is applied to all examples.

                            - A list of tuples with length equal to the number of
                              examples in inputs (dim 0), and each tuple containing
                              #output_dims - 1 elements. Each tuple is applied as the
                              target for the corresponding example.

                            Default: None
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
                            correspond to the number of examples. For all other types,
                            the given argument is used for all forward evaluations.
                            Note that attributions are not computed with respect
                            to these arguments.
                            Default: None
                perturbations_per_eval (int, optional): Allows multiple occlusions
                            to be included in one batch (one call to forward_fn).
                            By default, perturbations_per_eval is 1, so each occlusion
                            is processed individually.
                            Each forward pass will contain a maximum of
                            perturbations_per_eval * #examples samples.
                            For DataParallel models, each batch is split among the
                            available devices, so evaluations on each available
                            device contain at most
                            (perturbations_per_eval * #examples) / num_devices
                            samples.
                            Default: 1
                attributions_fn (Callable, optional): Applies a function to the
                        attributions before performing the weighted sum.
                        Default: None
                show_progress (bool, optional): Displays the progress of computation.
                            It will try to use tqdm if available for advanced features
                            (e.g. time estimation). Otherwise, it will fallback to
                            a simple output of progress.
                            Default: False

        Returns:
                *tensor* or tuple of *tensors* of **attributions**:
                - **attributions** (*tensor* or tuple of *tensors*):
                            The attributions with respect to each input feature.
                            Attributions will always be
                            the same size as the provided inputs, with each value
                            providing the attribution of the corresponding input index.
                            If a single tensor is provided as inputs, a single tensor is
                            returned. If a tuple is provided for inputs, a tuple of
                            corresponding sized tensors is returned.
        """

        inputs_tpl = _format_inputs(inputs)

        assert all(
            x.shape[1] == inputs_tpl[0].shape[1] for x in inputs_tpl
        ), "All inputs must have the same time dimension. (dimension 1)"

        # The time sliding must be equal to the time dim as we only
        # perform the perturbation on the last time
        sliding_window_shapes = (
            inputs_tpl[0].shape[1],
        ) + sliding_window_shapes

        # Append one stride on the time dimension
        if strides is not None:
            strides = (1,) + strides

        return super().attribute.__wrapped__(
            self,
            inputs=inputs,
            sliding_window_shapes=sliding_window_shapes,
            strides=strides,
            baselines=baselines,
            target=target,
            additional_forward_args=additional_forward_args,
            perturbations_per_eval=perturbations_per_eval,
            attributions_fn=attributions_fn,
            show_progress=show_progress,
        )

    def _construct_ablated_input(
        self,
        expanded_input: Tensor,
        input_mask: Union[None, Tensor],
        baseline: Union[Tensor, int, float],
        start_feature: int,
        end_feature: int,
        **kwargs: Any,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Ablates given expanded_input tensor with given feature mask, feature range,
        and baselines, and any additional arguments.
        expanded_input shape is (num_features, num_examples, ...)
        with remaining dimensions corresponding to remaining original tensor
        dimensions and num_features = end_feature - start_feature.

        input_mask is None for occlusion, and the mask is constructed
        using sliding_window_tensors, strides, and shift counts, which are provided in
        kwargs. baseline is expected to
        be broadcastable to match expanded_input.

        This method returns the ablated input tensor, which has the same
        dimensionality as expanded_input as well as the corresponding mask with
        either the same dimensionality as expanded_input or second dimension
        being 1. This mask contains 1s in locations which have been ablated (and
        thus counted towards ablations for that feature) and 0s otherwise.
        """
        input_mask = torch.stack(
            [
                self._occlusion_mask(
                    expanded_input,
                    j,
                    kwargs["sliding_window_tensors"],
                    kwargs["strides"],
                    kwargs["shift_counts"],
                )
                for j in range(start_feature, end_feature)
            ],
            dim=0,
        ).long()

        # Only apply occlusion on the last time
        input_mask[:, :, :-1] = 0

        ablated_tensor = (
            expanded_input
            * (
                torch.ones(1, dtype=torch.long, device=expanded_input.device)
                - input_mask
            ).to(expanded_input.dtype)
        ) + (baseline * input_mask.to(expanded_input.dtype))
        return ablated_tensor, input_mask
