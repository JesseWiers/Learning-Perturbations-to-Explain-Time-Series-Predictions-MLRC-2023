from captum.log import log_usage
from captum._utils.common import (
    _is_tuple,
    _format_inputs,
    _format_output,
    _validate_input,
)
from captum._utils.typing import TensorOrTupleOfTensorsGeneric

from typing import Callable, Tuple

EPS = 1e-5


@log_usage()
def _base_white_box_metric(
    metric: Callable,
    attributions: TensorOrTupleOfTensorsGeneric,
    true_attributions: TensorOrTupleOfTensorsGeneric,
    normalize: bool = True,
    hard_labels: bool = True,
) -> Tuple[float]:
    # Convert attributions into tuple
    is_inputs_tuple = _is_tuple(attributions)
    attributions = _format_inputs(attributions)
    true_attributions = _format_inputs(true_attributions)

    # Validate input
    _validate_input(attributions, true_attributions)

    # Normalise attributions
    if normalize:
        min_tpl = tuple(
            attr.reshape(len(attr), -1).min(dim=-1).values
            for attr in attributions
        )
        max_tpl = tuple(
            attr.reshape(len(attr), -1).max(dim=-1).values
            for attr in attributions
        )
        min_tpl = tuple(
            min_.view((len(attr),) + (1,) * (len(attr.shape) - 1))
            for min_, attr in zip(min_tpl, attributions)
        )
        max_tpl = tuple(
            max_.view((len(attr),) + (1,) * (len(attr.shape) - 1))
            for max_, attr in zip(max_tpl, attributions)
        )
        attributions = tuple(
            (attr - min_) / (max_ - min_ + EPS)
            for attr, min_, max_ in zip(attributions, min_tpl, max_tpl)
        )

    # Reshape attributions
    attributions = tuple(attr.reshape(-1) for attr in attributions)
    true_attributions = tuple(attr.reshape(-1) for attr in true_attributions)

    # Get int value if necessary
    if hard_labels:
        true_attributions = tuple(attr.int() for attr in true_attributions)

    # Get a subset corresponding to the true attributions
    attributions_subset = tuple(
        attr[true_attr != 0]
        for attr, true_attr in zip(attributions, true_attributions)
    )

    # Convert to numpy
    attributions = tuple(attr.detach().cpu().numpy() for attr in attributions)
    true_attributions = tuple(
        attr.detach().cpu().numpy() for attr in true_attributions
    )
    attributions_subset = tuple(
        attr.detach().cpu().numpy() for attr in attributions_subset
    )

    # Compute metric
    output = metric(attributions, true_attributions, attributions_subset)

    # Return output
    return _format_output(is_inputs_tuple, output)  # type: ignore
