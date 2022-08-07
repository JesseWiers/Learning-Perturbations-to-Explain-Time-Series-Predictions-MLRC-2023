import torch

from captum.log import log_usage
from captum._utils.common import _select_targets
from captum._utils.typing import (
    BaselineType,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)

from torch import Tensor
from typing import Any, Callable

from .base import _base_metric


def _cross_entropy(
    prob_original: Tensor, prob_pert: Tensor, target: Tensor
) -> Tensor:
    return -_select_targets(torch.log(prob_pert), target)


@log_usage()
def cross_entropy(
    forward_func: Callable,
    inputs: TensorOrTupleOfTensorsGeneric,
    attributions: TensorOrTupleOfTensorsGeneric,
    baselines: BaselineType = None,
    additional_forward_args: Any = None,
    target: TargetType = None,
    topk: float = 0.2,
) -> Tensor:
    return _base_metric(
        metric=_cross_entropy,
        forward_func=forward_func,
        inputs=inputs,
        attributions=attributions,
        baselines=baselines,
        additional_forward_args=additional_forward_args,
        target=target,
        topk=topk,
        largest=True,
    )
