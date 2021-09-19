import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import _reduction as _Reduction
import numpy as np
from collections import Counter



def mse_loss(input, target, mask, size_average=None, reduce=None, reduction='mean'):
    # type: # (Tensor, Tensor, Optional[bool], Optional[bool], str) -> Tensor
    r"""mse_loss(input, target, size_average=None, reduce=None, reduction='mean') -> Tensor

    Measures the element-wise mean squared error.

    See :class:`~torch.nn.MSELoss` for details.
    """
    # charbonnier loss的正则项
    eps = 1e-6

    if not (target.size() == input.size()):
        warnings.warn("Using a target size ({}) that is different to the input size ({}). "
                      "This will likely lead to incorrect results due to broadcasting. "
                      "Please ensure they have the same size.".format(target.size(), input.size()),
                      stacklevel=2)
    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)
    if target.requires_grad:
        ret = (input - target) ** 2
        if reduction != 'none':
            ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    else:
        mask = torch.unsqueeze(mask, 1)
        mask_new = torch.cat([mask, mask], dim=1)
        # vaild_pixel = torch.sum(mask) * 2
        vaild_pixel = torch.sum(mask_new)
        ret_origin = input - target
        ret_vaild = ret_origin * mask_new
        ret_mask = ret_vaild ** 2
        ret_old = ret_origin ** 2
        if reduction != 'none':
            # charbonnier loss的loss
            ret_mask = torch.sum(ret_mask + eps) / vaild_pixel
            # 原始loss
            # ret_mask = torch.sum(ret_mask) / vaild_pixel

            ret_old = torch.mean(ret_old) if reduction == 'mean' else torch.sum(ret_old)

        # expanded_input, expanded_target = torch.broadcast_tensors(input, target)
        # ret = torch._C._nn.mse_loss(expanded_input, expanded_target, _Reduction.get_enum(reduction))
    return ret_mask


class _Loss(nn.Module):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction = reduction
        else:
            self.reduction = reduction

class MSELoss(_Loss):
    __constants__ = ['reduction']

    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target, mask):
        return mse_loss(input, target, mask, reduction=self.reduction)

