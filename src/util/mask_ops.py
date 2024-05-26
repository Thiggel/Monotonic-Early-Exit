from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def split_tensors_by_mask(
    tensors: Optional[torch.FloatTensor] = None, 
    skip_mask: Optional[torch.BoolTensor] = None,
    ids_restore: Optional[torch.LongTensor] = None,
):
    """
    0 and 1 values in skip_mask denote the index for tensors to keep and skip, respectively.
    """
    if tensors.dim() > skip_mask.dim():
        expanded_size = list(tensors.shape)
        expanded_size[-1] = 1  # We do not expand the last dimension
        skip_mask = skip_mask.view(*([1] * (tensors.dim() - skip_mask.dim())), *skip_mask.shape)
        skip_mask = skip_mask.expand(*expanded_size)

    if ids_restore is None:
        ids_shuffle = torch.argsort(skip_mask.long(), dim=1, stable=True)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Apply mask to split the tensors
    keep_tensors = tensors.masked_select(~skip_mask).view(-1, tensors.size(-1))  # Reshape to keep the feature dimension intact
    skip_tensors = tensors.masked_select(skip_mask).view(-1, tensors.size(-1))

    return keep_tensors, skip_tensors, ids_restore


def restore_tensors_by_mask(
    keep_tensors: Optional[torch.FloatTensor] = None, 
    skip_tensors: Optional[torch.FloatTensor] = None, 
    ids_restore: Optional[torch.IntTensor] = None,
):  
    # when using this function with skip_mask for early-exit
    skip_tensors = skip_tensors.to(keep_tensors.device)
    if not len(keep_tensors.shape):
        keep_tensors = keep_tensors.reshape(-1,)
    if not len(skip_tensors.shape):
        skip_tensors = skip_tensors.reshape(-1,)
    tensors_ = torch.cat([keep_tensors, skip_tensors], dim=0)
    t_shape = tensors_.shape
    ids_restore = ids_restore.to(torch.int64).to(tensors_.device)


    if len(t_shape) == 1:
        tensors = torch.gather(tensors_, 0, index=ids_restore)
    elif len(t_shape) == 2:
        tensors = torch.gather(tensors_, 0, index=ids_restore.reshape(-1, 1).repeat(1, t_shape[-1]))
    elif len(t_shape) == 3:
        tensors = torch.gather(tensors_, 0, index=ids_restore.reshape(-1, 1, 1).repeat(1, t_shape[-2], t_shape[-1]))
    elif len(t_shape) == 4:
        tensors = torch.gather(tensors_, 0, index=ids_restore.reshape(-1, 1, 1, 1).repeat(1, t_shape[-3], t_shape[-2], t_shape[-1]))

    return tensors
