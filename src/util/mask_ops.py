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
    if skip_mask.dim() == 1:
        skip_mask = skip_mask.unsqueeze(-1)  # Add a dimension to make it [batch_size, 1]

    # If ids_restore is provided, use it to shuffle tensors according to some prior state
    if ids_restore is not None:
        tensors = torch.index_select(tensors, 0, ids_restore)
    else:
        # Calculate ids_restore based on the skip_mask to reorder tensors back to their original order later
        ids_shuffle = torch.argsort(skip_mask.long(), dim=0, stable=True)  # Only one dimension to sort
        ids_restore = torch.argsort(ids_shuffle, dim=0)  # Inverse permutation

    # Expanding skip_mask to match the dimensions of tensors
    if tensors.dim() > skip_mask.dim():
        expanded_size = list(skip_mask.shape) + [1] * (tensors.dim() - skip_mask.dim())
        skip_mask = skip_mask.expand(*expanded_size)

    keep_tensors = tensors[~skip_mask]
    skip_tensors = tensors[skip_mask]

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
