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
    if tensors.dim() == skip_mask.dim() + 1:
        # Expand mask for features dimension if present
        skip_mask = skip_mask.unsqueeze(-1)
    if tensors.dim() > 2 and skip_mask.dim() < tensors.dim():
        # Expand mask to match all dimensions of the tensors except for the feature dimension
        skip_mask = skip_mask.expand(*tensors.shape[:-1], tensors.shape[-1])

    if ids_restore is not None:
        # Sort the mask and compute indices for restoration if provided
        ids_shuffle = torch.argsort(skip_mask.long(), stable=True)
        ids_restore = torch.argsort(ids_shuffle)

    # Apply the mask to split the tensors
    keep_tensors = tensors[~skip_mask]
    skip_tensors = tensors[skip_mask]

    # It's important to reshape the tensors if they are not the same shape
    if keep_tensors.numel() == 0:
        keep_tensors = keep_tensors.view(*tensors.shape[:-1], 0)
    if skip_tensors.numel() == 0:
        skip_tensors = skip_tensors.view(*tensors.shape[:-1], 0)

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
