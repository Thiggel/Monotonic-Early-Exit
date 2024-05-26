from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


def split_tensors_by_mask(
    tensors: Optional[torch.FloatTensor] = None, 
    skip_mask: Optional[torch.BoolTensor] = None,
    ids_restore: Optional[torch.LongTensor] = None,
    change_dims=False
):
    """
    0 and 1 values in skip_mask denote the index for tensors to keep and skip, respectively.
    """
    if ids_restore is None:
        ids_shuffle = torch.argsort(skip_mask.long(), stable=True)
        ids_restore = torch.argsort(ids_shuffle)
    print("mask shape: " + str(skip_mask.shape))
    print("tensor shape: " + str(tensors.shape))
    if skip_mask.dim() < tensors.dim() and change_dims:
        # Add singleton dimensions to skip_mask until it matches tensors dimensions
        expand_shape = list(skip_mask.shape) + [1] * (tensors.dim() - skip_mask.dim())
        skip_mask = skip_mask.reshape(expand_shape)
        
    # Broadcast skip_mask to match the exact shape of tensors
    if skip_mask.shape != tensors.shape and change_dims:
        skip_mask = skip_mask.expand_as(tensors)

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
