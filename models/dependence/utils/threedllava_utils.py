"""Utility functions for point cloud processing and data collation."""

import torch
import numpy as np
from typing import Sequence, Mapping


def ponder_collate_fn(batch, max_point=-1):
    """
    Collate function for point cloud which supports dict and list.
    'coord' is necessary to determine 'offset'.
    
    Args:
        batch: List of samples or dict/tensor data
        max_point: Maximum number of points in a batch. If > 0, drops samples exceeding this limit.
    
    Returns:
        Collated batch data
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")

    # Drop large data if it exceeds max_point
    if max_point > 0:
        accum_num_points = 0
        ret_batches = []
        for batch_id, data in enumerate(batch):
            num_coords = data["coord"].shape[0]
            if accum_num_points + num_coords > max_point:
                continue
            accum_num_points += num_coords
            ret_batches.append(data)
        return ponder_collate_fn(ret_batches)

    if isinstance(batch[0], torch.Tensor):
        return torch.cat(list(batch))
    elif isinstance(batch[0], str):
        # str is also a kind of Sequence, judgment should be before Sequence
        return list(batch)
    elif isinstance(batch[0], Sequence):
        for data in batch:
            data.append(torch.tensor([data[0].shape[0]]))
        batch = [ponder_collate_fn(samples) for samples in zip(*batch)]
        batch[-1] = torch.cumsum(batch[-1], dim=0).int()
        return batch
    elif isinstance(batch[0], Mapping):
        batch_dict = {key: ponder_collate_fn([d[key] for d in batch]) for key in batch[0]}
        for key in batch_dict.keys():
            if "offset" in key:
                batch_dict[key] = torch.cumsum(batch_dict[key], dim=0)
        return batch_dict
    else:
        from torch.utils.data.dataloader import default_collate
        return default_collate(batch)
