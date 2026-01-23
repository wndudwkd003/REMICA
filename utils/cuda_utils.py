# utils/cuda_utils.py

import gc

import torch
import torch.nn as nn


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_gpu(model: nn.Module | None = None):
    try:
        if model is not None:
            model.to("cpu")
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def split_workers(total_workers, gpu_ids):
    """
    예: total_workers=10, gpu_ids=[0,1,2,3] -> [(0,3),(1,3),(2,2),(3,2)]
    """
    g = len(gpu_ids)
    base, rem = divmod(total_workers, g)
    out = []
    for i, gid in enumerate(gpu_ids):
        n = base + (1 if i < rem else 0)
        out.append((gid, n))
    return out
