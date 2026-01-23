# utils/cuda_utils.py

import torch

import torch.nn as nn
import gc


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
