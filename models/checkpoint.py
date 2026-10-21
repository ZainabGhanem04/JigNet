"""Checkpoint utilities."""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Union

import torch
from torch import nn

from .jignet import JigNet


def save_checkpoint(path: Union[str, Path], model, optimizer, scheduler, epoch: int, history: dict, cfg=None) -> None:
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "history": history,
        "num_classes": model.mlp[-1].out_features,
        "config": cfg,
    }, path)


def get_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[str]:
    ckpts = sorted(glob.glob(os.path.join(str(checkpoint_dir), "checkpoint_epoch_*.pth")))
    return ckpts[-1] if ckpts else None


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location=device)


def load_jignet(path: Union[str, Path], device: Union[str, torch.device] = "cpu", num_classes: int = 5) -> JigNet:
    """Load JigNet in eval mode from a bare state_dict, a training checkpoint, a fully pickled model,
    or a directory of `checkpoint_epoch_*.pth` files (latest is used)."""
    path = str(path)
    if os.path.isdir(path):
        latest = get_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No checkpoint_epoch_*.pth in {path}")
        path = latest
    obj = _torch_load(path, device)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        state = obj["model_state_dict"] if isinstance(obj, dict) and "model_state_dict" in obj else obj
        model = JigNet(num_classes=num_classes)
        model.load_state_dict(state)
    return model.to(device).eval()
