"""Atomic checkpoint save/load with full training-state round-trip (model, projector,
optimizer, scheduler, scaler, RNG state, resolved config)."""

import os
from dataclasses import dataclass

import torch

from methylfm.utils import seed as seed_utils


@dataclass
class CheckpointMeta:
    epoch: int
    global_step: int
    config: dict


def save_checkpoint(
    path: str,
    model,
    projector,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_step: int,
    config: dict,
) -> None:
    payload = {
        "model": model.state_dict(),
        "projector": projector.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "rng_state": seed_utils.get_rng_state(),
        "config": config,
    }
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(
    path: str,
    model,
    projector,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location="cpu",
) -> CheckpointMeta:
    payload = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(payload["model"])
    projector.load_state_dict(payload["projector"])

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])

    seed_utils.set_rng_state(payload["rng_state"])

    return CheckpointMeta(
        epoch=payload["epoch"],
        global_step=payload["global_step"],
        config=payload["config"],
    )


def save_latest_and_maybe_best(
    ckpt_dir: str,
    model,
    projector,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_step,
    config,
    metric: float | None = None,
    best_so_far: float | None = None,
) -> float | None:
    os.makedirs(ckpt_dir, exist_ok=True)

    latest_path = os.path.join(ckpt_dir, "latest.pt")
    save_checkpoint(latest_path, model, projector, optimizer, scheduler, scaler, epoch, global_step, config)

    if metric is None:
        return best_so_far

    if best_so_far is None or metric < best_so_far:
        best_path = os.path.join(ckpt_dir, "best.pt")
        save_checkpoint(best_path, model, projector, optimizer, scheduler, scaler, epoch, global_step, config)
        return metric

    return best_so_far
