"""Accelerate-path training smoke + checkpoint/resume equivalence tests for
`methylfm.training.pretrain` -- the one canonical LeJEPA training path."""
from __future__ import annotations

import copy

import torch

from methylfm.training.pretrain import SyntheticMethylationDataset, run

MODEL_CFG = {
    "model_dim": 32, "num_latents": 8, "num_layers": 2, "num_heads": 2,
    "mlp_ratio": 2, "mhc_expansion_rate": 2, "mhc_sinkhorn_iters": 3,
    "locus_dim": 32, "dropout": 0.0,
    "cpg_encoder": {"type": "mock", "vocab_size": 512},
    "projector": {"hidden_dim": 32, "output_dim": 16},
    "cross_attention": {"chunk_size": None},
}

SCHEDULE_TOTAL_STEPS = 6  # fixed cosine-schedule horizon shared by every phase below


def _base_config(ckpt_dir: str) -> dict:
    """`optimizer.max_steps` is the scheduler's cosine-decay horizon for the
    whole run and must stay identical across every phase of a train/resume
    comparison, regardless of how many steps a given `run()` call actually
    executes (controlled via `max_steps_override`)."""
    return {
        "seed": 7,
        "model": copy.deepcopy(MODEL_CFG),
        "data": {
            "batch_size": 4, "num_workers": 0,
            "views": {"global_keep_fraction": 0.8, "num_local_views": 2, "local_keep_fraction": 0.25},
        },
        "sigreg": {"lambda": 0.02, "num_slices": 32, "num_points": 9},
        "optimizer": {
            "lr": 1e-3, "weight_decay": 0.0, "betas": [0.9, 0.95],
            "warmup_steps": 1, "max_steps": SCHEDULE_TOTAL_STEPS, "grad_clip": 1.0,
        },
        "training": {
            "precision": "no", "grad_accum_steps": 1,
            "log_interval": 1000, "repr_log_interval": 1000,
            "ckpt_interval": 1000, "ckpt_dir": ckpt_dir,
        },
    }


def _dataset() -> SyntheticMethylationDataset:
    return SyntheticMethylationDataset(num_samples=64, vocab_size=512, min_cpgs=20, max_cpgs=60, seed=0)


def test_accelerate_smoke_training_runs(tmp_path):
    config = _base_config(ckpt_dir=str(tmp_path / "ckpts"))
    result = run(config, _dataset(), run_name="test-smoke", wandb_enabled=False, max_steps_override=4)
    assert result.global_step == 4
    assert torch.isfinite(torch.tensor(result.last_loss))


def test_checkpoint_save_reload_continue(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")

    torch.manual_seed(0)
    config_a = _base_config(ckpt_dir=ckpt_dir + "_a")
    result_a = run(config_a, _dataset(), run_name="a", wandb_enabled=False, max_steps_override=6)
    assert result_a.global_step == 6
    assert torch.isfinite(torch.tensor(result_a.last_loss))

    torch.manual_seed(0)
    config_b1 = _base_config(ckpt_dir=ckpt_dir)
    result_b1 = run(config_b1, _dataset(), run_name="b1", wandb_enabled=False, max_steps_override=3)
    assert result_b1.global_step == 3

    latest_ckpt = f"{ckpt_dir}/latest.pt"
    torch.manual_seed(12345)  # deliberately different -- resume must not depend on process RNG state
    config_b2 = _base_config(ckpt_dir=ckpt_dir)
    result_b2 = run(
        config_b2, _dataset(), run_name="b2", wandb_enabled=False,
        resume_from=latest_ckpt, max_steps_override=6,
    )
    assert result_b2.global_step == 6
    assert abs(result_a.last_loss - result_b2.last_loss) < 1e-4


def test_checkpoint_resume_restores_step_count(tmp_path):
    ckpt_dir = str(tmp_path / "ckpts")
    config = _base_config(ckpt_dir=ckpt_dir)
    result = run(config, _dataset(), run_name="r1", wandb_enabled=False, max_steps_override=4)
    assert result.global_step == 4

    latest_ckpt = f"{ckpt_dir}/latest.pt"
    config_resume = _base_config(ckpt_dir=ckpt_dir)
    result_resume = run(
        config_resume, _dataset(), run_name="r2", wandb_enabled=False,
        resume_from=latest_ckpt, max_steps_override=4,
    )
    # already at max_steps -- resuming should immediately report the same step.
    assert result_resume.global_step == 4
