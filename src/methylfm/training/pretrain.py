"""The ONE canonical LeJEPA pretraining entry point, built on top of
Hugging Face Accelerate so the same code path runs single-GPU, multi-GPU
DDP, and CPU, with BF16/FP16 and gradient accumulation handled by
`Accelerator` rather than hand-rolled distributed code.

Usable directly:
    python -m methylfm.training.pretrain --config configs/pretrain.yaml

or under Accelerate for single/multi-GPU:
    accelerate launch -m methylfm.training.pretrain --config configs/pretrain.yaml

Pass --smoke to run a short synthetic end-to-end smoke pretraining instead
of loading the real CpGCorpus -- same code path, synthetic data only.

Distributed LeJEPA correctness (see `lejepa_loss`'s `gather_fn` argument):
SIGReg's Epps-Pulley statistic is a nonlinear function of the empirical
batch distribution, so it must be computed over the GLOBAL batch, not
independently per process. Each view is therefore run through the
Accelerate-wrapped model separately (so DDP's gradient hooks fire for every
contributing forward call) and the resulting per-process `z` is gathered
with a differentiable all-gather before SIGReg is computed; `invariance_loss`
is left per-process-local since it has no cross-sample coupling (DDP's
ordinary gradient averaging already reproduces its global-batch gradient).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.distributed.nn.functional as dist_nn_functional
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset

from methylfm.configuration_methylfm import MethylFMConfig
from methylfm.data.collate import collate_methylation_batch
from methylfm.data.cpgcorpus import CpGCorpusConfig, CpGCorpusDataset, CpGCorpusNotFoundError
from methylfm.data.sample import MethylationSample, validate_sample
from methylfm.data.views import ViewBuilder, ViewConfig
from methylfm.modeling_methylfm import MethylFMForPreTraining
from methylfm.pretraining.lejepa import lejepa_loss
from methylfm.training.checkpoint import load_checkpoint, save_latest_and_maybe_best
from methylfm.training.logging import WandbLogger
from methylfm.utils.config import load_config
from methylfm.utils.seed import seed_everything


def _raw_collate(batch: list[MethylationSample]) -> list[MethylationSample]:
    return batch


def build_dataloader(dataset: Dataset, batch_size: int, num_workers: int = 0, seed: int = 0) -> DataLoader:
    """A single, explicitly-seeded shuffling generator (not PyTorch's
    unpredictable default) so resumed runs reproduce the same epoch
    ordering -- see `_fast_forward_dataloader`."""
    shuffle_generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=_raw_collate, drop_last=True, generator=shuffle_generator,
    )


def _fast_forward_dataloader(dataloader: DataLoader, num_batches_to_skip: int):
    """Advances `dataloader`'s shuffling generator through exactly
    `num_batches_to_skip` already-consumed batches (possibly spanning
    multiple full epochs) so a resumed run reproduces the same future batch
    order an uninterrupted run would have seen (requires num_workers=0 for
    exact single-process determinism)."""
    if num_batches_to_skip <= 0:
        return iter(dataloader)
    batches_per_epoch = len(dataloader)
    full_epochs, remainder = divmod(num_batches_to_skip, batches_per_epoch)
    for _ in range(full_epochs):
        for _ in dataloader:
            pass
    data_iter = iter(dataloader)
    for _ in range(remainder):
        next(data_iter)
    return data_iter


def _step_generator(seed: int, data_step: int) -> torch.Generator:
    """Deterministic, stateless-across-steps generator for view construction:
    a pure function of (seed, data_step), so view sampling is exactly
    resumable regardless of how many steps ran before it in this process."""
    return torch.Generator().manual_seed(seed * 1_000_003 + data_step)


def build_view_batches(
    samples: list[MethylationSample],
    view_builder: ViewBuilder,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[list[dict], list[int]]:
    """Builds reference_view + broad_view + sparse_views for each sample and
    collates each view "slot" into its own batch dict (cpg_ids/beta_values/
    attention_mask/num_cpgs), ready to pass to `MethylFMForPreTraining`.
    Indices [0, 1] are reference_view/broad_view (the 2 global views)."""
    reference_samples, broad_samples = [], []
    sparse_samples: list[list[MethylationSample]] = [[] for _ in range(view_builder.config.num_local_views)]

    for raw_sample in samples:
        sample = validate_sample(raw_sample)
        views = view_builder.build_views(sample, generator)
        reference_samples.append(views["reference_view"])
        broad_samples.append(views["broad_view"])
        for k, sparse_view in enumerate(views["sparse_views"]):
            sparse_samples[k].append(sparse_view)

    collated = [collate_methylation_batch(reference_samples), collate_methylation_batch(broad_samples)]
    for k in range(view_builder.config.num_local_views):
        collated.append(collate_methylation_batch(sparse_samples[k]))

    view_batches = [
        {
            "cpg_ids": cb.cpg_ids.to(device),
            "beta_values": cb.beta_values.to(device),
            "attention_mask": cb.mask.to(device),
            "num_cpgs": cb.num_cpgs.to(device),
        }
        for cb in collated
    ]
    return view_batches, [0, 1]


def compute_repr_health(h_views: list[torch.Tensor]) -> dict:
    """Cheap representation-health diagnostics -- gate behind a logging interval."""
    h_ref = h_views[0].detach()
    h_all = torch.cat([h.detach() for h in h_views], dim=0)

    mean = h_all.mean().item()
    std = h_all.std().item()
    norm = h_all.norm(dim=-1).mean().item()

    centered = h_ref - h_ref.mean(dim=0, keepdim=True)
    svdvals = torch.linalg.svdvals(centered)
    p = svdvals / svdvals.sum().clamp_min(1e-12)
    p = p.clamp_min(1e-12)
    effective_rank = torch.exp(-(p * p.log()).sum()).item()

    same_sample_cos = [
        torch.nn.functional.cosine_similarity(h_views[0].detach(), h_views[v].detach(), dim=-1).mean().item()
        for v in range(1, len(h_views))
    ]
    cosine_same_sample_views = sum(same_sample_cos) / max(1, len(same_sample_cos))

    if h_ref.shape[0] > 1:
        shuffled = h_ref[torch.randperm(h_ref.shape[0], device=h_ref.device)]
        cosine_random_samples = torch.nn.functional.cosine_similarity(h_ref, shuffled, dim=-1).mean().item()
    else:
        cosine_random_samples = float("nan")

    return {
        "mean": mean, "std": std, "norm": norm, "effective_rank": effective_rank,
        "cosine_same_sample_views": cosine_same_sample_views, "cosine_random_samples": cosine_random_samples,
    }


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        model.parameters(), lr=config.get("lr", 3.0e-4),
        weight_decay=config.get("weight_decay", 0.05), betas=tuple(config.get("betas", [0.9, 0.95])),
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = config.get("warmup_steps", 2000)
    max_steps = config.get("max_steps", 200000)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_model_config(model_cfg: dict) -> MethylFMConfig:
    cpg_cfg = model_cfg.get("cpg_encoder", {})
    proj_cfg = model_cfg.get("projector", {})
    model_dim = model_cfg.get("model_dim", 512)
    locus_dim = model_cfg.get("locus_dim", model_dim)
    return MethylFMConfig(
        hidden_size=model_dim,
        num_latents=model_cfg.get("num_latents", 128),
        num_hidden_layers=model_cfg.get("num_layers", 16),
        num_attention_heads=model_cfg.get("num_heads", 8),
        intermediate_size=model_dim * model_cfg.get("mlp_ratio", 4),
        mhc_expansion_rate=model_cfg.get("mhc_expansion_rate", 4),
        mhc_sinkhorn_iters=model_cfg.get("mhc_sinkhorn_iters", 20),
        attention_dropout=model_cfg.get("dropout", 0.0),
        hidden_dropout=model_cfg.get("dropout", 0.0),
        locus_dim=locus_dim,
        cross_attention_chunk_size=model_cfg.get("cross_attention", {}).get("chunk_size"),
        cpg_encoder_type=cpg_cfg.get("type", "mock"),
        cpg_encoder_vocab_size=cpg_cfg.get("vocab_size", 2**20),
        cpg_embedding_1bp_dim=cpg_cfg.get("embedding_1bp_dim", 1536),
        cpg_embedding_128bp_dim=cpg_cfg.get("embedding_128bp_dim", 3072),
        cpg_adapter_hidden_size=cpg_cfg.get("adapter_hidden_size", locus_dim),
        cpg_adapter_bottleneck_size=cpg_cfg.get("adapter_bottleneck_size", 64),
        cpg_embedding_asset_id=cpg_cfg.get("asset_id"),
        cpg_lookup_mode=cpg_cfg.get("lookup_mode", "gather"),
        cpg_lookup_chunk_size=cpg_cfg.get("lookup_chunk_size", 65536),
        projector_hidden_size=proj_cfg.get("hidden_dim", 1024),
        projector_output_size=proj_cfg.get("output_dim", 512),
    )


def _differentiable_all_gather(tensor: torch.Tensor, accelerator: Accelerator) -> torch.Tensor:
    if accelerator.num_processes <= 1:
        return tensor
    gathered = dist_nn_functional.all_gather(tensor)
    return torch.cat(gathered, dim=0)


def _cpgcorpus_config_from_dict(data_cfg: dict) -> CpGCorpusConfig:
    import os

    fields = {"root", "use_qc_processed", "min_cpgs_per_sample", "split_seed", "val_fraction"}
    root = data_cfg.get("root") or os.environ.get("METHYLFM_DATA_ROOT")
    kwargs = {k: v for k, v in data_cfg.items() if k in fields and k != "root"}
    return CpGCorpusConfig(root=root, **kwargs)


@dataclass
class TrainResult:
    global_step: int
    last_loss: float


def run(
    config: dict,
    dataset: Dataset,
    run_name: str,
    resume_from: str | None = None,
    max_steps_override: int | None = None,
    wandb_enabled: bool = True,
    cpg_embedding_store=None,
) -> TrainResult:
    data_cfg = config["data"]
    train_cfg = config["training"]
    opt_cfg = config["optimizer"]
    sigreg_cfg = config["sigreg"]
    wandb_cfg = config.get("wandb", {})
    seed = config.get("seed", 1337)

    precision = train_cfg.get("precision", "bf16")
    accelerator = Accelerator(
        mixed_precision=precision if precision in ("bf16", "fp16") else "no",
        gradient_accumulation_steps=train_cfg.get("grad_accum_steps", 1),
    )
    seed_everything(seed)

    model = MethylFMForPreTraining(build_model_config(config["model"]), cpg_embedding_store=cpg_embedding_store)

    views_cfg = data_cfg.get("views", {})
    view_builder = ViewBuilder(
        ViewConfig(
            global_keep_fraction=views_cfg.get("global_keep_fraction", 0.8),
            num_local_views=views_cfg.get("num_local_views", 4),
            local_keep_fraction=views_cfg.get("local_keep_fraction", 0.25),
        )
    )

    optimizer = build_optimizer(model, opt_cfg)
    scheduler = build_scheduler(optimizer, opt_cfg)
    dataloader = build_dataloader(
        dataset, batch_size=data_cfg.get("batch_size", 32), num_workers=data_cfg.get("num_workers", 0), seed=seed
    )

    # The scheduler is deliberately NOT passed through `prepare`: Accelerate's
    # AcceleratedScheduler advances it `num_processes` times per optimizer step,
    # which would shrink warmup/decay by a factor of world size. It is stepped
    # manually, once per real optimizer step, below.
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    global_step = 0
    best_so_far: float | None = None
    if resume_from is not None:
        unwrapped = accelerator.unwrap_model(model)
        meta = load_checkpoint(
            resume_from, unwrapped.methylfm, unwrapped.projector, optimizer, scheduler,
            scaler=None, map_location=accelerator.device,
        )
        global_step = meta.global_step
    accelerator.wait_for_everyone()

    max_steps = max_steps_override if max_steps_override is not None else opt_cfg.get("max_steps", 200000)
    grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
    grad_clip = opt_cfg.get("grad_clip", 1.0)
    log_interval = train_cfg.get("log_interval", 50)
    repr_log_interval = train_cfg.get("repr_log_interval", 500)
    ckpt_interval = train_cfg.get("ckpt_interval", 1000)
    ckpt_dir = train_cfg.get("ckpt_dir", "outputs/checkpoints")

    data_iter = _fast_forward_dataloader(dataloader, global_step * grad_accum_steps)

    wandb_logger = WandbLogger(
        config=config, enabled=wandb_enabled and accelerator.is_main_process,
        project=wandb_cfg.get("project", "MethylFM"), entity=wandb_cfg.get("entity"),
        group=wandb_cfg.get("group", "pretrain"), job_type=wandb_cfg.get("job_type", "pretrain"),
        tags=wandb_cfg.get("tags"), run_name=run_name,
    )

    def gather_fn(z: torch.Tensor) -> torch.Tensor:
        return _differentiable_all_gather(z, accelerator)

    last_loss = float("nan")
    micro_step = global_step * grad_accum_steps
    accum_loss_sum = accum_inv_sum = accum_sigreg_sum = 0.0
    accum_count = 0
    step_start = time.time()
    try:
        while global_step < max_steps:
            try:
                samples = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                samples = next(data_iter)

            step_generator = _step_generator(seed, micro_step)
            view_batches, global_view_indices = build_view_batches(
                samples, view_builder, step_generator, accelerator.device
            )

            # `accelerator.accumulate(model)` is entered once per MICRO-BATCH
            # (Accelerate's own documented idiom), not once per optimizer
            # step -- it internally gates DDP gradient sync / optimizer.step()
            # / scheduler.step() so they only actually fire every
            # `gradient_accumulation_steps` entries (exposed via
            # `accelerator.sync_gradients`).
            with accelerator.accumulate(model):
                # Each view goes through the Accelerate/DDP-wrapped
                # `model.__call__` separately (NOT an unwrapped method call)
                # so DDP's gradient-sync hooks fire for every contributing
                # forward pass.
                h_views, z_views = [], []
                for vb in view_batches:
                    out = model(vb["cpg_ids"], vb["beta_values"], vb["attention_mask"])
                    h_views.append(out.pooler_output)
                    z_views.append(out.embeddings)

                losses = lejepa_loss(
                    z_views, global_view_indices=global_view_indices,
                    sigreg_lambda=sigreg_cfg.get("lambda", 0.02),
                    num_slices=sigreg_cfg.get("num_slices", 1024),
                    num_points=sigreg_cfg.get("num_points", 17),
                    gather_fn=gather_fn,
                )
                accelerator.backward(losses["total"])
                accum_loss_sum += losses["total"].item()
                accum_inv_sum += losses["invariance"].item()
                accum_sigreg_sum += losses["sigreg"].item()
                accum_count += 1

                grad_norm = torch.tensor(0.0)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                optimizer.zero_grad()

            micro_step += 1
            if not accelerator.sync_gradients:
                continue  # still accumulating -- not a real step yet

            global_step += 1
            last_loss = accum_loss_sum / max(1, accum_count)
            avg_inv = accum_inv_sum / max(1, accum_count)
            avg_sigreg = accum_sigreg_sum / max(1, accum_count)
            step_time_s = time.time() - step_start
            step_start = time.time()
            accum_loss_sum = accum_inv_sum = accum_sigreg_sum = 0.0
            accum_count = 0

            if global_step % log_interval == 0:
                cpg_tokens_per_batch = sum(int(vb["attention_mask"].sum().item()) for vb in view_batches)
                samples_per_batch = view_batches[0]["attention_mask"].shape[0]
                ref_cpgs = view_batches[0]["num_cpgs"].float()
                broad_frac = (view_batches[1]["num_cpgs"].float() / ref_cpgs.clamp_min(1)).mean().item()
                sparse_frac = torch.stack(
                    [(vb["num_cpgs"].float() / ref_cpgs.clamp_min(1)).mean() for vb in view_batches[2:]]
                ).mean().item()
                wandb_logger.log_optimization(
                    loss=last_loss, invariance_loss=avg_inv,
                    sigreg_loss=avg_sigreg, lr=scheduler.get_last_lr()[0],
                    grad_norm=float(grad_norm), step_time_s=step_time_s, step=global_step,
                )
                wandb_logger.log_data_stats(
                    samples_per_batch=samples_per_batch, cpg_tokens_per_batch=cpg_tokens_per_batch,
                    cpg_tokens_per_sample_mean=cpg_tokens_per_batch / max(1, samples_per_batch * len(view_batches)),
                    reference_cpg_fraction=1.0, broad_cpg_fraction=broad_frac, sparse_cpg_fraction=sparse_frac,
                    step=global_step,
                )

            if global_step % repr_log_interval == 0:
                wandb_logger.log_repr_health(**compute_repr_health(h_views), step=global_step)

            if global_step % ckpt_interval == 0 and accelerator.is_main_process:
                unwrapped = accelerator.unwrap_model(model)
                best_so_far = save_latest_and_maybe_best(
                    ckpt_dir, unwrapped.methylfm, unwrapped.projector, optimizer, scheduler, scaler=None,
                    epoch=0, global_step=global_step, config=config, metric=last_loss, best_so_far=best_so_far,
                )

    except KeyboardInterrupt:
        if accelerator.is_main_process:
            unwrapped = accelerator.unwrap_model(model)
            save_latest_and_maybe_best(
                ckpt_dir, unwrapped.methylfm, unwrapped.projector, optimizer, scheduler, scaler=None,
                epoch=0, global_step=global_step, config=config, metric=None, best_so_far=best_so_far,
            )
        raise

    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        save_latest_and_maybe_best(
            ckpt_dir, unwrapped.methylfm, unwrapped.projector, optimizer, scheduler, scaler=None,
            epoch=0, global_step=global_step, config=config, metric=last_loss, best_so_far=best_so_far,
        )

    wandb_logger.finish()
    return TrainResult(global_step=global_step, last_loss=last_loss)


class SyntheticMethylationDataset(Dataset):
    """Synthetic methylation samples for --smoke ONLY. Not a claim about real
    data -- random CpG ids and random beta values, purely to exercise the
    pipeline end-to-end with finite shapes and finite losses."""

    def __init__(self, num_samples: int, vocab_size: int, min_cpgs: int = 200, max_cpgs: int = 800, seed: int = 0):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.min_cpgs = min_cpgs
        self.max_cpgs = max_cpgs
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> MethylationSample:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
        n = int(torch.randint(self.min_cpgs, self.max_cpgs, (1,), generator=gen).item())
        cpg_ids = torch.randint(0, self.vocab_size, (n,), dtype=torch.int64, generator=gen)
        beta_values = torch.rand(n, dtype=torch.float32, generator=gen)
        return MethylationSample(
            sample_id=f"synthetic-{idx}", cpg_ids=cpg_ids, beta_values=beta_values,
            dataset_id="synthetic", platform="synthetic",
        )


def _smoke_config(seed: int, max_steps: int) -> tuple[dict, int]:
    vocab_size = 4096
    return {
        "seed": seed,
        "model": {
            "model_dim": 64, "num_latents": 16, "num_layers": 2, "num_heads": 4,
            "mlp_ratio": 2, "mhc_expansion_rate": 2, "mhc_sinkhorn_iters": 5,
            "locus_dim": 64, "dropout": 0.0,
            "cpg_encoder": {"type": "mock", "vocab_size": vocab_size},
            "projector": {"hidden_dim": 128, "output_dim": 64},
            "cross_attention": {"chunk_size": None},
        },
        "data": {
            "batch_size": 4, "num_workers": 0,
            "views": {"global_keep_fraction": 0.8, "num_local_views": 2, "local_keep_fraction": 0.25},
        },
        "sigreg": {"lambda": 0.02, "num_slices": 64, "num_points": 17},
        "optimizer": {
            "lr": 3.0e-4, "weight_decay": 0.05, "betas": [0.9, 0.95],
            "warmup_steps": 2, "max_steps": max_steps, "grad_clip": 1.0,
        },
        "training": {
            "precision": "no", "grad_accum_steps": 1, "log_interval": 1,
            "repr_log_interval": max(1, max_steps // 2), "ckpt_interval": max(1, max_steps // 2),
            "ckpt_dir": "outputs/smoke_checkpoints",
        },
        "wandb": {"project": "MethylFM", "group": "pretrain", "job_type": "smoke", "tags": ["cpgcorpus", "lejepa", "mhc", "smoke"]},
    }, vocab_size


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a short synthetic end-to-end smoke pretraining")
    parser.add_argument("--steps", type=int, default=10, help="--smoke only: number of steps")
    parser.add_argument("--seed", type=int, default=1337, help="--smoke only")
    parser.add_argument("--wandb", action="store_true", help="--smoke only: enable W&B (default: disabled)")
    args = parser.parse_args(argv)

    if args.smoke:
        config, vocab_size = _smoke_config(seed=args.seed, max_steps=args.steps)
        dataset = SyntheticMethylationDataset(num_samples=32, vocab_size=vocab_size, seed=config["seed"])
        run_name = args.run_name or f"smoke-s{config['seed']}"
        result = run(config, dataset, run_name=run_name, wandb_enabled=args.wandb)
        print(f"[pretrain --smoke] completed {result.global_step} steps, final loss={result.last_loss:.4f}")
        return

    if not args.config:
        raise SystemExit("--config is required unless --smoke is passed")
    config = load_config(args.config)
    if args.resume:
        config["training"]["resume_from"] = args.resume

    corpus_config = _cpgcorpus_config_from_dict(config["data"])
    # The AlphaGenome encoder's CpG ids are row indices of an external embedding
    # store; the dataset and the model must use the very same store.
    cpg_store = None
    model_config = build_model_config(config["model"])
    if model_config.cpg_encoder_type == "alphagenome":
        from methylfm.model.cpg_encoder import resolve_cpg_embedding_store

        cpg_store = resolve_cpg_embedding_store(model_config, config["model"].get("cpg_encoder", {}).get("store_path"))
    try:
        dataset = CpGCorpusDataset(corpus_config, split="train", cpg_store=cpg_store)
    except CpGCorpusNotFoundError as e:
        raise SystemExit(str(e))

    seed = config.get("seed", 1337)
    job_type = "resume" if config["training"].get("resume_from") else "pretrain"
    run_name = args.run_name or f"{job_type}-s{seed}"
    result = run(
        config, dataset, run_name=run_name,
        resume_from=config["training"].get("resume_from"),
        wandb_enabled=config.get("wandb", {}).get("enabled", True),
        cpg_embedding_store=cpg_store,
    )
    print(f"[pretrain] completed {result.global_step} steps, final loss={result.last_loss:.4f}")


if __name__ == "__main__":
    main()
