"""W&B logging wrapper.

Run-name convention (per project spec): simple deterministic names such as
``smoke-s17`` / ``pretrain-s17`` / ``resume-s17`` — never verbose hyperparameter
dumps. The caller is responsible for choosing `run_name`; this class only passes
it through. Architecture/config details belong in W&B's `config`, not the name.
"""


class WandbLogger:
    def __init__(
        self,
        config: dict,
        enabled: bool = True,
        project: str = "MethylFM",
        entity: str | None = None,
        group: str | None = None,
        job_type: str | None = None,
        tags: list[str] | None = None,
        run_name: str | None = None,
    ):
        self.enabled = enabled
        self._run = None
        if self.enabled:
            import wandb

            self._wandb = wandb
            self._run = wandb.init(
                project=project,
                entity=entity,
                group=group,
                job_type=job_type,
                tags=tags or [],
                name=run_name,
                config=config,
            )

    def log(self, metrics: dict, step: int) -> None:
        if not self.enabled:
            return
        self._wandb.log(metrics, step=step)

    def log_optimization(self, loss, invariance_loss, sigreg_loss, lr, grad_norm, step_time_s, step) -> None:
        self.log(
            {
                "train/loss": loss,
                "train/invariance_loss": invariance_loss,
                "train/sigreg_loss": sigreg_loss,
                "train/lr": lr,
                "train/grad_norm": grad_norm,
                "train/step_time_s": step_time_s,
            },
            step=step,
        )

    def log_data_stats(
        self,
        samples_per_batch,
        cpg_tokens_per_batch,
        cpg_tokens_per_sample_mean,
        reference_cpg_fraction,
        broad_cpg_fraction,
        sparse_cpg_fraction,
        step,
    ) -> None:
        self.log(
            {
                "data/samples_per_batch": samples_per_batch,
                "data/cpg_tokens_per_batch": cpg_tokens_per_batch,
                "data/cpg_tokens_per_sample_mean": cpg_tokens_per_sample_mean,
                "data/reference_cpg_fraction": reference_cpg_fraction,
                "data/broad_cpg_fraction": broad_cpg_fraction,
                "data/sparse_cpg_fraction": sparse_cpg_fraction,
            },
            step=step,
        )

    def log_repr_health(
        self,
        mean,
        std,
        norm,
        effective_rank,
        cosine_same_sample_views,
        cosine_random_samples,
        step,
    ) -> None:
        self.log(
            {
                "repr/mean": mean,
                "repr/std": std,
                "repr/norm": norm,
                "repr/effective_rank": effective_rank,
                "repr/cosine_same_sample_views": cosine_same_sample_views,
                "repr/cosine_random_samples": cosine_random_samples,
            },
            step=step,
        )

    def log_system_stats(self, gpu_allocated, gpu_reserved, samples_per_sec, cpg_tokens_per_sec, step) -> None:
        self.log(
            {
                "system/gpu_allocated_mb": gpu_allocated,
                "system/gpu_reserved_mb": gpu_reserved,
                "system/samples_per_s": samples_per_sec,
                "system/cpg_tokens_per_s": cpg_tokens_per_sec,
            },
            step=step,
        )

    def watch_config(self, resolved_config: dict) -> None:
        if not self.enabled:
            return
        self._wandb.config.update(resolved_config, allow_val_change=True)

    def finish(self) -> None:
        if not self.enabled:
            return
        self._wandb.finish()
