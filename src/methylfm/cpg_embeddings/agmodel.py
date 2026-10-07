"""Official AlphaGenome (google-deepmind/alphagenome_research, JAX) loading
and trunk-embedding extraction. Imports JAX: only used in the dedicated
extraction environment, never during MethylFM training.

The embeddings are the *trunk* representations ``embeddings_1bp``
(``[B, S, 1536]``) and ``embeddings_128bp`` (``[B, S//128, 3072]``) returned by
``AlphaGenome.forward_trunk`` -- before any task-specific head and not any
predicted track.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

SOURCE_REPOSITORY = "google-deepmind/alphagenome_research"
HF_REPO = "google/alphagenome-{version}"
D_1BP, D_128BP = 1536, 3072
# AlphaGenome's training / documented maximum context (README: "up to 1 million
# base pairs"; finetune colab SEQUENCE_LENGTH = 2**20).
OFFICIAL_MAX_SEQUENCE_LENGTH = 2**20


def _git(path: Path, *args: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def software_provenance() -> dict:
    import alphagenome_research
    import jax

    ag_root = Path(alphagenome_research.__file__).resolve().parents[2]
    here = Path(__file__).resolve()
    repo = next((p for p in here.parents if (p / ".git").exists()), None)
    dirty = bool(repo and _git(repo, "status", "--porcelain"))
    return {
        "source_model": "AlphaGenome",
        "source_repository": SOURCE_REPOSITORY,
        "model_commit": _git(ag_root, "rev-parse", "HEAD") or "unknown (not a git checkout)",
        "extraction_software_commit": ((_git(repo, "rev-parse", "HEAD") if repo else None) or "unknown")
        + ("+dirty" if dirty else ""),
        "jax_version": jax.__version__,
    }


def resolve_checkpoint(model_cfg: dict) -> tuple[str, dict]:
    """Download/locate the official checkpoint; returns (path, provenance)."""
    src = model_cfg["source"]
    if src == "huggingface":
        import huggingface_hub

        repo = HF_REPO.format(version=model_cfg["version"].replace("_", "-").lower())
        path = huggingface_hub.snapshot_download(repo_id=repo)
        return path, {"model_checkpoint": f"huggingface:{repo}@{Path(path).name}"}
    if src == "kaggle":
        import kagglehub

        handle = f"google/alphagenome/jax/{model_cfg['version'].lower()}"
        path = kagglehub.model_download(handle)
        return path, {"model_checkpoint": f"kaggle:{handle}@{Path(path).name}"}
    if src == "path":
        return str(model_cfg["checkpoint_path"]), {"model_checkpoint": f"path:{model_cfg['checkpoint_path']}"}
    raise ValueError(f"unknown model source {src!r}")


class AlphaGenomeTrunk:
    """Official model + a jitted trunk function. ``compute='bf16'`` is the
    official inference precision; ``'fp32'`` is used only for the precision audit."""

    def __init__(self, cfg: dict):
        import jax
        from alphagenome_research.model import dna_model

        self.jax = jax
        checkpoint, self.checkpoint_info = resolve_checkpoint(cfg["model"])
        self.device = jax.devices()[0]
        self.model = dna_model.create(checkpoint, device=self.device)
        self.organism_index = np.array([dna_model.convert_to_organism_index(_organism(cfg["model"]["organism"]))], np.int32)
        self._fns = {"bf16": jax.jit(self.model._trunk_apply_fn)}

    def _trunk_fn(self, compute: str):
        if compute in self._fns:
            return self._fns[compute]
        import haiku as hk
        import jmp
        from alphagenome_research.model import model as ag

        assert compute == "fp32"
        policy = jmp.get_policy("params=float32,compute=float32,output=float32")
        metadata = self.model._metadata

        @hk.transform_with_state
        def fwd(x, org):
            with hk.mixed_precision.push_policy(ag.AlphaGenome, policy):
                return ag.AlphaGenome(
                    metadata,
                    num_splice_sites=ag.DEFAULT_NUM_SPLICE_SITES,
                    splice_site_threshold=ag.DEFAULT_SPLICE_SITE_THRESHOLD,
                ).forward_trunk(x, org)

        fn = self.jax.jit(lambda p, s, x, o: fwd.apply(p, s, None, x, o)[0])
        self._fns[compute] = fn
        return fn

    def check_geometry(self, window_size: int) -> dict:
        """Assert the official output geometry assumed by `coordinates`:
        1-bp output aligned with the input, 128-bp output with S//128 bins."""
        import jax
        import jax.numpy as jnp

        if window_size > OFFICIAL_MAX_SEQUENCE_LENGTH:
            raise ValueError(f"window_size {window_size} exceeds the official maximum {OFFICIAL_MAX_SEQUENCE_LENGTH}")
        out = jax.eval_shape(
            self._fns["bf16"], self.model._params, self.model._state,
            jax.ShapeDtypeStruct((1, window_size, 4), jnp.float32), jax.ShapeDtypeStruct((1,), jnp.int32),
        )
        s1, s128 = out.embeddings_1bp.shape, out.embeddings_128bp.shape
        if s1 != (1, window_size, D_1BP) or s128 != (1, window_size // 128, D_128BP):
            raise AssertionError(f"unexpected AlphaGenome trunk output geometry: {s1}, {s128}")
        return {"embeddings_1bp_shape": list(s1), "embeddings_128bp_shape": list(s128)}

    def embed(self, onehot: np.ndarray, compute: str = "bf16"):
        """``onehot`` ``[B,S,4]`` float32 -> (e1bp ``[B,S,1536]``, e128 ``[B,S//128,3072]``) device arrays."""
        import jax.numpy as jnp

        org = jnp.asarray(np.repeat(self.organism_index, onehot.shape[0]))
        out = self._trunk_fn(compute)(self.model._params, self.model._state, jnp.asarray(onehot), org)
        return out.embeddings_1bp, out.embeddings_128bp

    @staticmethod
    def encode_sequence(seq: str) -> np.ndarray:
        from alphagenome_research.model import one_hot_encoder

        return one_hot_encoder.DNAOneHotEncoder().encode(seq)


def _organism(name: str):
    from alphagenome_research.model import dna_model

    return {"human": dna_model.Organism.HOMO_SAPIENS, "mouse": dna_model.Organism.MUS_MUSCULUS}[name]


def gather_rows(e1, e128, bp_idx: np.ndarray, bin_idx: np.ndarray, batch_item: int = 0, bucket: int = 512):
    """Select the owned CpG positions from device embeddings of one window and
    return float32 host arrays. Index arrays are padded to a bucket multiple so
    the jitted gather compiles for only a handful of shapes."""
    import jax
    import jax.numpy as jnp

    n = len(bp_idx)
    padded = -(-n // bucket) * bucket
    bp = np.zeros(padded, np.int32)
    bn = np.zeros(padded, np.int32)
    bp[:n], bn[:n] = bp_idx, bin_idx
    out1, out128 = _gather_jit()(e1[batch_item], e128[batch_item], jnp.asarray(bp), jnp.asarray(bn))
    return np.asarray(out1)[:n], np.asarray(out128)[:n]


_GATHER = None


def _gather_jit():
    global _GATHER
    if _GATHER is None:
        import jax
        import jax.numpy as jnp

        @jax.jit
        def g(e1, e128, bp, bn):
            return jnp.take(e1, bp, axis=0).astype(jnp.float32), jnp.take(e128, bn, axis=0).astype(jnp.float32)

        _GATHER = g
    return _GATHER
