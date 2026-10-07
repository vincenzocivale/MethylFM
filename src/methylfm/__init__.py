"""MethylFM: a Hugging Face-native foundation model for DNA methylation arrays.

Public API::

    from methylfm import (
        MethylFMConfig,
        MethylFMModel,
        MethylFMForPreTraining,
        MethylFMProcessor,
    )

Internal modules (`model/`, `pretraining/`, `data/`, `training/`, `utils/`)
are not part of the public API and may change without notice.
"""
__version__ = "0.1.0"

try:
    from methylfm.configuration_methylfm import MethylFMConfig
    from methylfm.modeling_methylfm import MethylFMForPreTraining, MethylFMModel, MethylFMPreTrainedModel
    from methylfm.outputs import MethylFMForPreTrainingOutput, MethylFMOutput
    from methylfm.processing_methylfm import MethylFMProcessor

    _HAS_TORCH_STACK = True
except ModuleNotFoundError as _e:  # pragma: no cover
    # The offline AlphaGenome extraction environment (JAX) deliberately has no
    # torch/transformers; only `methylfm.cpg_embeddings` is used there. Any
    # other missing module is a real error.
    if _e.name not in {"torch", "transformers", "accelerate", "safetensors"}:
        raise
    _HAS_TORCH_STACK = False

__all__ = [
    "MethylFMConfig",
    "MethylFMPreTrainedModel",
    "MethylFMModel",
    "MethylFMForPreTraining",
    "MethylFMOutput",
    "MethylFMForPreTrainingOutput",
    "MethylFMProcessor",
]


def _register_auto_classes() -> None:
    try:
        from transformers import AutoConfig, AutoModel

        AutoConfig.register("methylfm", MethylFMConfig)
        AutoModel.register(MethylFMConfig, MethylFMModel)
    except Exception:
        # AutoClass registration is a convenience for Hub interop, not
        # required for `from methylfm import MethylFMModel` to work -- never
        # let a registration failure (e.g. re-registration in a long-lived
        # process, or an older transformers version) break the import.
        pass


if _HAS_TORCH_STACK:
    _register_auto_classes()
