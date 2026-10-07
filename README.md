# MethylFM

A foundation model for DNA methylation arrays, built Hugging Face-native.

```python
from methylfm import MethylFMModel, MethylFMProcessor

processor = MethylFMProcessor.from_pretrained("path/to/methylfm-base")
model = MethylFMModel.from_pretrained("path/to/methylfm-base")

inputs = processor(
    cpg_ids=["cg00000029", "cg00000108"],
    beta_values=[0.82, 0.14],
    return_tensors="pt",
)

outputs = model(**inputs)
embedding = outputs.pooler_output  # [batch_size, hidden_size]
```

> MethylFM accepts arbitrary sets of observed CpGs and returns a
> fixed-dimensional representation of the sample. It does not require a
> fixed CpG panel, a specific array platform, or a dense `samples x
> universal_CpGs` matrix -- just whatever CpGs were actually observed.

## Goal

Learn a platform-robust foundation representation of human DNA methylation
arrays from arbitrary sets of observed CpGs.

## Current model

```text
CpG locus representation
    -> FT-style methylation tokenization
    -> set-to-latent cross-attention
    -> mHC Transformer
    -> attention pooling
```

Pretrained using **LeJEPA** (multi-view invariance + SIGReg).

| Class | Role |
|---|---|
| `MethylFMConfig` | Architecture hyperparameters (`hidden_size`, `num_latents`, `num_hidden_layers`, `num_attention_heads`, `intermediate_size`, `mhc_expansion_rate`, ...). No dataset/optimizer/W&B settings. |
| `MethylFMModel` | The pretrained backbone: CpGEncoder -> tokenizer -> cross-attention -> mHC backbone -> attention pooling. Returns `last_hidden_state` (latent tokens) and `pooler_output` (the canonical sample embedding). No LeJEPA projector. |
| `MethylFMForPreTraining` | `MethylFMModel` + LeJEPA projector + the LeJEPA loss interface (`compute_lejepa_loss`). Backbone weights are not duplicated -- `pretraining_model.methylfm` is a real, usable `MethylFMModel` that can be exported on its own. |
| `MethylFMProcessor` | Converts CpG ids (`cg########`) + beta values into model tensors: validation, NaN removal, duplicate detection, padding, attention mask. With `cpg_store=` it maps ids to AlphaGenome-store rows (unmapped probes dropped and counted). |

## CpG locus encoder

The CpG encoder is a **frozen AlphaGenome prior + a small trainable adapter**
(`AlphaGenomeCpGEncoder`, 2.4M trainable parameters). AlphaGenome's 1-bp and
128-bp trunk embeddings of every CpG locus on the reference genome are
extracted **once, offline** (separate JAX environment, one GPU job) into a
memory-mappable store; MethylFM only looks them up. AlphaGenome is never run
during MethylFM training and the training environment has no JAX.

```text
reference DNA -> AlphaGenome (frozen, offline) -> (h_1bp, h_128bp)
              -> LN(W_in [h_1bp; h_128bp]) -> residual adapter -> l_i  -> FT tokenizer(l_i, beta) -> ...
```

Build and check the store with `methylfm audit-cpg-coordinates`,
`extract-cpg-embeddings`, `validate-cpg-embeddings`; see
[docs/CPG_ENCODER.md](docs/CPG_ENCODER.md) for the design, coordinate
conventions, tiling policy, store format and validation. `MockCpGEncoder`
(hash embedding, no biology) remains only for unit tests and `--smoke`.

## Current pretraining corpus

CpGCorpus, as used by [CpGPT](https://github.com/lcamillo/CpGPT) (Camillo et
al., bioRxiv 10.1101/2024.10.24.619766): 2000+ GEO datasets, 150,000+ samples
across 8 Illumina array platforms, stored as Apache Arrow tables
(`betas/QCDPB.arrow`, `betas/gse_betas.arrow`, `metadata/metadata.arrow`)
under `{GSE_ID}/{GPL_ID}/`, hosted on S3 (`s3://cpgpt-lucascamillo-public`,
requester-pays). See `src/methylfm/data/cpgcorpus.py`'s module docstring for
the exact, explicitly-flagged list of schema details that have **not** been
independently verified against a real downloaded shard (column names,
missing-value sentinel, EPICv2 probe-id harmonization) -- the adapter raises
a clear, actionable error rather than guessing at these.

## Non-goals for this milestone

- no WGBS;
- no phenotype fine-tuning;
- no beta reconstruction;
- no CpG imputation decoder;
- no sequence FM integration;
- no functional annotation encoder;
- no architecture ablation framework.

## Installation

```bash
pip install -e .
```

## Usage

### Pretraining

```bash
# real CpGCorpus (requires data.root set in configs/data/cpgcorpus.yaml or METHYLFM_DATA_ROOT)
python -m methylfm.training.pretrain --config configs/pretrain.yaml
accelerate launch -m methylfm.training.pretrain --config configs/pretrain.yaml

# synthetic smoke pretraining (no data required)
python -m methylfm.training.pretrain --smoke --steps 10
accelerate launch -m methylfm.training.pretrain --smoke --steps 10
methylfm smoke-test --steps 10   # thin CLI wrapper over the same path
```

### Data audit

```bash
methylfm audit-data --config configs/data/cpgcorpus.yaml
```

Reports dataset/platform/CpG-count distributions, missing betas, duplicate
sample ids, suspected duplicate profiles, and beta range -- or an actionable
error if the corpus has not yet been materialized locally.

### Downstream export

After pretraining, export just the backbone (no projector, no
pretraining-specific code):

```python
pretraining_model = MethylFMForPreTraining.from_pretrained("outputs/checkpoints/latest")
pretraining_model.methylfm.save_pretrained("outputs/methylfm-base")
```

## Repository structure

```text
src/methylfm/
├── configuration_methylfm.py   # MethylFMConfig
├── modeling_methylfm.py        # MethylFMPreTrainedModel / MethylFMModel / MethylFMForPreTraining
├── processing_methylfm.py      # MethylFMProcessor
├── outputs.py                  # MethylFMOutput / MethylFMForPreTrainingOutput
├── cli.py
│
├── model/              # CpGEncoder, tokenizer, cross-attention, mHC, pooling (unchanged architecture)
├── pretraining/        # LeJEPA projector, invariance, SIGReg, combined objective
├── data/               # MethylationSample contract, CpGCorpus adapter, views, collator, audit
├── training/           # pretrain.py (Accelerate), checkpoint.py, logging.py (W&B)
└── utils/
```

## Engineering notes

- One canonical architecture, one LeJEPA training path (`training/pretrain.py`),
  run identically single-GPU/multi-GPU/CPU via Hugging Face Accelerate.
- SIGReg's Epps-Pulley statistic is nonlinear in the empirical batch
  distribution, so under multi-process training each view's embeddings are
  gathered across processes (a differentiable all-gather) before SIGReg is
  computed -- see `training/pretrain.py` and
  `tests/pretraining/test_distributed_sigreg.py`.
- Checkpoints restore model, projector, optimizer, scheduler, RNG state, and
  the full resolved config; resume reproduces the same training trajectory
  as an uninterrupted run (see `tests/training/test_pretrain.py`).
- W&B is mandatory for non-test runs (project `MethylFM`, short run names
  like `smoke-s17`/`pretrain-s17`); disabled automatically for unit tests.
