# CpG locus encoder: frozen AlphaGenome prior + trainable adapter

```
reference DNA ──AlphaGenome (frozen, offline, once)──▶ (h¹ᵇᵖ, h¹²⁸ᵇᵖ)   per CpG locus
                                                              │ lookup (no AlphaGenome at train time)
                                                              ▼
            u = LN(W_in [h¹ᵇᵖ ; h¹²⁸ᵇᵖ])                     trainable projection → 512
            l = LN(u + W_up GELU(W_down u))                    trainable residual adapter (r = 64)
                                                              │
            b = f_b(l),  w = f_w(l),  t_{s,i} = LN(b + β_{s,i} w)   existing FT tokenizer (unchanged)
```

Three objects that must not be conflated (in code or in prose):

| object | what it is | depends on | learned? |
|---|---|---|---|
| `h_i^AG = [h¹ᵇᵖ ; h¹²⁸ᵇᵖ]` | **frozen prior**: AlphaGenome trunk embeddings at CpG *i* | reference genome only | no — extracted once, stored |
| `l_i` | **methylation-adapted locus embedding** | `h_i^AG` + trainable adapter | yes (MethylFM) |
| `t_{s,i}` | **patient-specific methylation token** | `l_i` **and** `β_{s,i}` | via `f_b`, `f_w` |

`AlphaGenomeCpGEncoder` is patient-independent: it never sees β. AlphaGenome is
never executed during MethylFM training; the training environment has no JAX.

## 1. What is extracted

From the official `google-deepmind/alphagenome_research` model (JAX), via
`AlphaGenome.forward_trunk`, i.e. the **trunk representations before any
task-specific head**:

* `embeddings_1bp`   — `[S, 1536]`, one vector per base;
* `embeddings_128bp` — `[S/128, 3072]`, one vector per 128-bp bin.

(Not predicted tracks; no ATAC/DNase/histone/TF outputs are used.) The geometry
is asserted against the model at extraction time (`AlphaGenomeTrunk.check_geometry`):
the 1-bp output is aligned 1:1 with the input window and the 128-bp output has
exactly `S/128` bins with no crop or padding offset. Both scales are stored
independently, as float16 (see §6). Official max sequence length is `2**20`;
default window is `2**20` (read from the model by the check above; recorded in the manifest).

## 2. Coordinate convention

* Positions are **0-based**. A CpG locus is the 0-based coordinate `p` of the
  **cytosine on the forward strand of the reference**: `ref[p] == 'C'`, `ref[p+1] == 'G'`.
  CpG is self-complementary, so a probe on the opposite strand has the same `p`.
* Input manifests are the Zhou-lab hg38 annotations (HM27, HM450, EPIC, EPICv2, MSA);
  they are BED-like: `CpG_beg` is 0-based and **is** `p`; `CpG_end = CpG_beg + 2`.
* **Nothing is lifted over.** `reference_genome` in the config must be the assembly of
  both the manifests and the FASTA, and `methylfm audit-cpg-coordinates` verifies
  `ref[p:p+2] == "CG"` for every id (case-insensitive). A wrong build or a 0/1-based
  slip shows up as a mismatch rate ≈ 1 and the report says whether `CG` is found at
  `p±1, p±2`. Extraction refuses to start unless the audit passed *and* still matches
  the FASTA/manifest checksums.
* In-window indices (`coordinates.py`, unit-tested): 1-bp index `p − window_start`;
  128-bp bin `(p − window_start) // 128` (valid because window starts are multiples of 128
  and the model output has no offset — asserted, not assumed).

Ids → loci: several ids can name one locus (EPICv2 `cg…_TC21/_TC22`, aliases). The store
has one embedding per **locus**; `cpg_index.parquet` maps every id to its `row_index`.
The suffix-less base id of EPICv2/MSA replicates is registered as an alias only if all
replicates agree. Ids that are non-CpG probes, lack coordinates, conflict between arrays,
or fail the reference check get no row (`-1`) and are dropped by the data layer, never hashed.

## 3. Window tiling

`tiling: {window_size, core_size}` (default `2**20` / `2**19`). A chromosome is cut into
consecutive **cores**; core *k* owns `[k·core, (k+1)·core)`. Each core is processed in a
window of `window_size` centred on it, so every owned CpG has `margin = (window − core)/2`
(262 144 bp) of context on each side. Each CpG belongs to exactly one core, hence to
exactly one window: **embeddings from different windows are never averaged**. One forward
serves every CpG of its window. Near chromosome ends the window is shifted inside the
chromosome (or `N`-padded for chromosomes shorter than a window) and the CpGs that then lie
closer than `margin` to a window edge get `edge_flag = True`.

## 4. Pipeline

```bash
# 0. environments
pip install -e ".[dev]"                                 # training env: torch, no JAX
python -m venv --without-pip .venv-alphagenome          # extraction env: JAX, no torch
#   (or python -m venv .venv-alphagenome where ensurepip exists)
.venv-alphagenome/bin/pip install 'jax[cuda12]'
git clone https://github.com/google-deepmind/alphagenome_research && .venv-alphagenome/bin/pip install -e alphagenome_research pyfaidx
.venv-alphagenome/bin/pip install --no-deps -e .        # methylfm.cpg_embeddings only
# weights: accept the terms at https://huggingface.co/google/alphagenome-all-folds (or Kaggle), then `hf auth login`

# 1. audit (cheap, CPU, any env)                     -> outputs/cpg_encoder/{coordinate_audit.json,cpg_universe.parquet}
methylfm audit-cpg-coordinates --config configs/cpg_encoder/alphagenome.yaml
# 2. benchmark, then extract (GPU, JAX env; resumable)
methylfm extract-cpg-embeddings --config configs/cpg_encoder/alphagenome.yaml --benchmark
methylfm extract-cpg-embeddings --config configs/cpg_encoder/alphagenome.yaml
# 3. validate
methylfm validate-cpg-embeddings --config configs/cpg_encoder/alphagenome.yaml
```

Extraction is resumable: progress is persisted after every batch of windows *after* the data
is flushed; completed windows are never recomputed; a partial store cannot be opened for
training. Re-running with different tiling/FASTA/model settings into an existing store is refused.
`configs/cpg_encoder/alphagenome_pilot.yaml` runs the same pipeline on chr22 only.

## 5. Store format (`store_dir/`)

| file | content |
|---|---|
| `manifest.json` | provenance (below), geometry, checksums, `complete`, `store_id` |
| `cpg_index.parquet` | `internal_index, cpg_id, chromosome, position, row_index, window_id, edge_flag, reference_valid, is_alias` |
| `loci.parquet` | one row per embedding row: position, owning window/core, `window_start`, `bp_index`, `bin_index`, `edge_flag` |
| `alphagenome_1bp.npy`, `alphagenome_128bp.npy` | float16 `[num_loci, 1536]`, `[num_loci, 3072]`, memory-mappable |
| `progress.json` | completed window ids |

`np.load(..., mmap_mode="r")`: lookups touch only the requested rows; nothing is loaded whole.
Row order = (natural chromosome order, position), a pure function of the universe.
Manifest provenance: source model/repo/commit, checkpoint id, reference genome + FASTA SHA-256,
manifest SHA-256, sequence length, stride, core size, embedding dims, dtype, counts, creation
time, extraction software commit. `store_id` is a hash of the immutable content descriptors.

## 6. Storage precision

AlphaGenome's official inference computes in bfloat16 and emits bfloat16 embeddings. The store is
float16 (more mantissa bits than bf16; the extraction aborts if any value overflows float16 range
or is non-finite). `validate-cpg-embeddings` reports fp32-compute vs official bf16-compute and
fp16-storage vs bf16 (cosine + max abs error).

## 7. The trainable encoder (`model/cpg_encoder.py`)

`AlphaGenomeCpGEncoder`: `u = LN(W_in [h¹ᵇᵖ; h¹²⁸ᵇᵖ])` (4608 → 512), `l = LN(u + W_up GELU(W_down u))`
(512 → 64 → 512), shared over all loci. **No per-CpG parameters** (no `nn.Embedding(num_cpgs, …)`);
any locus with an AlphaGenome row is supported. `W_up` is zero-initialised (also after Hugging Face's
`post_init`), so at the start `l = LN(u) ≈ u`. The encoder adds **2 427 968** trainable parameters (`W_in` 2 359 808, adapter 66 112, 2 LayerNorms 2 048).
Not included by design: scale gating, cross-scale attention, per-scale branches, extra tracks.

* **Input ids** are store row indices. `MethylFMProcessor(cpg_store=…)` and `CpGCorpusDataset(…, cpg_store=…)`
  apply the same mapping (`CpGEmbeddingStore.encode`); unmapped probes are dropped and counted
  (`unmapped_counts`). Two ids naming the same locus in one sample raise (deduplicate/average first).
  Negative ids are padding and encode to zeros.
* **Efficiency.** `l_i` depends only on the locus, so each forward computes it once per *distinct* locus in the
  batch, in chunks, under activation checkpointing (the 4608-d frozen input is not kept for backward), then
  indexes. `lookup_mode: gather` (default) reads only the needed rows from the memory-mapped store;
  `resident` keeps the table on the GPU (`num_loci · (1536+3072) · 2` bytes = 10.1 GiB in fp16 for the
  1 176 333-locus universe of the 5 arrays) — use it on large GPUs if host→device traffic dominates. The store is a plain attribute: not in
  `state_dict`, not a buffer, not broadcast by DDP.
* **Not cached:** adapter outputs (they change every optimizer step). Only immutable AlphaGenome rows are ever cached.
* **HF integration.** `config.json` carries `cpg_embedding_asset_id` (the store's id) and dims — never a path. The store
  is located by `cpg_embedding_store=` argument, `$METHYLFM_CPG_EMBEDDING_STORE`, or
  `~/.cache/methylfm/cpg_embeddings/<asset_id>`; a store whose id/dims differ from the config is rejected.
  Likewise `preprocessor_config.json`.

## 8. Validation (what `validate-cpg-embeddings` does)

Integrity (checksums, completeness, no NaN/unwritten rows, id↔row consistency); representation sanity
(norms, per-dimension variance, effective rank, duplicate rate, random-pair cosine, nearest-neighbour
chromosome structure); and, with the official model: **tiled vs independently centred window** (1-bp and
128-bp, phase-matched and exactly centred), **determinism** of re-extraction, **boundary sensitivity**
(shifted windows; CpG placed 1 kb / 8 kb / 64 kb from a window edge), **precision** (fp32 vs bf16 compute,
fp16 storage), and a **reverse-complement audit** (audit only; the store uses the forward orientation).

## 9. Known limitations

* Embeddings are context-dependent on the *reference* sequence only; individual genotypes are ignored by design.
* The CpG id spelling used for EPICv2/MSA in the pretraining corpus is unverified (`data/cpgcorpus.py`); both
  suffixed and base ids resolve, but ids spelled otherwise are dropped (and counted).
* Only the forward genomic orientation is stored.
