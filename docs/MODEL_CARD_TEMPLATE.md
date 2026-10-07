# Model Card Template -- MethylFM

This is a template for the model card to accompany a future
`methylfm/MethylFM-base` Hub release. Nothing has been published yet (see
the README's Status section) -- do not fill in claims this repository
hasn't validated.

## Model description

MethylFM is a foundation model for human DNA methylation arrays. It
consumes an arbitrary, variable-size set of observed `(CpG, beta)` pairs
(not a fixed universal CpG panel) and produces a fixed-dimensional sample
embedding (`pooler_output`), intended to be platform-robust across Illumina
array generations.

## Architecture

```text
CpG locus representation
  -> FT-style methylation tokenization: t = LayerNorm(f_b(l) + beta * f_w(l))
  -> set-to-latent cross-attention (learned latent array, chunked K/V)
  -> mHC Transformer backbone (manifold-constrained hyper-connections,
     doubly-stochastic residual mixing via Sinkhorn-Knopp)
  -> attention pooling -> pooler_output
```

<!-- Fill in final published hyperparameters once trained: hidden_size,
     num_hidden_layers, num_attention_heads, num_latents, mhc_expansion_rate,
     total parameter count. -->

## Pretraining objective

LeJEPA (Balestriero & LeCun, arXiv:2511.08544): multi-view invariance +
SIGReg (sliced Epps-Pulley Gaussianity regularization), no stop-gradient, no
teacher/EMA network, no contrastive negatives.

## CpGCorpus provenance

<!-- Fill in once a real pretraining run has completed: exact corpus
     snapshot/date, number of datasets and samples actually used,
     train/val split policy, any datasets excluded and why. -->

## Supported input

<!-- Fill in: which CpG identifier format (Illumina `cg########`?), beta
     value range and expected preprocessing (e.g. sesame QCDPB), minimum/
     maximum CpG set size tested. -->

## Array platforms evaluated

<!-- Do NOT claim support for a platform until validated. List only
     platforms this model has actually been evaluated on, with the
     evaluation method and result. -->

## Intended use

Produce a sample-level methylation embedding for downstream tasks (e.g.
phenotype prediction, clustering, transfer learning) via
`MethylFMModel`/`MethylFMProcessor`. Not intended for direct clinical
decision-making.

## Limitations

<!-- Fill in once known: e.g. the CpG locus encoder used at release time and
     its known failure modes, platforms/populations not represented in
     CpGCorpus, sensitivity to very small observed-CpG-set sizes, etc. -->

## Citation

<!-- Add once published: MethylFM citation, plus required citations for
     CpGPT/CpGCorpus and LeJEPA. -->

## License

<!-- Fill in before publishing. -->
