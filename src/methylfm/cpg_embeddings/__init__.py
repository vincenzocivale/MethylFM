"""Offline pipeline producing the frozen AlphaGenome CpG embedding store, and
the read-only store used by `methylfm.model.cpg_encoder.AlphaGenomeCpGEncoder`.

Importing this package never imports JAX or torch; `extract` / `validate`
import JAX lazily and only run in the dedicated extraction environment.
"""
