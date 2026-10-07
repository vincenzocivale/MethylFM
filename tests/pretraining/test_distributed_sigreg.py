"""Distributed-correctness test for SIGReg (spec section 14): SIGReg's
Epps-Pulley statistic is a nonlinear function of the empirical batch
distribution, so it must be computed over the GLOBAL batch, not
independently per process. This spawns 2 real `gloo` processes (CPU-only,
no GPU needed) and checks that gathering `z` across processes before
`sigreg_loss` reproduces the single-process global-batch statistic, while
naively averaging each process's LOCAL statistic does not.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from methylfm.pretraining.sigreg import sigreg_loss

N_PER_RANK = 6
D = 4
NUM_SLICES = 128
NUM_POINTS = 17
SEED = 123


def _make_full_z() -> torch.Tensor:
    gen = torch.Generator().manual_seed(SEED)
    return torch.randn(2 * N_PER_RANK, D, generator=gen)


def _worker(rank: int, world_size: int, port: int, results: dict) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    full_z = _make_full_z()
    local_z = full_z[rank * N_PER_RANK : (rank + 1) * N_PER_RANK].clone().requires_grad_(True)

    # (a) naive: SIGReg computed independently on this rank's local shard only.
    local_only_stat = sigreg_loss(local_z, num_slices=NUM_SLICES, num_points=NUM_POINTS,
                                   generator=torch.Generator().manual_seed(SEED))

    # (b) correct: differentiable all-gather across ranks, then SIGReg on the
    # reassembled global batch.
    gathered = torch.distributed.nn.functional.all_gather(local_z)
    global_z = torch.cat(gathered, dim=0)
    gathered_stat = sigreg_loss(global_z, num_slices=NUM_SLICES, num_points=NUM_POINTS,
                                 generator=torch.Generator().manual_seed(SEED))
    gathered_stat.backward()

    results[rank] = {
        "local_only_stat": local_only_stat.item(),
        "gathered_stat": gathered_stat.item(),
        "local_grad_norm": local_z.grad.norm().item(),
    }
    dist.destroy_process_group()


def test_gathered_sigreg_matches_single_process_global_batch():
    world_size = 2
    manager = mp.Manager()
    results = manager.dict()
    port = 29513

    mp.spawn(_worker, args=(world_size, port, results), nprocs=world_size, join=True)

    assert set(results.keys()) == {0, 1}

    # Single-process ground truth: SIGReg over the full concatenated batch.
    full_z = _make_full_z()
    single_process_stat = sigreg_loss(
        full_z, num_slices=NUM_SLICES, num_points=NUM_POINTS, generator=torch.Generator().manual_seed(SEED)
    ).item()

    for rank in (0, 1):
        gathered_stat = results[rank]["gathered_stat"]
        assert abs(gathered_stat - single_process_stat) < 1e-4, (
            f"rank {rank}: gathered multi-process SIGReg ({gathered_stat}) should match the "
            f"single-process global-batch SIGReg ({single_process_stat})"
        )
        # Each rank's local shard should have received a non-trivial gradient
        # from the GLOBAL statistic (proving gradients flow back through the
        # differentiable all-gather to this rank's own local z).
        assert results[rank]["local_grad_norm"] > 0.0

    # And demonstrate why this matters: naively computing SIGReg
    # independently per rank gives a DIFFERENT (wrong) statistic than the
    # true global-batch one -- this is the bug `gather_fn` avoids.
    naive_average = (results[0]["local_only_stat"] + results[1]["local_only_stat"]) / 2
    assert abs(naive_average - single_process_stat) > 1e-3, (
        "expected naive per-rank-averaged SIGReg to differ materially from the true "
        "global-batch statistic -- if this fails, the test's premise (and the need for "
        "gather_fn) should be revisited"
    )
