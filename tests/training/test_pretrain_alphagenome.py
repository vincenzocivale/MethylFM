"""A real (tiny) LeJEPA training run with the AlphaGenome encoder: the path
CpG ids -> frozen lookup -> adapter -> FT tokenizer -> MethylFM -> LeJEPA."""
import torch

from methylfm.cpg_embeddings.store import write_synthetic_store
from methylfm.training import pretrain as P


def test_pretrain_run_with_alphagenome_encoder(tmp_path):
    n = 200
    store = write_synthetic_store(tmp_path / "store", n, 16, 32, seed=5)
    config, _ = P._smoke_config(seed=0, max_steps=4)
    config["model"]["cpg_encoder"] = {
        "type": "alphagenome", "embedding_1bp_dim": 16, "embedding_128bp_dim": 32,
        "adapter_hidden_size": 64, "adapter_bottleneck_size": 8, "lookup_chunk_size": 128,
    }
    config["training"]["ckpt_dir"] = str(tmp_path / "ck")
    dataset = P.SyntheticMethylationDataset(16, vocab_size=n, min_cpgs=40, max_cpgs=120, seed=0)  # ids = store rows

    before = {}
    orig_build = P.MethylFMForPreTraining

    class Spy(orig_build):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            before.update({k_: v.detach().clone() for k_, v in self.named_parameters()})
            Spy.instance = self

    P.MethylFMForPreTraining = Spy
    try:
        result = P.run(config, dataset, "t", wandb_enabled=False, cpg_embedding_store=store)
    finally:
        P.MethylFMForPreTraining = orig_build

    assert result.global_step == 4 and torch.isfinite(torch.tensor(result.last_loss))
    after = dict(Spy.instance.named_parameters())
    enc = "methylfm.cpg_encoder."
    assert not torch.equal(before[enc + "adapter_up.weight"], after[enc + "adapter_up.weight"])  # adapter trained
    assert not torch.equal(before[enc + "input_proj.weight"], after[enc + "input_proj.weight"])
    # the 200 x (16+32) frozen table is not part of the checkpointed model
    ck = torch.load(tmp_path / "ck" / "latest.pt", map_location="cpu", weights_only=False)
    assert not any("embedding" in k.lower() for k in str(list(ck["model"].keys())).split(","))
