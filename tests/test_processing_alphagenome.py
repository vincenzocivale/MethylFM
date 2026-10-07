import json

import pytest
import torch

from methylfm import MethylFMConfig, MethylFMModel, MethylFMProcessor
from methylfm.cpg_embeddings.store import StoreIncompatibleError, write_synthetic_store

IDS = [f"cg{i:08d}" for i in range(40)]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return write_synthetic_store(tmp_path_factory.mktemp("s"), len(IDS), 16, 32, seed=1, cpg_ids=IDS)


def config(**kw):
    return MethylFMConfig(
        hidden_size=32, num_latents=8, num_hidden_layers=2, num_attention_heads=4, intermediate_size=64,
        mhc_expansion_rate=2, mhc_sinkhorn_iters=3, locus_dim=24, cpg_encoder_type="alphagenome",
        cpg_embedding_1bp_dim=16, cpg_embedding_128bp_dim=32, cpg_adapter_hidden_size=24,
        cpg_adapter_bottleneck_size=6, **kw)


def test_processor_maps_ids_to_store_rows_and_drops_unmapped(store):
    proc = MethylFMProcessor(cpg_store=store)
    out = proc(cpg_ids=[["cg00000005", "cg00000001", "unknown_probe", "cg00000007"], ["cg00000002", "nope"]],
               beta_values=[[0.1, 0.2, 0.3, float("nan")], [0.5, 0.6]])
    assert out["cpg_ids"][0, :2].tolist() == [5, 1] and out["attention_mask"].tolist() == [[True, True], [True, False]]
    assert out.unmapped_counts == [1, 1] and out.dropped_counts == [1, 0]
    assert out["cpg_ids"][1, 0].item() == 2


def test_processor_rejects_two_ids_for_the_same_locus(tmp_path):
    s = write_synthetic_store(tmp_path / "s", 3, 4, 4, cpg_ids=["cg1", "cg1_TC21", "cg2"])
    # make the two spellings resolve to one row, as the real store does for replicates
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    t = pq.read_table(s.path / "cpg_index.parquet")
    rows = t["row_index"].to_numpy().copy(); rows[1] = rows[0]
    s._vocab = (np.array(t["cpg_id"].to_pylist(), dtype=str), rows)
    with pytest.raises(ValueError, match="same genomic locus"):
        MethylFMProcessor(cpg_store=s)(cpg_ids=["cg1", "cg1_TC21"], beta_values=[0.2, 0.4])


def test_processor_output_feeds_the_model(store):
    proc = MethylFMProcessor(cpg_store=store)
    model = MethylFMModel(config(), cpg_embedding_store=store).eval()
    inputs = proc(cpg_ids=[IDS[:10] + IDS[20:25], IDS[5:20]],
                  beta_values=[[0.3] * 15, [0.7] * 15])
    out = model(**inputs)
    assert out.pooler_output.shape == (2, 32) and torch.isfinite(out.pooler_output).all()


def test_processor_save_load_stores_only_the_asset_id(store, tmp_path, monkeypatch):
    MethylFMProcessor(cpg_store=store).save_pretrained(tmp_path / "p")
    text = (tmp_path / "p" / "preprocessor_config.json").read_text()
    assert str(store.path) not in text and json.loads(text)["cpg_embedding_asset_id"] == store.asset_id
    with pytest.raises(FileNotFoundError):
        MethylFMProcessor.from_pretrained(tmp_path / "p")
    monkeypatch.setenv("METHYLFM_CPG_EMBEDDING_STORE", str(store.path))
    again = MethylFMProcessor.from_pretrained(tmp_path / "p")
    a = again(cpg_ids=IDS[:4], beta_values=[0.1] * 4)["cpg_ids"]
    assert a.tolist() == [[0, 1, 2, 3]]
    other = write_synthetic_store(tmp_path / "o", len(IDS), 16, 32, seed=2, cpg_ids=IDS)
    with pytest.raises(StoreIncompatibleError):
        MethylFMProcessor.from_pretrained(tmp_path / "p", cpg_embedding_store=other)


def test_legacy_mock_processor_unchanged():
    out = MethylFMProcessor(vocab_size=1000)(cpg_ids=["cg00000029"], beta_values=[0.5])
    assert out["cpg_ids"].tolist() == [[29]] and out.unmapped_counts == [0]
