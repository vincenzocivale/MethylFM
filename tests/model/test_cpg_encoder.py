import pytest
import torch

from methylfm.model.cpg_encoder import CpGEncoder, MockCpGEncoder


def test_mock_encoder_output_shape():
    encoder = MockCpGEncoder(locus_dim=8)
    cpg_ids = torch.randint(0, 2**20, (4, 37), dtype=torch.long)
    out = encoder(cpg_ids)
    assert out.shape == cpg_ids.shape + (8,)


class _BrokenEncoder(CpGEncoder):
    def forward(self, cpg_ids, cpg_metadata=None):
        output = torch.zeros(cpg_ids.shape + (self.locus_dim + 1,))
        self._validate_output(cpg_ids, output)
        return output


def test_validate_output_raises_on_shape_mismatch():
    encoder = _BrokenEncoder(locus_dim=8)
    cpg_ids = torch.randint(0, 1000, (3, 5), dtype=torch.long)
    with pytest.raises(AssertionError):
        encoder(cpg_ids)


def test_base_cpg_encoder_forward_not_implemented():
    encoder = CpGEncoder(locus_dim=8)
    cpg_ids = torch.randint(0, 1000, (2, 3), dtype=torch.long)
    with pytest.raises(NotImplementedError):
        encoder(cpg_ids)
