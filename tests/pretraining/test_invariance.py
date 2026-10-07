import torch

from methylfm.pretraining.invariance import invariance_loss


def test_identical_views_near_zero_loss():
    z = torch.randn(8, 16)
    z_views = [z.clone() for _ in range(4)]
    loss = invariance_loss(z_views, global_view_indices=[0, 1])
    assert loss.item() < 1e-5


def test_divergent_views_give_positive_loss():
    torch.manual_seed(0)
    z_views = [torch.randn(8, 16) for _ in range(4)]
    loss = invariance_loss(z_views, global_view_indices=[0, 1])
    assert loss.item() > 1e-3


def test_empty_global_indices_raises():
    z_views = [torch.randn(4, 8) for _ in range(2)]
    try:
        invariance_loss(z_views, global_view_indices=[])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_out_of_range_index_raises():
    z_views = [torch.randn(4, 8) for _ in range(2)]
    try:
        invariance_loss(z_views, global_view_indices=[5])
        assert False, "expected ValueError"
    except ValueError:
        pass
