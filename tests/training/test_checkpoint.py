import torch

from methylfm.training.checkpoint import save_checkpoint, load_checkpoint, save_latest_and_maybe_best
from methylfm.utils.seed import seed_everything


def _make_model_projector_optimizer():
    model = torch.nn.Linear(8, 8)
    projector = torch.nn.Linear(8, 4)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(projector.parameters()), lr=1e-3
    )
    return model, projector, optimizer


def _train_steps(model, projector, optimizer, steps=3):
    for _ in range(steps):
        optimizer.zero_grad()
        x = torch.randn(4, 8)
        out = projector(model(x))
        loss = out.pow(2).mean()
        loss.backward()
        optimizer.step()


def test_checkpoint_roundtrip(tmp_path):
    seed_everything(1234)
    model, projector, optimizer = _make_model_projector_optimizer()
    _train_steps(model, projector, optimizer, steps=3)

    ckpt_path = str(tmp_path / "ckpt.pt")
    save_checkpoint(
        ckpt_path, model, projector, optimizer, scheduler=None, scaler=None,
        epoch=2, global_step=42, config={"lr": 1e-3, "nested": {"a": 1}},
    )

    reference_draw = torch.randn(5, 5)

    new_model, new_projector, new_optimizer = _make_model_projector_optimizer()
    for p_old, p_new in zip(
        list(model.parameters()) + list(projector.parameters()),
        list(new_model.parameters()) + list(new_projector.parameters()),
    ):
        assert not torch.allclose(p_old, p_new)

    meta = load_checkpoint(ckpt_path, new_model, new_projector, optimizer=new_optimizer)

    assert meta.epoch == 2
    assert meta.global_step == 42
    assert meta.config == {"lr": 1e-3, "nested": {"a": 1}}

    for (k, v_old), (k2, v_new) in zip(model.state_dict().items(), new_model.state_dict().items()):
        assert k == k2
        assert torch.allclose(v_old, v_new)
    for (k, v_old), (k2, v_new) in zip(
        projector.state_dict().items(), new_projector.state_dict().items()
    ):
        assert k == k2
        assert torch.allclose(v_old, v_new)

    old_opt_state = optimizer.state_dict()["state"]
    new_opt_state = new_optimizer.state_dict()["state"]
    assert set(old_opt_state.keys()) == set(new_opt_state.keys())
    for key in old_opt_state:
        for sk in old_opt_state[key]:
            v_old = old_opt_state[key][sk]
            v_new = new_opt_state[key][sk]
            if torch.is_tensor(v_old):
                assert torch.allclose(v_old, v_new)
            else:
                assert v_old == v_new

    post_load_draw = torch.randn(5, 5)
    assert torch.allclose(reference_draw, post_load_draw)


def test_save_latest_and_maybe_best(tmp_path):
    model, projector, optimizer = _make_model_projector_optimizer()
    ckpt_dir = str(tmp_path / "ckpts")

    best_so_far = save_latest_and_maybe_best(
        ckpt_dir, model, projector, optimizer, None, None,
        epoch=0, global_step=10, config={}, metric=1.0, best_so_far=None,
    )
    assert best_so_far == 1.0

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    best_so_far = save_latest_and_maybe_best(
        ckpt_dir, model, projector, optimizer, None, None,
        epoch=1, global_step=20, config={}, metric=0.5, best_so_far=best_so_far,
    )
    assert best_so_far == 0.5

    best_model_snapshot = {k: v.clone() for k, v in model.state_dict().items()}

    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)

    best_so_far_after_worse = save_latest_and_maybe_best(
        ckpt_dir, model, projector, optimizer, None, None,
        epoch=2, global_step=30, config={}, metric=0.9, best_so_far=best_so_far,
    )
    assert best_so_far_after_worse == best_so_far == 0.5

    fresh_model, fresh_projector, fresh_optimizer = _make_model_projector_optimizer()
    meta = load_checkpoint(
        str(__import__("pathlib").Path(ckpt_dir) / "best.pt"),
        fresh_model, fresh_projector, optimizer=fresh_optimizer,
    )
    assert meta.global_step == 20
    for k, v in fresh_model.state_dict().items():
        assert torch.allclose(v, best_model_snapshot[k])

    fresh_model2, fresh_projector2, fresh_optimizer2 = _make_model_projector_optimizer()
    meta_latest = load_checkpoint(
        str(__import__("pathlib").Path(ckpt_dir) / "latest.pt"),
        fresh_model2, fresh_projector2, optimizer=fresh_optimizer2,
    )
    assert meta_latest.global_step == 30
