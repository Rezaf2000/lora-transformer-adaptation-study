"""Tests for the plain MiniGPT, independent of LoRA."""

from __future__ import annotations

import torch

from lora.model import GPTConfig, MiniGPT


def make_model(seed=0, **overrides):
    torch.manual_seed(seed)
    defaults = dict(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    defaults.update(overrides)
    cfg = GPTConfig(**defaults)
    return MiniGPT(cfg), cfg


def test_forward_shapes():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = model(idx)
    assert logits.shape == (3, 10, cfg.vocab_size)
    assert loss is None


def test_forward_with_targets_returns_scalar_loss():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (3, 10))
    targets = torch.randint(0, cfg.vocab_size, (3, 10))
    logits, loss = model(idx, targets)
    assert loss.dim() == 0
    assert loss.item() > 0


def test_sequence_longer_than_block_size_raises():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (1, cfg.block_size + 1))
    try:
        model(idx)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_causal_masking():
    """Changing a later token must not change the logits at earlier
    positions -- the defining property of an autoregressive model."""
    model, cfg = make_model()
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    logits_a, _ = model(idx)

    idx_changed = idx.clone()
    idx_changed[0, -1] = (idx_changed[0, -1] + 1) % cfg.vocab_size
    logits_b, _ = model(idx_changed)

    assert torch.allclose(logits_a[0, :-1], logits_b[0, :-1], atol=1e-5)
    assert not torch.allclose(logits_a[0, -1], logits_b[0, -1])


def test_generate_appends_requested_number_of_tokens():
    model, cfg = make_model()
    idx = torch.randint(0, cfg.vocab_size, (2, 4))
    out = model.generate(idx, max_new_tokens=5)
    assert out.shape == (2, 9)
    assert torch.equal(out[:, :4], idx)


def test_num_parameters_matches_manual_sum():
    model, _ = make_model()
    manual = sum(p.numel() for p in model.parameters())
    assert model.num_parameters() == manual
    assert model.num_parameters() > 0


def test_n_embd_not_divisible_by_n_head_raises():
    try:
        make_model(n_head=3)  # 16 is not divisible by 3
        assert False, "expected ValueError"
    except ValueError:
        pass
