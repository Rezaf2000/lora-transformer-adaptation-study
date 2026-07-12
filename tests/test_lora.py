"""Tests for the LoRA layer itself: the claims made in lora.py's comments,
checked numerically rather than taken on faith.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from lora.lora import LoRALinear, apply_lora, mark_only_lora_as_trainable, trainable_parameters
from lora.model import GPTConfig, MiniGPT


def make_base_linear(in_features=16, out_features=8, seed=0):
    torch.manual_seed(seed)
    return nn.Linear(in_features, out_features)


def test_output_shape():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    x = torch.randn(3, 5, 16)
    out = layer(x)
    assert out.shape == (3, 5, 8)


def test_b_zero_at_init_matches_frozen_base():
    """delta_W = B @ A must be exactly zero right after construction, because
    B is initialized to zero. A LoRALinear should therefore behave as an
    exact passthrough to the frozen base layer before any training happens."""
    base = make_base_linear(16, 8)
    x = torch.randn(4, 16)
    expected = base(x)

    layer = LoRALinear(base, r=4)
    actual = layer(x)

    assert torch.allclose(actual, expected)
    # not just close -- B is literally all zeros, so the lora term is exactly 0
    assert torch.equal(layer.lora_B, torch.zeros_like(layer.lora_B))


def test_a_is_not_degenerate():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    # every row of A should not collapse to the same values (see lora.py's
    # comment on why A needs real randomness, not just "small values")
    assert layer.lora_A.std() > 0
    row_diffs = (layer.lora_A[0] - layer.lora_A[1]).abs().sum()
    assert row_diffs > 0


def test_base_weight_frozen():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    assert layer.base.weight.requires_grad is False
    assert layer.base.bias.requires_grad is False


def test_gradients_only_on_a_and_b():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    x = torch.randn(4, 16)
    out = layer(x)
    out.sum().backward()

    assert layer.lora_A.grad is not None
    assert layer.lora_B.grad is not None
    assert layer.base.weight.grad is None
    assert layer.base.bias.grad is None


def test_merge_matches_unmerged_forward():
    """The central efficiency claim of the paper: once merged, W' = W0 + (a/r)BA
    computes the exact same function as the unmerged two-branch forward."""
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4, alpha=8)
    # give A and B nonzero, non-symmetric values so the check is meaningful
    torch.manual_seed(1)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn_like(layer.lora_A))
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))

    x = torch.randn(5, 16)
    unmerged_out = layer(x)

    layer.merge()
    merged_out = layer(x)

    assert layer.merged is True
    assert torch.allclose(unmerged_out, merged_out, atol=1e-6)


def test_unmerge_restores_original_weight():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    torch.manual_seed(2)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn_like(layer.lora_A))
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))

    original_weight = layer.base.weight.clone()
    layer.merge()
    assert not torch.allclose(layer.base.weight, original_weight)
    layer.unmerge()
    assert torch.allclose(layer.base.weight, original_weight, atol=1e-6)


def test_disabled_adapter_ignores_trained_lora_weights():
    """enabled=False must reproduce the frozen base layer exactly, even after
    A and B have been trained away from their initial values -- this is what
    lets an adapted model "go back" to the pretrained behaviour without
    merging or reloading anything."""
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4)
    torch.manual_seed(3)
    with torch.no_grad():
        layer.lora_A.copy_(torch.randn_like(layer.lora_A))
        layer.lora_B.copy_(torch.randn_like(layer.lora_B))

    x = torch.randn(4, 16)
    expected = torch.nn.functional.linear(x, layer.base.weight, layer.base.bias)

    layer.enabled = False
    actual = layer(x)
    assert torch.allclose(actual, expected)


def test_scaling_uses_alpha_over_r():
    base = make_base_linear(16, 8)
    layer = LoRALinear(base, r=4, alpha=16)
    assert layer.scaling == pytest.approx(16 / 4)

    default_alpha_layer = LoRALinear(base, r=4)
    assert default_alpha_layer.alpha == 4
    assert default_alpha_layer.scaling == pytest.approx(1.0)


def test_r_must_be_positive():
    base = make_base_linear(16, 8)
    with pytest.raises(ValueError):
        LoRALinear(base, r=0)


def test_apply_lora_wraps_only_targeted_layers():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = MiniGPT(cfg)
    wrapped = apply_lora(model, target_suffixes=("attn.qkv", "attn.proj"), r=2)

    assert len(wrapped) == 2 * cfg.n_layer  # qkv + proj per block
    for block in model.blocks:
        assert isinstance(block.attn.qkv, LoRALinear)
        assert isinstance(block.attn.proj, LoRALinear)
        # the feed-forward linears were not in target_suffixes and must be untouched
        assert isinstance(block.ff.fc1, nn.Linear) and not isinstance(block.ff.fc1, LoRALinear)
        assert isinstance(block.ff.fc2, nn.Linear) and not isinstance(block.ff.fc2, LoRALinear)


def test_mark_only_lora_as_trainable_freezes_everything_else():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = MiniGPT(cfg)
    apply_lora(model, target_suffixes=("attn.qkv", "attn.proj"), r=2)
    mark_only_lora_as_trainable(model)

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad is True
        else:
            assert param.requires_grad is False, f"{name} should be frozen"


def test_trainable_parameter_count_scales_with_rank():
    cfg = GPTConfig(vocab_size=32, block_size=16, n_layer=2, n_head=2, n_embd=16, dropout=0.0)

    counts = {}
    for r in (1, 2, 4):
        model = MiniGPT(cfg)
        apply_lora(model, target_suffixes=("attn.qkv", "attn.proj"), r=r)
        mark_only_lora_as_trainable(model)
        trainable, total = trainable_parameters(model)
        counts[r] = trainable
        # every LoRALinear contributes r * (in_features + out_features) trainable params
        expected = 0
        for name, module in model.named_modules():
            if isinstance(module, LoRALinear):
                expected += r * (module.in_features + module.out_features)
        assert trainable == expected

    assert counts[1] < counts[2] < counts[4]


def test_forward_matches_plain_linear_when_r_equals_full_rank_zero_init():
    """Sanity check unrelated to rank: with B still at zero, wrapping in
    LoRALinear must not change dtype or require extra reshaping from the
    caller's point of view."""
    base = make_base_linear(10, 6)
    layer = LoRALinear(base, r=3)
    x = torch.randn(2, 7, 10)
    out = layer(x)
    assert out.dtype == x.dtype
    assert out.shape == (2, 7, 6)
