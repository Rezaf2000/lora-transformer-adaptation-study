"""A tiny character-level GPT used as the test bed for the LoRA experiment.

Nothing here is LoRA-specific -- it is a plain, small, autoregressive
transformer (embeddings, causal self-attention, a feed-forward block, a
language-modeling head) trained to predict the next character. The point of
keeping it small (2 layers, 2 heads, 128-d embeddings, 128-token context) is
that the whole pretrain-then-adapt experiment in scripts/run_experiment.py
has to fit in a modest CPU time budget; the architecture itself is standard
GPT-2-style and would scale up unchanged.

The attention and feed-forward linear layers are named `qkv` and `proj`
(inside SelfAttention) so that lora.apply_lora can target them by name
suffix without any GPT-specific code in lora.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 128
    dropout: float = 0.1


class SelfAttention(nn.Module):
    """Causal multi-head self-attention.

    The query, key and value projections live in one fused linear layer
    (`qkv`) and are split apart after the matmul; this is purely a
    performance convenience (one matmul instead of three) and does not
    change what the layer computes. `proj` is the output projection that
    mixes the concatenated heads back into a single n_embd-wide vector.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head

        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size))
        self.register_buffer("causal_mask", causal_mask.view(1, 1, config.block_size, config.block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, n_embd = x.shape
        q, k, v = self.qkv(x).split(n_embd, dim=2)

        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.causal_mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = self.attn_dropout(F.softmax(att, dim=-1))

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, n_embd)
        return self.resid_dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.fc2 = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm transformer block: attention then feed-forward, each with a
    residual connection around it."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ff = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        batch, seq_len = idx.shape
        if seq_len > self.config.block_size:
            raise ValueError(f"sequence length {seq_len} exceeds block_size {self.config.block_size}")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
