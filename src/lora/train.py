"""Shared training loop utilities for both pretraining and adaptation.

Kept deliberately generic: it knows nothing about LoRA, full fine-tuning or
head-only training. Every one of those is just "some parameters have
requires_grad=True and the rest don't" by the time a model reaches this
module, so a single train_loop covers all of them -- the optimizer already
only sees trainable parameters, thanks to `filter(lambda p: p.requires_grad, ...)`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .data import get_batch


@torch.no_grad()
def estimate_loss(model, data: torch.Tensor, block_size: int, batch_size: int, eval_iters: int) -> float:
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(data, block_size, batch_size)
        _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()


@dataclass
class TrainResult:
    history: list[dict] = field(default_factory=list)
    final_val_loss: float = 0.0
    wall_seconds: float = 0.0


def train_loop(
    model,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    steps: int,
    lr: float,
    eval_every: int = 100,
    eval_iters: int = 20,
    weight_decay: float = 0.0,
    verbose: bool = True,
    log_prefix: str = "",
) -> TrainResult:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    history = []
    t0 = time.time()
    model.train()
    for step in range(steps + 1):
        if step % eval_every == 0 or step == steps:
            val_loss = estimate_loss(model, val_data, block_size, batch_size, eval_iters)
            train_loss = estimate_loss(model, train_data, block_size, batch_size, eval_iters)
            history.append({"step": step, "train_loss": train_loss, "val_loss": val_loss})
            if verbose:
                print(f"{log_prefix}step {step:>5}/{steps}  train {train_loss:.4f}  val {val_loss:.4f}")
        if step == steps:
            break
        x, y = get_batch(train_data, block_size, batch_size)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    wall = time.time() - t0
    if verbose:
        print(f"{log_prefix}done in {wall:.1f}s")
    return TrainResult(history=history, final_val_loss=history[-1]["val_loss"], wall_seconds=wall)
