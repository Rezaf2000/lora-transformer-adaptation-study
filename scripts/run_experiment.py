"""The real experiment: pretrain on Shakespeare, then adapt to Don Quijote
five different ways, and compare them on equal footing.

Methods compared, all given the exact same number of adaptation steps:

  full   -- every parameter in the model is trainable (the usual fine-tune).
  lora1  -- LoRA on the attention projections, rank 1.
  lora4  -- LoRA on the attention projections, rank 4.
  lora16 -- LoRA on the attention projections, rank 16.
  head   -- only the output head (the final linear layer) is trainable;
            everything else, including all of attention, stays exactly as
            pretrained.

For every method we record how many parameters were actually trainable, the
final validation loss on Quijote (corpus B), and -- the part that does not
show up in a normal fine-tuning writeup -- the validation loss back on
Shakespeare (corpus A) *after* adapting to Quijote. That last number is
catastrophic forgetting made concrete: it is what "the model that used to
speak Shakespearean English" looks like once you have spent 800 steps
overwriting its weights with Cervantes.

For the LoRA runs specifically, we also evaluate corpus A with the adapters
switched off (LoRALinear.enabled = False everywhere). Because LoRA never
writes into the frozen base weights during training, that number should
reproduce the pretrained model's corpus-A loss essentially exactly -- the
whole adaptation is sitting in a separate, removable set of matrices.

Run with: python scripts/run_experiment.py
Budget: roughly 20-30 minutes on a laptop CPU with the defaults below.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lora.data import CharTokenizer, encode_split, get_batch, load_text  # noqa: E402
from lora.lora import apply_lora, mark_only_lora_as_trainable, trainable_parameters  # noqa: E402
from lora.model import GPTConfig, MiniGPT  # noqa: E402
from lora.train import estimate_loss, train_loop  # noqa: E402

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"

SEED = 1337
BLOCK_SIZE = 128
BATCH_SIZE = 64
N_LAYER = 2
N_HEAD = 2
N_EMBD = 128
DROPOUT = 0.1
LR = 3e-4

PRETRAIN_STEPS = 2500
PRETRAIN_EVAL_EVERY = 250
ADAPT_STEPS = 800
ADAPT_EVAL_EVERY = 100
EVAL_ITERS = 20
FINAL_EVAL_ITERS = 50

LORA_TARGETS = ("attn.qkv", "attn.proj")
GENERATE_TOKENS = 300


def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = GENERATE_TOKENS) -> str:
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.8)
    model.train()
    return tokenizer.decode(out[0].tolist())


def build_method_model(cfg: GPTConfig, pretrained_state: dict, method: str):
    """Return (model, trainable_params, total_params) for one of the five methods."""
    model = MiniGPT(cfg)
    model.load_state_dict(pretrained_state)

    if method == "full":
        for p in model.parameters():
            p.requires_grad_(True)
    elif method == "head":
        for p in model.parameters():
            p.requires_grad_(False)
        model.head.weight.requires_grad_(True)
    elif method.startswith("lora"):
        r = int(method.replace("lora", ""))
        apply_lora(model, LORA_TARGETS, r=r)
        mark_only_lora_as_trainable(model)
    else:
        raise ValueError(f"unknown method {method!r}")

    trainable, total = trainable_parameters(model)
    return model, trainable, total


def disable_all_adapters(model, enabled: bool) -> None:
    for module in model.modules():
        if hasattr(module, "enabled") and hasattr(module, "lora_A"):
            module.enabled = enabled


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    experiment_t0 = time.time()

    torch.manual_seed(SEED)

    text_a = load_text(DATA_DIR / "shakespeare.txt")
    text_b = load_text(DATA_DIR / "quijote.txt")
    tokenizer = CharTokenizer.build(text_a, text_b)
    tokenizer.save(CHECKPOINT_DIR / "tokenizer.json")
    print(f"vocab size (union of both corpora): {tokenizer.vocab_size}")

    train_a, val_a = encode_split(text_a, tokenizer)
    train_b, val_b = encode_split(text_b, tokenizer)
    print(f"corpus A (Shakespeare): {len(train_a):,} train / {len(val_a):,} val tokens")
    print(f"corpus B (Quijote):     {len(train_b):,} train / {len(val_b):,} val tokens")

    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_layer=N_LAYER,
        n_head=N_HEAD,
        n_embd=N_EMBD,
        dropout=DROPOUT,
    )

    # ---- Stage 1: pretrain on corpus A -----------------------------------
    print("\n=== pretraining on corpus A (Shakespeare) ===")
    base_model = MiniGPT(cfg)
    total_params = base_model.num_parameters()
    print(f"model: {N_LAYER} layers, {N_HEAD} heads, d_model={N_EMBD}, "
          f"block_size={BLOCK_SIZE}, {total_params:,} parameters")

    pretrain_result = train_loop(
        base_model, train_a, val_a, BLOCK_SIZE, BATCH_SIZE,
        steps=PRETRAIN_STEPS, lr=LR, eval_every=PRETRAIN_EVAL_EVERY,
        eval_iters=EVAL_ITERS, log_prefix="[pretrain] ",
    )
    pretrain_val_a = estimate_loss(base_model, val_a, BLOCK_SIZE, BATCH_SIZE, FINAL_EVAL_ITERS)
    pretrain_sample = generate_sample(base_model, tokenizer, prompt="ROMEO:")
    print(f"pretrain final val loss (A): {pretrain_val_a:.4f}")
    print(f"sample:\n{pretrain_sample}\n")

    pretrained_state = copy.deepcopy(base_model.state_dict())
    torch.save(
        {"model_state": pretrained_state, "config": cfg.__dict__},
        CHECKPOINT_DIR / "pretrained.pt",
    )

    # ---- Stage 2: adapt to corpus B, five ways ----------------------------
    methods = ["full", "lora1", "lora4", "lora16", "head"]
    method_results = []

    for i, method in enumerate(methods):
        print(f"\n=== adapting to corpus B (Quijote) -- method: {method} ===")
        torch.manual_seed(SEED + 1 + i)
        model, trainable, total = build_method_model(cfg, pretrained_state, method)
        print(f"trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

        result = train_loop(
            model, train_b, val_b, BLOCK_SIZE, BATCH_SIZE,
            steps=ADAPT_STEPS, lr=LR, eval_every=ADAPT_EVAL_EVERY,
            eval_iters=EVAL_ITERS, log_prefix=f"[{method}] ",
        )
        val_loss_b = estimate_loss(model, val_b, BLOCK_SIZE, BATCH_SIZE, FINAL_EVAL_ITERS)
        val_loss_a_after = estimate_loss(model, val_a, BLOCK_SIZE, BATCH_SIZE, FINAL_EVAL_ITERS)
        sample_b = generate_sample(model, tokenizer, prompt="En un lugar de la Mancha")

        val_loss_a_adapter_off = None
        if method.startswith("lora"):
            disable_all_adapters(model, enabled=False)
            val_loss_a_adapter_off = estimate_loss(model, val_a, BLOCK_SIZE, BATCH_SIZE, FINAL_EVAL_ITERS)
            disable_all_adapters(model, enabled=True)

        print(f"final val loss on B (Quijote): {val_loss_b:.4f}")
        print(f"val loss on A (Shakespeare) after adapting to B: {val_loss_a_after:.4f} "
              f"(pretrain was {pretrain_val_a:.4f})")
        if val_loss_a_adapter_off is not None:
            print(f"val loss on A with LoRA adapters switched off: {val_loss_a_adapter_off:.4f}")

        method_results.append({
            "method": method,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "history": result.history,
            "val_loss_b_final": round(val_loss_b, 4),
            "val_loss_a_after_adapt": round(val_loss_a_after, 4),
            "val_loss_a_adapter_off": round(val_loss_a_adapter_off, 4) if val_loss_a_adapter_off is not None else None,
            "wall_seconds": round(result.wall_seconds, 1),
            "sample": sample_b,
        })

    total_wall = time.time() - experiment_t0
    results = {
        "config": {
            "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
            "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE, "lr": LR,
            "pretrain_steps": PRETRAIN_STEPS, "adapt_steps": ADAPT_STEPS,
            "vocab_size": tokenizer.vocab_size, "total_params": total_params,
        },
        "pretrain": {
            "val_loss_a": round(pretrain_val_a, 4),
            "history": pretrain_result.history,
            "wall_seconds": round(pretrain_result.wall_seconds, 1),
            "sample": pretrain_sample,
        },
        "methods": method_results,
        "total_wall_seconds": round(total_wall, 1),
    }
    (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
    print(f"\ntotal experiment time: {total_wall / 60:.1f} minutes")
    print(f"wrote {RESULTS_DIR / 'results.json'}")

    print("\n| method | trainable params | % of total | val loss B (Quijote) | val loss A after adapt | val loss A, adapter off |")
    print("| --- | --- | --- | --- | --- | --- |")
    print(f"| pretrained (no adapt) | 0 | 0.00% | -- | {pretrain_val_a:.4f} | -- |")
    for m in method_results:
        off = f"{m['val_loss_a_adapter_off']:.4f}" if m["val_loss_a_adapter_off"] is not None else "--"
        print(f"| {m['method']} | {m['trainable_params']:,} | {m['trainable_pct']:.2f}% | "
              f"{m['val_loss_b_final']:.4f} | {m['val_loss_a_after_adapt']:.4f} | {off} |")


if __name__ == "__main__":
    main()
