# LoRA Transformer Adaptation Study

A small, inspectable PyTorch experiment on low-rank adaptation of a character-level Transformer. The upstream project compares full fine-tuning with several LoRA ranks while measuring adaptation and forgetting.

## Model and experiment

The upstream model has two layers, two heads, 128-dimensional embeddings, and 438,016 parameters. It is pretrained on Shakespeare and adapted to Quijote. LoRA adds trainable low-rank updates while leaving the base weights frozen; disabling the adapter allows direct inspection of the original behavior.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/lora/` | Transformer and LoRA implementation |
| `scripts/` | Experiment entry points |
| `results/` | Recorded upstream metrics and generated samples |
| `notebook.ipynb` | Exploration |
| `tests/` | Existing checks |
| `pyproject.toml` | Project configuration |

## Upstream reported results

| Method | Trainable parameters | Quijote validation loss |
| --- | ---: | ---: |
| Full fine-tuning | 438,016 | 1.7567 |
| LoRA rank 4 | 6,144 | 2.3996 |
| LoRA rank 16 | 24,576 | 2.2862 |

The original experiment includes other ranks, validation curves, and generated text in `results/`. These values come from the [upstream README](UPSTREAM_README.md); no model was retrained for this fork.

## Source and license

Based on and adapted from [delcenjo/lora-from-scratch](https://github.com/delcenjo/lora-from-scratch). The original documentation, result files, tests, and [MIT license](LICENSE) are retained.