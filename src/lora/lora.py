"""LoRA (Low-Rank Adaptation), implemented from the equations in the paper.

Hu, Edward J., et al. "LoRA: Low-Rank Adaptation of Large Language Models."
arXiv:2106.09685 (2021).

The paper's starting observation: when you fine-tune a pretrained weight
matrix W0, the update you learn, delta_W, tends to have a low "intrinsic
rank" -- most of what fine-tuning changes can be captured by a much smaller
matrix than W0 itself. So instead of learning a full delta_W with the same
shape as W0 (d x k parameters), LoRA factorizes it as the product of two
skinny matrices:

    delta_W = B @ A,      B in R^(d x r),  A in R^(r x k),  r << min(d, k)

and freezes W0 completely. The adapted layer computes

    h = W0 x + (alpha / r) * B A x

instead of h = W x directly. Only A and B are trained; W0 never receives a
gradient. This is the whole idea -- everything below is bookkeeping around
that one equation: how A and B are shaped and initialized, how the forward
pass combines them with the frozen base, and how to fold them back into W0
when you want a plain dense layer again.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A frozen nn.Linear plus a trainable low-rank side path.

    Wraps an existing nn.Linear (called `base` below, playing the role of
    W0 in the paper) and adds two small matrices, `lora_A` and `lora_B`,
    whose product is the low-rank update delta_W = lora_B @ lora_A.

    Shapes, matching torch.nn.Linear's convention where weight has shape
    (out_features, in_features) and y = x @ weight.T + bias:

        base.weight : (out_features, in_features)      -- this is W0
        lora_A      : (r, in_features)                  -- this is A
        lora_B      : (out_features, r)                 -- this is B

    so that lora_B @ lora_A has the same shape as base.weight, exactly as
    delta_W = B A has the same shape as W0 in the paper. Because a row
    vector x is used here instead of the paper's column vector, the forward
    pass computes the update as (x @ A.T) @ B.T rather than B @ A @ x, but
    it is the same product, just transposed to match how nn.Linear applies
    its weight.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if r <= 0:
            raise ValueError("r must be positive")

        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        # The paper introduces alpha as a constant and scales the update by
        # alpha / r. Practically this means: when you change r, you can
        # leave alpha (and the learning rate) alone and the *magnitude* of
        # the update at initialization time stays comparable, because the
        # 1/r factor compensates for r extra terms being summed inside B A.
        # Defaulting alpha to r makes the scaling factor 1 unless the
        # caller asks for something else.
        self.alpha = alpha if alpha is not None else r
        self.scaling = self.alpha / self.r

        # --- W0: frozen base weight -------------------------------------
        # This is the pretrained weight matrix. LoRA's entire premise is
        # that it stays exactly as it was pretrained; only the low-rank
        # side path below is trained. requires_grad_(False) is what makes
        # optimizer.step() a no-op for these tensors even though they are
        # still part of the module (and still needed for every forward
        # pass -- freezing is not the same as removing).
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        # --- A and B: the trainable low-rank factors ---------------------
        # A is initialized with small random values and B with all zeros.
        # This is not a minor detail, it is the reason LoRA can be dropped
        # into a pretrained model safely: at step 0, delta_W = B @ A = 0
        # regardless of what A contains, because B is zero. The adapted
        # model therefore starts as an exact copy of the pretrained model
        # (h = W0 x + 0), and training moves it away from that point
        # gradually as B picks up a nonzero gradient. If both A and B were
        # initialized randomly, delta_W would be a random matrix from the
        # first step onward and training would begin by un-learning noise
        # injected into a model that used to work.
        #
        # A still needs *some* randomness, otherwise the two factors would
        # start out degenerate: if every row of A were identical (e.g. all
        # zeros, or all the same constant), every row of the gradient
        # w.r.t. B would also be identical during early training (since
        # dL/dB flows through A), collapsing the r update directions into
        # one instead of r independent ones. Kaiming-uniform (the same
        # family of init nn.Linear itself uses) gives A a spread of small
        # values scaled by 1/sqrt(in_features), which is the standard
        # "small random init" the paper refers to as A ~ N(0, sigma^2) --
        # the exact distribution matters much less than "small, and not
        # degenerate."
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Two independent on/off switches, used for different purposes:
        #
        # `merged`  -- whether delta_W has been physically added into
        #              base.weight.data (see merge()/unmerge() below). Once
        #              merged, the forward pass must NOT add the lora term
        #              again, or the update would be applied twice.
        # `enabled` -- a runtime toggle that lets you turn the adapter off
        #              *without* touching base.weight at all. Because W0 is
        #              never mutated outside of merge(), disabling the
        #              adapter (enabled=False, merged=False) makes the
        #              module compute exactly h = W0 x again -- bit for
        #              bit the pretrained layer, as if LoRA had never been
        #              attached. This is what makes it possible to check,
        #              after adapting to a new domain, whether the
        #              original model is still intact underneath: turn the
        #              adapters off and the frozen base is untouched.
        self.merged = False
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h = W0 x  (+ bias). This runs unconditionally: whether or not the
        # adapter is active, the frozen base layer always contributes its
        # output. If the update is already merged into base.weight, this
        # single call already includes it and nothing else needs to happen.
        base_out = F.linear(x, self.base.weight, self.base.bias)

        if self.merged or not self.enabled:
            return base_out

        # h = W0 x + (alpha / r) * B A x
        # Computed as (x @ A.T) @ B.T to match nn.Linear's row-vector
        # convention; algebraically this is the transpose of B A x.
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return base_out + lora_out * self.scaling

    @torch.no_grad()
    def merge(self) -> None:
        """Fold the low-rank update into the base weight: W' = W0 + (alpha/r) B A.

        After this call, base.weight *is* the adapted weight, and forward()
        stops adding the lora term (it would otherwise be double-counted).
        This is the "no extra inference latency" trick from the paper: a
        merged LoRALinear does exactly one matmul, identical in cost to the
        plain nn.Linear it replaced, versus the unmerged version which does
        an extra pair of small matmuls through rank r on every call.
        """
        if self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data += delta_w
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo merge(): subtract the low-rank update back out of base.weight."""
        if not self.merged:
            return
        delta_w = (self.lora_B @ self.lora_A) * self.scaling
        self.base.weight.data -= delta_w
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.4g}"
        )


def apply_lora(
    model: nn.Module,
    target_suffixes: tuple[str, ...],
    r: int,
    alpha: int | None = None,
    dropout: float = 0.0,
) -> list[str]:
    """Replace every nn.Linear whose qualified name ends with one of
    `target_suffixes` with a LoRALinear wrapping it in place.

    The paper studies adapting different subsets of the attention
    projections (query only, query+value, all four, ...) and finds that
    spreading a fixed parameter budget across more matrices at a smaller
    rank beats concentrating it in one matrix at a large rank. This
    experiment applies LoRA to every linear layer inside attention (the
    fused qkv projection and the output projection), which is close to the
    paper's "all projections" setting, adapted to this model's fused-qkv
    layout instead of four separate q/k/v/o matrices.

    Returns the list of module names that were wrapped, mostly so the
    caller can sanity-check that it actually found something.
    """
    wrapped = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(suffix) for suffix in target_suffixes):
            continue
        parent, attr = _resolve_parent(model, name)
        setattr(parent, attr, LoRALinear(module, r=r, alpha=alpha, dropout=dropout))
        wrapped.append(name)
    return wrapped


def _resolve_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def mark_only_lora_as_trainable(model: nn.Module) -> None:
    """Freeze everything except lora_A / lora_B parameters.

    Wrapping a layer in LoRALinear already freezes that layer's own W0 and
    bias (done in LoRALinear.__init__), but a full model has plenty of
    other parameters LoRA never touches: token and position embeddings,
    layer norms, the untouched feed-forward layers, the output head. This
    walks every parameter in the model and freezes anything that is not
    part of a lora_A/lora_B pair, so a single call after apply_lora() is
    enough to guarantee the optimizer only ever updates the adapters.
    """
    for name, param in model.named_parameters():
        param.requires_grad_("lora_A" in name or "lora_B" in name)


def trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable_count, total_count) over all parameters in model."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
