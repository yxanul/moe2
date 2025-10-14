# TODO: Integrate Muon Optimizer Alongside AdamW

This note explains what Muon changes compared to AdamW, exactly how to split
parameters in this repo, and the minimal drop‑in edits to `train.py` to run
Muon (for matrix params) together with AdamW (for embeddings, LM head, and
non‑matrix params).

## 1) Muon vs AdamW — What Changes

- AdamW: adaptive (per‑parameter) step via first/second moments, plus decoupled
  weight decay.
- Muon: SGD + momentum produces a proposed update Δ, which is then
  orthogonalized via a fast Newton–Schulz (NS) iteration on the last two dims
  (the “zeroth‑power” or polar‑factor step) before applying.
- Intended targets: ≥2‑D “matrix‑like” params (linear layers, MoE expert mats).
  Do not use for embeddings, LM head, biases, or 1‑D norm scales.
- Practical: often enables higher LR and improves stability; adds a few
  matmuls per step (NS iterations).

## 2) Param Mapping for This Repo

Eligible for Muon (≥2‑D, excluding tied embeddings/LM head):
- All attention and FFN linear weights:
  - `GroupedQueryAttention.qkv_proj.weight`, `GroupedQueryAttention.o_proj.weight`
  - (Optional) `GroupedQueryAttention.gate_proj.weight` when gating is enabled
  - Any other `nn.Linear.weight` inside the model
- MoE expert batched weights (3‑D) in `src/model/experts.py`:
  - `SwiGLUExpert.w_gate`, `w_up`, `w_down` (shape `[E, in, out]`)

Keep on AdamW (not Muon):
- Tied token embeddings and LM head (same tensor):
  - `model.embed.weight` and `model.lm_head.weight` (weight_decay=0.0)
- All 0‑D/1‑D params: biases, RMSNorm scale (`RMSNorm.weight`), gating biases
  (`gate_proj.bias`), etc. (weight_decay=0.0)

Notes:
- Because `lm_head.weight is model.embed.weight`, include this tensor only once
  (treat it as embeddings group, weight_decay=0.0).
- If you prefer AdamW for gating weights, exclude `gate_proj.weight` from Muon.

## 3) Minimal Muon Implementation (drop‑in)

Create `src/optim/muon.py` with:

```python
import torch
from torch import Tensor

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params = list(params)
        param_groups = []
        for size in {p.numel() for p in params}:
            group = dict(params=[p for p in params if p.numel() == size])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                buf = state.setdefault("momentum_buffer", torch.zeros_like(g))
                buf.lerp_(g, 1 - group["momentum"])  # momentum update
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])  # orthogonalize
                scale = max(1, p.size(-2) / p.size(-1)) ** 0.5
                p.add_(g, alpha=-group["lr"] * scale)
```

You may remove `@torch.compile` if your stack has compile issues.

## 4) Exact Parameter Filters

Add a param‑split helper (e.g., in `train.py` near `create_optimizer` or a new
`optim.py` helper):

```python
from typing import Dict, List, Tuple

def split_params_for_muon_and_adamw(model: torch.nn.Module) -> Dict[str, List[torch.nn.Parameter]]:
    embed_w = model.embed.weight
    lm_head_w = model.lm_head.weight

    muon_params: List[torch.nn.Parameter] = []
    embed_params: List[torch.nn.Parameter] = []
    adamw_other: List[torch.nn.Parameter] = []  # biases, norms, etc.
    seen = set()

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # Handle tied embeddings / lm_head once
        if p is embed_w or p is lm_head_w:
            if id(embed_w) not in seen:
                embed_params.append(embed_w)
                seen.add(id(embed_w))
            continue

        if p.dim() >= 2:
            muon_params.append(p)
        else:
            adamw_other.append(p)

    return {
        "muon": muon_params,
        "embed": embed_params,      # weight_decay = 0.0
        "adamw_other": adamw_other, # weight_decay = 0.0
    }
```

If you want to exclude some 2‑D weights (e.g., gating weights) from Muon, add a
name filter before appending to `muon_params`.

## 5) Replace `create_optimizer` → `create_optimizers`

In `train.py`, replace the single‑optimizer factory with a dual‑optimizer one:

```python
import bitsandbytes as bnb
from src.optim.muon import Muon

def create_optimizers(model: nn.Module, config: Dict):
    groups = split_params_for_muon_and_adamw(model)

    # AdamW for embeddings and all non‑matrix params
    adamw_groups = []
    if groups["embed"]:
        adamw_groups.append({"params": groups["embed"], "weight_decay": 0.0})
    if groups["adamw_other"]:
        adamw_groups.append({"params": groups["adamw_other"], "weight_decay": 0.0})

    adamw = bnb.optim.AdamW8bit(
        adamw_groups,
        lr=config["training"]["learning_rate"],
        betas=config["optimizer"]["betas"],
        eps=config["optimizer"]["eps"],
    )

    # Muon for all matrix‑like params
    muon_cfg = config.get("muon", {"lr": 0.02, "momentum": 0.95, "nesterov": True, "ns_steps": 5})
    muon = Muon(
        groups["muon"],
        lr=muon_cfg["lr"], momentum=muon_cfg["momentum"],
        nesterov=muon_cfg.get("nesterov", True), ns_steps=muon_cfg.get("ns_steps", 5),
    )

    return {"adamw": adamw, "muon": muon}
```

Update the call site where `create_optimizer` is used to `create_optimizers` and
store both optimizers and (optionally) a scheduler for AdamW only.

## 6) Training Loop Changes (accumulation boundary)

At the point where you currently clip, all‑reduce, and step (see `train.py`):

```python
# Before: scaler.unscale_(optimizer)  # single optimizer
# After: unscale AdamW, then clip once, step both

scaler.unscale_(optimizers["adamw"])           # if scaler is enabled (fp16)
torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

# All‑reduce grads (unchanged)
# ... your existing all‑reduce loop ...

# Step AdamW under scaler
if scaler.is_enabled():
    scaler.step(optimizers["adamw"])  # AdamW8bit participates in AMP
else:
    optimizers["adamw"].step()

# Step Muon directly (after unscale_)
optimizers["muon"].step()

# Scheduler for AdamW (and Muon if you add one)
scheduler.step()
if scaler.is_enabled():
    scaler.update()

# Zero both
optimizers["adamw"].zero_grad(set_to_none=True)
optimizers["muon"].zero_grad(set_to_none=True)
```

Checkpointing: add Muon state dict alongside AdamW’s (save/load):

```python
torch.save({
    'step': global_step,
    'model_state_dict': model.state_dict(),
    'adamw_state_dict': optimizers['adamw'].state_dict(),
    'muon_state_dict':  optimizers['muon'].state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'config': config,
}, checkpoint_path)

# When loading:
optimizers['adamw'].load_state_dict(ckpt['adamw_state_dict'])
optimizers['muon'].load_state_dict(ckpt['muon_state_dict'])
```

## 7) Config Additions (optional)

Extend your YAML with Muon hyperparams:

```yaml
muon:
  lr: 0.02
  momentum: 0.95
  nesterov: true
  ns_steps: 5
```

Leave your existing `optimizer:` block for AdamW unchanged. Consider using a
slightly higher LR for Muon than AdamW.

## 8) Tips & Pitfalls

- AMP: With fp16, ensure you `scaler.unscale_(adamw)` before clipping; Muon is
  stepped after unscale (it doesn’t use `scaler.step`). With bf16 (default),
  scaler is disabled and no changes are needed.
- Weight decay: Typical setup is no decay for Muon group; embeddings/LM head
  also no decay. Keep your no‑decay filters for biases/norms.
- MoE experts: The 3‑D expert weights are great candidates for Muon. NS runs on
  the last two dims and will orthogonalize per expert in batch.
- Gated attention: It’s fine to include `gate_proj.weight` in Muon; if you want
  to keep gates conservative early on, exclude it from Muon initially.
- Performance: If overhead is noticeable, reduce `ns_steps` to 3; compile helps.

---

Summary: Use AdamW for embeddings and 0/1‑D params, Muon for all other matrix
weights. Add the split helper, instantiate two optimizers, unscale/clip once,
step AdamW under AMP and Muon directly, and save both states in checkpoints.

