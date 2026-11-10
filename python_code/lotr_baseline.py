"""
lotr_baseline.py — LoTR-style Tucker-2 PEFT baseline
=====================================================
Implements LoTR (Low-rank Tensor Reparametrisation) for RoBERTa-base,
as described in Bershatsky et al. 2024 (https://arxiv.org/abs/2402.01376).

Tucker-2 decomposition of the stacked weight update ΔW ∈ R^{d_out × d_in × L}:
    ΔW_{ij,ℓ} = Σ_{p,q} G_{pq,ℓ} · A_{ip} · B_{jq}

where:
  A ∈ R^{d_out × r}   — shared left factor  (frozen after init by default)
  B ∈ R^{d_in  × r}   — shared right factor (frozen after init by default)
  G_ℓ ∈ R^{r × r}     — per-layer core      (always trainable)

At equal rank r, LoTR uses  r*(d_out + d_in) + L*r^2  parameters vs
LoRA's  L*r*(d_out + d_in).  LoTR is cheaper when L*r^2 ≪ (L-1)*r*(d+d).

Usage
-----
python lotr_baseline.py --task cola --rank 8 --epochs 10 --lr 5e-4 --seed 0
python lotr_baseline.py --task mrpc --rank 8 --epochs 5  --lr 5e-4 --seed 0
python lotr_baseline.py --task rte  --rank 8 --epochs 10 --lr 1e-4 --seed 0
python lotr_baseline.py --task sst2 --rank 8 --epochs 5  --lr 5e-4 --seed 0
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ---------------------------------------------------------------------------
# Task config
# ---------------------------------------------------------------------------

TASKS = {
    "cola":  ("sentence",  None,        "mcc"),
    "sst2":  ("sentence",  None,        "acc"),
    "mrpc":  ("sentence1", "sentence2", "acc"),
    "rte":   ("sentence1", "sentence2", "acc"),
    "qnli":  ("question",  "sentence",  "acc"),
    "mnli":  ("premise",   "hypothesis","acc"),
}

TASK_NUM_LABELS = {"cola": 2, "sst2": 2, "mrpc": 2, "rte": 2, "qnli": 2, "mnli": 3}


# ---------------------------------------------------------------------------
# LoTR adapter modules
# ---------------------------------------------------------------------------

class LoTRAdapter(nn.Module):
    """
    Shared Tucker-2 adapter for L layers of the same projection type.

    Parameters
    ----------
    d_out, d_in : int   — weight matrix dimensions
    L : int             — number of layers
    rank : int          — Tucker-2 rank r
    alpha : float       — LoRA-style scaling (applied as alpha/rank)
    train_AB : bool     — if True, A and B are also trainable (full Tucker-2);
                          if False, only cores G_ℓ are trained (LoTR default)
    """

    def __init__(
        self,
        d_out: int,
        d_in: int,
        L: int,
        rank: int = 8,
        alpha: float = 1.0,
        train_AB: bool = True,
    ) -> None:
        super().__init__()
        self.d_out = d_out
        self.d_in = d_in
        self.L = L
        self.rank = rank
        self.scaling = alpha / rank
        self.train_AB = train_AB

        # Shared spatial factors
        self.A = nn.Parameter(torch.empty(d_out, rank), requires_grad=train_AB)
        self.B = nn.Parameter(torch.empty(d_in,  rank), requires_grad=train_AB)

        # Per-layer cores  G ∈ R^{L × r × r}
        self.G = nn.Parameter(torch.zeros(L, rank, rank))

        self._init_params()

    def _init_params(self) -> None:
        # Kaiming for A, B; G stays at zero (→ ΔW=0 at init, same as LoRA)
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        nn.init.kaiming_uniform_(self.B, a=5 ** 0.5)
        nn.init.zeros_(self.G)

    def get_delta_W(self, layer_idx: int) -> torch.Tensor:
        """
        Compute  ΔW_ℓ = A @ G_ℓ @ B^T  (d_out × d_in)
        using the Tucker-2 formula:
            ΔW_{ij,ℓ} = Σ_{pq} G_{ℓ,pq} A_{ip} B_{jq}
                       = A @ G_ℓ @ B^T
        """
        return self.scaling * (self.A @ self.G[layer_idx] @ self.B.T)

    def parameter_count(self) -> dict:
        ab = (self.d_out + self.d_in) * self.rank if self.train_AB else 0
        cores = self.L * self.rank * self.rank
        return {"AB": ab, "cores": cores, "total": ab + cores}

    def extra_repr(self) -> str:
        pc = self.parameter_count()
        return (
            f"d_out={self.d_out}, d_in={self.d_in}, L={self.L}, rank={self.rank}, "
            f"train_AB={self.train_AB}, params={pc['total']:,}"
        )


class LoTRLinear(nn.Module):
    """Wraps one nn.Linear with a LoTR adapter."""

    def __init__(self, linear: nn.Linear, adapter: LoTRAdapter, layer_idx: int) -> None:
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dW = self.adapter.get_delta_W(self.layer_idx)       # (d_out, d_in)
        return self.linear(x) + x @ dW.T

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}"


# ---------------------------------------------------------------------------
# Model wrapping
# ---------------------------------------------------------------------------

def wrap_roberta_lotr(
    model: nn.Module,
    proj_name: str = "query",
    rank: int = 8,
    alpha: float = 1.0,
    train_AB: bool = True,
) -> LoTRAdapter:
    """
    Replace all `proj_name` projections in a RoBERTa encoder with LoTRLinear wrappers.
    Returns the shared LoTRAdapter.
    """
    import re
    pattern = re.compile(
        rf"encoder\.layer\.(\d+)\.attention\.(?:self|output)\.{proj_name}$"
    )
    found: dict[int, tuple[str, nn.Linear]] = {}
    for name, module in model.named_modules():
        m = pattern.search(name)
        if m and isinstance(module, nn.Linear):
            found[int(m.group(1))] = (name, module)

    if not found:
        raise ValueError(f"No layers matched proj_name='{proj_name}'")

    layers_sorted = sorted(found)
    L = len(layers_sorted)
    sample = found[layers_sorted[0]][1]
    d_out, d_in = sample.weight.shape

    adapter = LoTRAdapter(d_out=d_out, d_in=d_in, L=L, rank=rank, alpha=alpha,
                          train_AB=train_AB)

    for local_idx, layer_idx in enumerate(layers_sorted):
        full_name, linear = found[layer_idx]
        wrapper = LoTRLinear(linear, adapter, layer_idx=local_idx)
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    pc = adapter.parameter_count()
    print(
        f"[LoTR] wrapped {L} '{proj_name}' layers "
        f"(rank={rank}, train_AB={train_AB}). "
        f"Params: AB={pc['AB']:,}  cores={pc['cores']:,}  total={pc['total']:,}"
    )
    return adapter


# ---------------------------------------------------------------------------
# Data helpers (identical to mt_lora_demo.py)
# ---------------------------------------------------------------------------

def make_loader(ds, tok, k1, k2, batch_size, shuffle, max_len):
    def encode(batch):
        args = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        return tok(*args, truncation=True, max_length=max_len)
    ds = ds.map(encode, batched=True)
    ds = ds.with_format("python", columns=["input_ids", "attention_mask", "label"])

    def collate(items):
        n = max(len(x["input_ids"]) for x in items)
        ids = torch.full((len(items), n), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(items), n), dtype=torch.long)
        for i, x in enumerate(items):
            L = len(x["input_ids"])
            ids[i, :L] = torch.tensor(x["input_ids"])
            att[i, :L] = 1
        labels = torch.tensor([x["label"] for x in items], dtype=torch.long)
        return ids, att, labels

    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate
    )


@torch.no_grad()
def evaluate(model, loader, device, metric):
    model.eval()
    preds, gold = [], []
    for ids, att, y in loader:
        logits = model(
            input_ids=ids.to(device), attention_mask=att.to(device)
        ).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(y.tolist())
    return matthews_corrcoef(gold, preds) if metric == "mcc" else accuracy_score(gold, preds)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",       default="cola", choices=sorted(TASKS))
    ap.add_argument("--model",      default="roberta-base")
    ap.add_argument("--projs",      nargs="+", default=["query", "value"])
    ap.add_argument("--rank",       type=int,   default=8)
    ap.add_argument("--alpha",      type=float, default=1.0)
    ap.add_argument("--train_AB",   action="store_true", default=True,
                    help="Train shared A,B factors in addition to cores G.")
    ap.add_argument("--no_train_AB", dest="train_AB", action="store_false",
                    help="Freeze A,B; train only per-layer cores G (strict LoTR).")
    ap.add_argument("--epochs",     type=int,   default=10)
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int,   default=32)
    ap.add_argument("--max_len",    type=int,   default=128)
    ap.add_argument("--seed",       type=int,   default=0)
    ap.add_argument("--out",        default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    tag = f"lotr_{args.task}_r{args.rank}_trainAB{int(args.train_AB)}_seed{args.seed}"
    out = args.out or f"results/{tag}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    k1, k2, metric = TASKS[args.task]
    device = pick_device()
    print(f"device={device}  task={args.task}  rank={args.rank}  "
          f"train_AB={args.train_AB}  projs={args.projs}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=TASK_NUM_LABELS.get(args.task, 2)
    )

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Attach LoTR adapters
    adapters: list[LoTRAdapter] = []
    for proj in args.projs:
        adapter = wrap_roberta_lotr(
            model, proj_name=proj, rank=args.rank,
            alpha=args.alpha, train_AB=args.train_AB,
        )
        adapters.append(adapter)

    # Unfreeze classifier head
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")

    # Data
    raw = load_dataset("nyu-mll/glue", args.task)
    train_ds = raw["train"].shuffle(seed=args.seed)
    val_split = "validation_matched" if args.task == "mnli" else "validation"
    train_dl = make_loader(train_ds, tok, k1, k2, args.batch_size, True,  args.max_len)
    val_dl   = make_loader(raw[val_split], tok, k1, k2, 64, False, args.max_len)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    history = []
    score = 0.0
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for ids, att, y in train_dl:
            loss = model(
                input_ids=ids.to(device),
                attention_mask=att.to(device),
                labels=y.to(device),
            ).loss
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()

        score = evaluate(model, val_dl, device, metric)
        avg_loss = running / len(train_dl)
        print(f"epoch {ep+1:2d}:  loss={avg_loss:.4f}  val_{metric}={score:.4f}")
        history.append(dict(epoch=ep+1, loss=avg_loss, score=float(score)))

    result = dict(
        task=args.task, rank=args.rank, train_AB=args.train_AB,
        alpha=args.alpha, epochs=args.epochs, lr=args.lr,
        trainable_params=trainable, history=history, final_score=float(score),
    )
    torch.save(result, out)
    print(f"\nFinal val_{metric}: {score:.4f}   saved to {out}")


if __name__ == "__main__":
    main()
