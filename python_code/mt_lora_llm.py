"""
mt_lora_llm.py
==============
MTLoRA / standard LoRA on a decoder-only LLM (Llama / Qwen2 / Mistral naming) for GLUE
classification. Used for the "deeper model" experiments: the encoder-only RoBERTa-base has
L = 12 layers; Qwen2.5-1.5B and -7B have L = 28.

Standard LoRA is the special case  --transform identity --n_faces L  (per-layer adapters),
exactly as in mt_lora_demo.py. The same MTLoRA module and the same fast forward are used;
only the layer-name pattern, dtype handling and model loading differ.

Precision: the frozen base model is loaded in bfloat16; the adapters and the classification head
are kept in float32 (inputs are cast to float32 inside the adapter and the delta is cast back).

Usage
-----
python mt_lora_llm.py --model Qwen/Qwen2.5-1.5B --task cola --transform dct --n_faces 28 \
    --rank 8 --alpha 4 --a_init lora --epochs 3 --lr 2e-4 --seed 0 --out results/x.pt
"""
import argparse
import os
import re

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from mt_lora import MTLoRA
from mt_lora_demo import NUM_LABELS, TASKS, evaluate, make_loader, pick_device


class LLMLinearFast(nn.Module):
    """Frozen (bf16) linear + MTLoRA delta computed in float32 and cast back."""

    def __init__(self, linear: nn.Linear, adapter: MTLoRA, layer_idx: int) -> None:
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        a = self.adapter
        Mt = a.get_M()[1] if hasattr(a, "cayley") else a.Mt     # learnable M is recomputed each call
        c = Mt[self.layer_idx, : a.n_faces] * a.face_mask.to(Mt.dtype)
        xf = x.to(a.A_hat.dtype)
        Bx = torch.einsum("...i,rip->...rp", xf, a.B_hat)
        ABx = torch.einsum("orp,...rp->...op", a.A_hat, Bx)
        delta = torch.einsum("...op,p->...o", ABx, c * a.scaling)
        return base + delta.to(base.dtype)


def wrap_llm_attention(model, proj_name, rank, n_faces, transform, alpha, a_init,
                       m_init="identity", freeze_M_epochs=1):
    pattern = re.compile(rf"\.layers\.(\d+)\.self_attn\.{proj_name}$")
    found = {}
    for name, module in model.named_modules():
        m = pattern.search(name)
        if m and isinstance(module, nn.Linear):
            found[int(m.group(1))] = (name, module)
    if not found:
        raise ValueError(f"No layers matched for proj_name={proj_name!r}")
    idxs = sorted(found)
    L = len(idxs)
    d_out, d_in = found[idxs[0]][1].weight.shape
    if transform == "cayley":
        from mt_lora_learnable_M import MTLoRALearnableM
        adapter = MTLoRALearnableM(d_out=d_out, d_in=d_in, L=L, rank=rank, n_faces=n_faces,
                                   alpha=alpha, freeze_M_epochs=freeze_M_epochs, a_init=a_init,
                                   m_init=m_init)
    else:
        adapter = MTLoRA(d_out=d_out, d_in=d_in, L=L, rank=rank, n_faces=n_faces,
                         transform=transform, alpha=alpha, a_init=a_init)
    for local_idx, layer_idx in enumerate(idxs):
        full_name, linear = found[layer_idx]
        parent = model
        parts = full_name.split(".")
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LLMLinearFast(linear, adapter, local_idx))
    print(f"[MTLoRA-llm] wrapped {L} '{proj_name}' layers ({d_out}x{d_in}), rank={rank}, "
          f"n_faces={adapter.n_faces}, transform={transform}, "
          f"adapter params={sum(p.numel() for p in adapter.parameters()):,}")
    return adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--task", default="cola", choices=sorted(TASKS))
    ap.add_argument("--projs", nargs="+", default=["q_proj", "v_proj"])
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--n_faces", type=int, default=None)
    ap.add_argument("--transform", default="dct", choices=["dct", "random", "identity", "cayley"])
    ap.add_argument("--m_init", default="identity", choices=["identity", "dct"],
                    help="Cayley only: start M at the identity or at the DCT matrix.")
    ap.add_argument("--lr_M", type=float, default=1e-3, help="Cayley only: learning rate of S.")
    ap.add_argument("--freeze_M_epochs", type=int, default=1,
                    help="Cayley only: keep M fixed for the first N epochs.")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--a_init", default="lora", choices=["normal", "lora"])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--grad_ckpt", action="store_true",
                    help="Gradient checkpointing (use for 7B-class models).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    k1, k2, metric = TASKS[args.task]
    device = pick_device()
    print(f"device={device} model={args.model} task={args.task} transform={args.transform} "
          f"rank={args.rank} n_faces={args.n_faces} lr={args.lr}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS.get(args.task, 2), torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id

    for p in model.parameters():
        p.requires_grad_(False)

    adapters = [wrap_llm_attention(model, proj, args.rank, args.n_faces, args.transform,
                                   args.alpha, args.a_init, args.m_init, args.freeze_M_epochs)
                for proj in args.projs]

    # Classification head ("score") in float32, trainable; cast its input up.
    model.score.float()
    for p in model.score.parameters():
        p.requires_grad_(True)
    model.score.register_forward_pre_hook(lambda m, a: (a[0].float(),))

    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False

    model = model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

    raw = load_dataset("nyu-mll/glue", args.task)
    train_ds = raw["train"].shuffle(seed=args.seed)
    val_split = "validation_matched" if args.task == "mnli" else "validation"
    train_dl = make_loader(train_ds, tok, k1, k2, args.batch_size, True, args.max_len)
    val_dl = make_loader(raw[val_split], tok, k1, k2, 64, False, args.max_len)

    # S (Cayley) starts frozen, so it gets its own parameter group explicitly (see mt_lora_demo.py).
    s_params = [p for a in adapters if hasattr(a, "cayley") for p in a.cayley.parameters()]
    s_ids = {id(p) for p in s_params}
    params = [p for p in model.parameters() if p.requires_grad and id(p) not in s_ids]
    groups = [dict(params=params, lr=args.lr, weight_decay=0.01)]
    if s_params:
        groups.append(dict(params=s_params, lr=args.lr_M, weight_decay=0.0))
    opt = torch.optim.AdamW(groups)
    params = params + s_params
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    history = []
    score = 0.0
    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for ids, att, y in train_dl:
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device),
                         labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
        for adapter in adapters:
            if hasattr(adapter, "step_epoch"):
                adapter.step_epoch()
        score = evaluate(model, val_dl, device, metric)
        avg_loss = running / len(train_dl)
        m_rot = None
        if hasattr(adapters[0], "M_init"):
            with torch.no_grad():
                m_rot = float((adapters[0].get_M()[0].cpu() - adapters[0].M_init.cpu()).norm())
        print(f"epoch {ep + 1:2d}:  loss={avg_loss:.4f}  val_{metric}={score:.4f}"
              + (f"  M moved {m_rot:.4f}" if m_rot is not None else ""))
        history.append(dict(epoch=ep + 1, loss=avg_loss, score=float(score), m_rot=m_rot))

    result = dict(
        task=args.task, model=args.model, transform=args.transform, rank=args.rank,
        n_layers=adapters[0].L, n_faces=args.n_faces, a_init=args.a_init, alpha=args.alpha,
        m_init=args.m_init if args.transform == "cayley" else None,
        lr_M=args.lr_M if args.transform == "cayley" else None,
        freeze_M_epochs=args.freeze_M_epochs if args.transform == "cayley" else None,
        epochs=args.epochs, lr=args.lr, seed=args.seed, trainable_params=trainable,
        history=history, final_score=float(score),
    )
    torch.save(result, args.out)
    print(f"\nFinal val_{metric}: {score:.4f}   saved to {args.out}")


if __name__ == "__main__":
    main()
