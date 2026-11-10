"""
mt_lora_learnable_M_demo.py
===========================
Fine-tune RoBERTa-base on a GLUE task using MTLoRA with learnable orthogonal M
(Cayley parametrisation).

Usage
-----
python mt_lora_learnable_M_demo.py \
    --task cola \
    --rank 8 \
    --n_faces 12 \
    --alpha 4.0 \
    --epochs 10 \
    --lr 5e-4 \
    --lr_M 1e-3 \
    --freeze_M_epochs 2 \
    --proj query value \
    --seed 42

This script:
1. Loads RoBERTa-base (frozen except adapters + classifier head).
2. Wraps Q and V projections with MTLoRALearnableM adapters.
3. Trains with AdamW: separate learning rates for (Â, B̂) vs. Cayley S.
4. Logs per-epoch MCC (CoLA), F1 (MRPC, RTE), or accuracy (SST-2) + face energies.
5. Saves results to results/learnable_M_{task}_{proj}_{run_tag}.pt
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEVICE = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="cola", choices=["cola","mrpc","rte","sst2"])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--n_faces", type=int, default=None,
                   help="Kept transform-domain faces (default = L)")
    p.add_argument("--alpha", type=float, default=4.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="Learning rate for A_hat, B_hat (and classifier head)")
    p.add_argument("--lr_M", type=float, default=1e-3,
                   help="Learning rate for Cayley S_upper (usually larger than lr)")
    p.add_argument("--freeze_M_epochs", type=int, default=2,
                   help="Freeze Cayley M for first N epochs (warm-up on A,B only)")
    p.add_argument("--prune_threshold", type=float, default=0.0,
                   help="Prune faces with ||B_f||_F below this after each epoch "
                        "(0 = disabled). NOTE: a fixed 0.1 used to be hardcoded and "
                        "pruned every face on RTE at lr=1e-4 (B starts at 0 and grows slowly).")
    p.add_argument("--proj", nargs="+", default=["query","value"],
                   choices=["query","key","value","dense"])
    p.add_argument("--init_from_weights", action="store_true")
    p.add_argument("--a_init", default="normal", choices=["normal", "lora"],
                   help="'lora' = Hu et al. init (B~U(+-1/sqrt(d_in)), A=0); default N(0,1) A, B=0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--out_dir", default="results")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

TASK_TO_KEYS = {
    "cola":  ("sentence", None),
    "mrpc":  ("sentence1", "sentence2"),
    "rte":   ("sentence1", "sentence2"),
    "sst2":  ("sentence", None),
}

TASK_TO_NUM_LABELS = {
    "cola": 2, "mrpc": 2, "rte": 2, "sst2": 2,
}

def get_metric_fn(task: str):
    """Return a function (logits, labels) → scalar metric."""
    if task == "cola":
        from sklearn.metrics import matthews_corrcoef
        def metric(logits, labels):
            preds = logits.argmax(-1).numpy()
            return matthews_corrcoef(labels.numpy(), preds)
        return metric, "MCC"
    elif task in ("mrpc", "rte"):
        def metric(logits, labels):
            preds = logits.argmax(-1)
            return (preds == labels).float().mean().item()
        return metric, "Acc"
    else:  # sst2
        def metric(logits, labels):
            preds = logits.argmax(-1)
            return (preds == labels).float().mean().item()
        return metric, "Acc"


def tokenise_dataset(task: str, tokenizer, max_length: int):
    from datasets import load_dataset
    raw = load_dataset("nyu-mll/glue", task)
    key1, key2 = TASK_TO_KEYS[task]

    def encode(batch):
        if key2:
            return tokenizer(batch[key1], batch[key2],
                             truncation=True, max_length=max_length, padding="max_length")
        return tokenizer(batch[key1],
                         truncation=True, max_length=max_length, padding="max_length")

    encoded = raw.map(encode, batched=True)
    encoded.set_format("torch", columns=["input_ids","attention_mask","label"])
    return encoded["train"], encoded["validation"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def run(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)

    # --- model ---
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=TASK_TO_NUM_LABELS[args.task]
    )

    # Freeze all base parameters
    for param in model.parameters():
        param.requires_grad_(False)
    # Unfreeze classifier head
    for param in model.classifier.parameters():
        param.requires_grad_(True)

    # --- adapters ---
    from mt_lora_learnable_M import wrap_roberta_learnable_M, MTLoRALearnableM
    adapters: list[MTLoRALearnableM] = []
    for proj in args.proj:
        adapter = wrap_roberta_learnable_M(
            model,
            proj_name=proj,
            rank=args.rank,
            n_faces=args.n_faces,
            alpha=args.alpha,
            init_from_weights=args.init_from_weights,
            freeze_M_epochs=args.freeze_M_epochs,
            a_init=args.a_init,
        )
        adapters.append(adapter)

    model = model.to(DEVICE)

    # --- parameter groups: AB at lr, Cayley S at lr_M ---
    ab_params, m_params, head_params = [], [], []
    seen = set()
    for adapter in adapters:
        for name, param in adapter.named_parameters():
            if param not in seen:
                seen.add(param)
                if "S_upper" in name:
                    m_params.append(param)
                else:
                    ab_params.append(param)
    for name, param in model.classifier.named_parameters():
        if param not in seen:
            seen.add(param)
            head_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": ab_params + head_params, "lr": args.lr, "weight_decay": 0.01},
        {"params": m_params,                "lr": args.lr_M, "weight_decay": 0.0},
    ])

    # --- data ---
    train_ds, val_ds = tokenise_dataset(args.task, tokenizer, args.max_length)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=False)
    val_dl = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                        num_workers=0)

    # LR schedule: linear warmup 6% then cosine decay
    total_steps = args.epochs * len(train_dl)
    warmup_steps = max(1, int(0.06 * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + __import__("math").cos(3.14159265 * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    metric_fn, metric_name = get_metric_fn(args.task)
    loss_fn = nn.CrossEntropyLoss()
    results: dict = {
        "task": args.task, "rank": args.rank, "n_faces": args.n_faces,
        "alpha": args.alpha, "lr": args.lr, "lr_M": args.lr_M,
        "freeze_M_epochs": args.freeze_M_epochs,
        "proj": args.proj, "seed": args.seed, "a_init": args.a_init,
        "epochs": [], "train_loss": [], "val_metric": [],
        "face_energies": [], "M_off_diag_norm": [],
    }

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0

        for batch in train_dl:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(out.logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0
            )
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            global_step += 1

        # Epoch-end: advance adapter epoch counters (may unfreeze M)
        for adapter in adapters:
            adapter.step_epoch()

        # Prune dead faces (disabled by default)
        if args.prune_threshold > 0:
            for adapter in adapters:
                n_pruned = adapter.prune_dead_faces(threshold=args.prune_threshold)
                if n_pruned > 0:
                    print(f"  [pruned] {n_pruned} faces in {adapter}")

        # Validation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in val_dl:
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                all_logits.append(out.logits.cpu())
                all_labels.append(batch["label"].cpu())
        logits_cat = torch.cat(all_logits, 0)
        labels_cat = torch.cat(all_labels, 0)
        val_m = metric_fn(logits_cat, labels_cat)

        # Face energies & M rotation
        face_e_all = []
        M_off_norms = []
        for adapter in adapters:
            fe = [float(adapter.B_hat[:, :, f].detach().norm(p="fro"))
                  for f in range(adapter.n_faces)]
            face_e_all.append(fe)
            # How far has M rotated from identity?
            M_cur, _ = adapter.get_M()
            off_diag = M_cur.detach() - torch.eye(adapter.L, device=M_cur.device)
            M_off_norms.append(float(off_diag.norm(p="fro")))

        avg_loss = epoch_loss / len(train_dl)
        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:2d}/{args.epochs} | loss={avg_loss:.4f} | "
            f"{metric_name}={val_m:.4f} | "
            f"M_rot={M_off_norms[0]:.4f} | "
            f"faces_active={adapters[0].active_face_count()}/{adapters[0].n_faces} | "
            f"{elapsed:.1f}s"
        )

        results["epochs"].append(epoch)
        results["train_loss"].append(avg_loss)
        results["val_metric"].append(float(val_m))
        results["face_energies"].append(face_e_all)
        results["M_off_diag_norm"].append(M_off_norms)

    # --- save ---
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.task}_r{args.rank}_nf{args.n_faces or 'all'}_fm{args.freeze_M_epochs}_seed{args.seed}"
    if args.a_init != "normal":
        tag = f"{args.task}_r{args.rank}_nf{args.n_faces or 'all'}_fm{args.freeze_M_epochs}_{args.a_init}_seed{args.seed}"
    out_path = os.path.join(args.out_dir, f"learnable_M_{tag}.pt")
    save_data = {
        "results": results,
        "adapter_state": [a.state_dict() for a in adapters],
    }
    torch.save(save_data, out_path)
    print(f"\nSaved to {out_path}")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run(args)
