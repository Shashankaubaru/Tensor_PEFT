"""
Fine-tune RoBERTa on a GLUE task and save the stacked weight updates
Delta W = W_finetuned - W_pretrained for the attention projections.

Output: a dict {'q','k','v','o': tensor (d_out, d_in, L), 'meta': {...}} saved with torch.save.
Each tensor stacks the per-layer updates along the third (depth) mode, i.e. the
tensor that LoTR / TensLoRA ("Depth") / M-product LoRA would parameterise.

Full fine-tuning is used on purpose: with LoRA the update is rank-r by construction,
which would bias any low-rank analysis of Delta W.

Example:
    python delta_w_extract.py --task cola --epochs 3 --out results/dw_cola.pt
"""
import argparse
import os
import re

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

TASKS = {
    # task: (text key 1, text key 2, metric)
    "cola": ("sentence", None, "mcc"),
    "sst2": ("sentence", None, "acc"),
    "mrpc": ("sentence1", "sentence2", "acc"),
    "rte": ("sentence1", "sentence2", "acc"),
}

PROJ_PATTERNS = {
    "q": r"encoder\.layer\.(\d+)\.attention\.self\.query\.weight$",
    "k": r"encoder\.layer\.(\d+)\.attention\.self\.key\.weight$",
    "v": r"encoder\.layer\.(\d+)\.attention\.self\.value\.weight$",
    "o": r"encoder\.layer\.(\d+)\.attention\.output\.dense\.weight$",
}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def snapshot_projections(model):
    """Return {proj: {layer_idx: weight clone (cpu, float32)}} for the attention projections."""
    snap = {p: {} for p in PROJ_PATTERNS}
    for name, param in model.named_parameters():
        for proj, pat in PROJ_PATTERNS.items():
            m = re.search(pat, name)
            if m:
                snap[proj][int(m.group(1))] = param.detach().cpu().clone().float()
    return snap


def stack_delta(pre, post):
    out = {}
    for proj in pre:
        layers = sorted(pre[proj])
        out[proj] = torch.stack([post[proj][i] - pre[proj][i] for i in layers], dim=2)
    return out


def make_loader(ds, tok, k1, k2, batch_size, shuffle, max_len, device_cpu=True):
    def encode(batch):
        args = (batch[k1],) if k2 is None else (batch[k1], batch[k2])
        return tok(*args, truncation=True, max_length=max_len)

    ds = ds.map(encode, batched=True)
    cols = ["input_ids", "attention_mask", "label"]
    ds = ds.with_format("python", columns=cols)

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

    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def evaluate(model, loader, device, metric):
    model.eval()
    preds, gold = [], []
    for ids, att, y in loader:
        logits = model(input_ids=ids.to(device), attention_mask=att.to(device)).logits
        preds.extend(logits.argmax(-1).cpu().tolist())
        gold.extend(y.tolist())
    return matthews_corrcoef(gold, preds) if metric == "mcc" else accuracy_score(gold, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="cola", choices=sorted(TASKS))
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_train_samples", type=int, default=None, help="subsample for quick runs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = args.out or f"results/dw_{args.task}_seed{args.seed}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    k1, k2, metric = TASKS[args.task]
    device = pick_device()
    print(f"device={device} task={args.task} model={args.model}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    pre = snapshot_projections(model)

    raw = load_dataset("nyu-mll/glue", args.task)
    train_ds = raw["train"].shuffle(seed=args.seed)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    train_dl = make_loader(train_ds, tok, k1, k2, args.batch_size, True, args.max_len)
    val_dl = make_loader(raw["validation"], tok, k1, k2, 64, False, args.max_len)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    for ep in range(args.epochs):
        model.train()
        running = 0.0
        for step, (ids, att, y) in enumerate(train_dl, 1):
            loss = model(input_ids=ids.to(device), attention_mask=att.to(device), labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            running += loss.item()
        score = evaluate(model, val_dl, device, metric)
        print(f"epoch {ep + 1}: train_loss={running / len(train_dl):.4f} val_{metric}={score:.4f}")

    delta = stack_delta(pre, snapshot_projections(model))
    delta["meta"] = {"task": args.task, "model": args.model, "epochs": args.epochs, "lr": args.lr,
                     "seed": args.seed, f"val_{metric}": float(score)}
    torch.save(delta, out)
    print("saved", out, {k: tuple(v.shape) for k, v in delta.items() if k != "meta"})


if __name__ == "__main__":
    main()
