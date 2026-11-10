"""
MTLoRA fine-tuning demo
=======================
Fine-tune RoBERTa-base on a GLUE task using MTLoRA adapters and compare against
plain LoRA (which is the special case transform='identity', n_faces=L, rank=r).

Usage examples
--------------
# MTLoRA with DCT transform (init from W₀, 3 kept faces):
python mt_lora_demo.py --task cola --transform dct --n_faces 3 --rank 8 --init_from_weights

# MTLoRA with identity transform = effectively independent per-layer adapters:
python mt_lora_demo.py --task cola --transform identity --n_faces 12 --rank 8

# Both projections (q + v):
python mt_lora_demo.py --task mrpc --projs query value --rank 8 --epochs 5

# Enable dynamic rank pruning during training:
python mt_lora_demo.py --task cola --rank 8 --prune_threshold 1e-3

All pretrained parameters are frozen; only MTLoRA adapter parameters are trained.
"""
import argparse
import os
import re

import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, matthews_corrcoef
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from mt_lora import MTLoRA, wrap_roberta_attention_fast

TASKS = {
    "cola":  ("sentence",  None,       "mcc"),
    "sst2":  ("sentence",  None,       "acc"),
    "mrpc":  ("sentence1", "sentence2","acc"),
    "rte":   ("sentence1", "sentence2","acc"),
    "qnli":  ("question",  "sentence", "acc"),
    "mnli":  ("premise",   "hypothesis","acc"),
}
NUM_LABELS = {"mnli": 3}   # every other task is binary


# ---------------------------------------------------------------------------
# Data helpers (re-used from delta_w_extract.py logic)
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


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if os.environ.get("REQUIRE_CUDA") == "1":
        # a GPU job that landed on a CPU-only node would otherwise run until the wall limit
        raise SystemExit("REQUIRE_CUDA=1 but no CUDA device is available; resubmit this job")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task",    default="cola", choices=sorted(TASKS))
    ap.add_argument("--model",   default="roberta-base")
    ap.add_argument("--projs",   nargs="+", default=["query", "value"],
                    help="Attention projections to adapt. Must match HuggingFace layer names.")
    ap.add_argument("--rank",    type=int, default=8)
    ap.add_argument("--n_faces", type=int, default=None,
                    help="Number of transform-domain faces to keep (≤ L). Default: all L.")
    ap.add_argument("--transform", default="dct", choices=["dct", "random", "identity", "cayley"],
                    help="'cayley' = learnable orthogonal M (Cayley map), see --m_init / --lr_M.")
    ap.add_argument("--m_init", default="identity", choices=["identity", "dct"],
                    help="Cayley only: start M at the identity or at the DCT matrix.")
    ap.add_argument("--lr_M", type=float, default=1e-3, help="Cayley only: learning rate of S.")
    ap.add_argument("--freeze_M_epochs", type=int, default=2,
                    help="Cayley only: keep M fixed for the first N epochs.")
    ap.add_argument("--alpha",   type=float, default=1.0)
    ap.add_argument("--init_from_weights", action="store_true",
                    help="Initialise Â from tSVD-M(W₀). B̂ is always zero at start.")
    ap.add_argument("--a_init", default="normal", choices=["normal", "lora"],
                    help="Random init: 'normal' = Â~N(0,1), B̂=0 (paper default); "
                         "'lora' = Hu et al. LoRA init (B̂ Kaiming-uniform, Â=0).")
    ap.add_argument("--epochs",  type=int, default=5)
    ap.add_argument("--lr",      type=float, default=1e-3,
                    help="Learning rate. MTLoRA needs ~10× higher LR than full fine-tune.")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--max_train_samples", type=int, default=None)
    ap.add_argument("--prune_threshold", type=float, default=0.0,
                    help="Prune faces with B-norm below this after each epoch (0 = disabled).")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--out",     default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    out = args.out or f"results/mtlora_{args.task}_{args.transform}_r{args.rank}_seed{args.seed}.pt"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    k1, k2, metric = TASKS[args.task]
    device = pick_device()
    print(f"device={device}  task={args.task}  transform={args.transform}  "
          f"rank={args.rank}  n_faces={args.n_faces}  init_from_W0={args.init_from_weights}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=NUM_LABELS.get(args.task, 2))

    # -- Freeze all model parameters --
    for p in model.parameters():
        p.requires_grad_(False)

    # -- Attach MTLoRA adapters --
    adapters: list = []
    for proj in args.projs:
        if args.transform == "cayley":
            from mt_lora_learnable_M import wrap_roberta_learnable_M
            adapter = wrap_roberta_learnable_M(
                model, proj_name=proj, rank=args.rank, n_faces=args.n_faces, alpha=args.alpha,
                init_from_weights=args.init_from_weights, freeze_M_epochs=args.freeze_M_epochs,
                a_init=args.a_init, m_init=args.m_init,
            )
        else:
            adapter = wrap_roberta_attention_fast(
                model, proj_name=proj, rank=args.rank, n_faces=args.n_faces,
                transform=args.transform, alpha=args.alpha,
                init_from_weights=args.init_from_weights, a_init=args.a_init,
            )
        adapters.append(adapter)

    # -- Always unfreeze the classifier head --
    for p in model.classifier.parameters():
        p.requires_grad_(True)

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,}  ({100*trainable/total:.3f}%)")

    # -- Data --
    raw = load_dataset("nyu-mll/glue", args.task)
    train_ds = raw["train"].shuffle(seed=args.seed)
    if args.max_train_samples:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    train_dl = make_loader(train_ds, tok, k1, k2, args.batch_size, True,  args.max_len)
    val_split = "validation_matched" if args.task == "mnli" else "validation"
    val_dl   = make_loader(raw[val_split], tok, k1, k2, 64, False, args.max_len)

    # -- Optimiser and scheduler --
    # S (Cayley) is frozen at construction for the first epochs, so it must be put in its own
    # parameter group explicitly: building the list from requires_grad would drop it for good.
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
                adapter.step_epoch()          # may unfreeze Cayley S
        score = evaluate(model, val_dl, device, metric)
        avg_loss = running / len(train_dl)
        m_rot = None
        if hasattr(adapters[0], "M_init"):    # distance of the learned M from its starting point
            with torch.no_grad():
                m_rot = float((adapters[0].get_M()[0].cpu() - adapters[0].M_init.cpu()).norm())

        # Dynamic rank pruning
        n_pruned_total = 0
        if args.prune_threshold > 0:
            for adapter in adapters:
                n_pruned_total += adapter.prune_dead_faces(args.prune_threshold)

        # Face energy report
        face_info = " | ".join(
            f"{args.projs[i]} active={adapters[i].active_face_count()}/{adapters[i].n_faces} "
            f"energies=[{', '.join(f'{e:.3f}' for e in adapters[i].face_energy.tolist())}]"
            for i in range(len(adapters))
        )
        print(f"epoch {ep+1:2d}:  loss={avg_loss:.4f}  val_{metric}={score:.4f}  "
              f"pruned={n_pruned_total}  {face_info}")
        if m_rot is not None:
            print(f"          M moved {m_rot:.4f} (Frobenius) from its initial value")
        history.append(dict(epoch=ep+1, loss=avg_loss, score=float(score), n_pruned=n_pruned_total,
                            m_rot=m_rot))

    result = dict(
        task=args.task, transform=args.transform, rank=args.rank,
        model=args.model, n_layers=adapters[0].L,
        n_faces=args.n_faces, init_from_weights=args.init_from_weights,
        a_init=args.a_init, m_init=args.m_init if args.transform == "cayley" else None,
        lr_M=args.lr_M if args.transform == "cayley" else None,
        freeze_M_epochs=args.freeze_M_epochs if args.transform == "cayley" else None,
        alpha=args.alpha, epochs=args.epochs, lr=args.lr,
        trainable_params=trainable, history=history,
        final_score=float(score),
    )
    torch.save(result, out)
    print(f"\nFinal val_{metric}: {score:.4f}   saved to {out}")


if __name__ == "__main__":
    main()
