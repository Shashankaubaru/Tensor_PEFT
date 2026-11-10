"""
collect_results.py — aggregate CCC experiment results into a summary table
==========================================================================
Reads every .pt result file under results/ (recursively; some CCC runs were
saved as results/<name>.pt/<file>.pt) and prints mean ± std across seeds for
each (method, task) pair at rank 8.

Design choices (so the table is reproducible and honest):
  * Run metadata (task, lr, seed, alpha, epochs, transform, n_faces) is read from
    the file CONTENTS, not parsed from the filename.
  * The reported score is the FINAL-epoch validation score (not the best epoch).
    Use --epoch N to report the score at epoch N instead.
  * Headline setting: lr = 1e-4 for RTE and 5e-4 otherwise, alpha = 4, rank 8,
    and 10 epochs for RTE / 5 epochs otherwise. Runs with a different alpha or lr
    are not used. Runs with a different epoch count form separate rows
    ("... (10 ep)") so they are never mixed into the headline rows.
  * Identity uses 12 faces (the 4-face DCT run is a separate ablation).
  * Several files for the same (method, task, seed) collapse to ONE run: the most
    recently modified file wins. Everything left out is listed (nothing is dropped
    silently).
  * Cayley runs in which every face was pruned (adapter dead; only the classifier
    head trained) are INVALID and excluded. These came from a hardcoded pruning
    threshold of 0.1 that has since been removed from mt_lora_learnable_M_demo.py.
  * Runs that look untrained (CoLA MCC ~ 0, or training loss that barely moved)
    are flagged under "Suspect runs" but still counted.

Usage
-----
python collect_results.py                 # table + per-run details
python collect_results.py --latex         # LaTeX tabular
python collect_results.py --epoch 5       # score at epoch 5 for all runs
python collect_results.py --out results/summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from collections import defaultdict
from glob import glob

import torch


TASK_ORDER = ["cola", "mrpc", "rte", "sst2"]
METHOD_ORDER = ["lotr", "lotr_r64", "lotr_r64_ep10", "lora", "lora_ep10",
                "mtlora_identity", "mtlora_identity_ep10", "cayley", "cayley_ep10",
                "cayley_lora", "cayley_lora_ep10"]
METHOD_LABEL = {
    "lotr":                 "LoTR (Tucker-2)",
    "lotr_r64":             "LoTR (Tucker-2, rank 64)",
    "lotr_r64_ep10":        "LoTR (Tucker-2, rank 64, 10 ep)",
    "lora":                 "LoRA (Hu et al. init)",
    "lora_ep10":            "LoRA (Hu et al. init, 10 ep)",
    "mtlora_identity":      r"$\star_M$-LoRA-Identity",
    "mtlora_identity_ep10": r"$\star_M$-LoRA-Identity (10 ep)",
    "cayley":               r"$\star_M$-LoRA-Cayley",
    "cayley_ep10":          r"$\star_M$-LoRA-Cayley (10 ep)",
    "cayley_lora":          r"$\star_M$-LoRA-Cayley, LoRA init",
    "cayley_lora_ep10":     r"$\star_M$-LoRA-Cayley, LoRA init (10 ep)",
}
HEADLINE_LR = {"rte": 1e-4}          # every other task: 5e-4
DEFAULT_LR = 5e-4
HEADLINE_ALPHA = 4.0
HEADLINE_SCALING = 0.5               # alpha / rank: 4/8 for rank 8, 32/64 for LoTR rank 64
HEADLINE_EPOCHS = {"rte": 10}        # every other task: 5
DEFAULT_EPOCHS = 5
HEADLINE_FACES = 12
# A run whose final score is at or below this is flagged as probably not trained
# (CoLA MCC 0 = constant prediction).
SUSPECT_MAX = {"cola": 0.05}


def read_run(path: str) -> dict | None:
    """Return a normalised record for one result file, or None if unusable."""
    try:
        r = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # corrupt / partial file
        print(f"  Warning: could not load {path}: {e}")
        return None

    if isinstance(r.get("results"), dict):             # mt_lora_learnable_M_demo.py
        i = r["results"]
        curve = [float(x) for x in i["val_metric"]]
        masks = [st["face_mask"] for st in r.get("adapter_state", []) if "face_mask" in st]
        dead = bool(masks) and all(not bool(m.any()) for m in masks)
        method = "cayley_lora" if i.get("a_init", "normal") == "lora" else "cayley"
        rec = dict(method=method, task=i["task"], lr=i["lr"], seed=i["seed"],
                   alpha=i["alpha"], rank=i["rank"], n_faces=i["n_faces"],
                   epochs=len(curve), curve=curve,
                   loss=[float(x) for x in i["train_loss"]],
                   m_rot=float(i["M_off_diag_norm"][-1][0]), all_faces_pruned=dead)
    elif "history" in r:                                # mt_lora_demo.py / lotr_baseline.py
        curve = [float(h["score"]) for h in r["history"]]
        if "transform" in r:
            # a_init='lora' (Hu et al. init) with M=I, p=L is the standard-LoRA baseline;
            # it must never be merged into the N(0,1)-initialised Identity rows.
            if r.get("a_init", "normal") == "lora":
                # only M=I is "standard LoRA"; DCT / random with the same init are separate rows
                method = "lora" if r["transform"] == "identity" else f"mtlora_{r['transform']}_lorainit"
            else:
                method = f"mtlora_{r['transform']}"
            faces = r.get("n_faces")
        else:
            method, faces = "lotr", None
        base = os.path.basename(path)
        seed = int(base[:-3].split("seed")[-1]) if "seed" in base else None
        rec = dict(method=method, task=r["task"], lr=r["lr"], seed=seed,
                   alpha=r.get("alpha"), rank=r["rank"], n_faces=faces,
                   epochs=len(curve), curve=curve,
                   loss=[float(h["loss"]) for h in r["history"]], m_rot=None,
                   all_faces_pruned=False)
    else:
        return None
    rec["path"] = path
    rec["mtime"] = os.path.getmtime(path)
    return rec


def load_runs(results_dir: str) -> tuple[list[dict], dict[str, list[dict]]]:
    """Load headline runs, one per (row, task, seed); return also what was left out."""
    files = sorted(p for p in glob(os.path.join(results_dir, "**", "*.pt"), recursive=True)
                   if os.path.isfile(p))
    chosen: dict = {}
    left_out: dict[str, list[dict]] = {"superseded": [], "invalid": [], "other_setting": []}
    for path in files:
        rec = read_run(path)
        if rec is None or rec["rank"] not in (8, 64):
            continue
        if rec["rank"] == 64:                       # only the parameter-matched LoTR baseline
            if rec["method"] != "lotr":
                continue
            rec["method"] = "lotr_r64"
        if rec["method"] not in ("lotr", "lotr_r64", "lora", "mtlora_identity", "cayley", "cayley_lora"):
            continue
        if rec["method"] in ("lora", "mtlora_identity") and rec["n_faces"] != HEADLINE_FACES:
            continue
        task = rec["task"]
        if (abs(rec["lr"] - HEADLINE_LR.get(task, DEFAULT_LR)) > 1e-12
                or abs((rec["alpha"] or 0) / rec["rank"] - HEADLINE_SCALING) > 1e-12):
            left_out["other_setting"].append(rec)
            continue
        if rec["all_faces_pruned"]:
            left_out["invalid"].append(rec)
            continue
        if rec["epochs"] != HEADLINE_EPOCHS.get(task, DEFAULT_EPOCHS):
            rec["method"] = f"{rec['method']}_ep{rec['epochs']}"
        if rec["method"] not in METHOD_ORDER:
            left_out["other_setting"].append(rec)
            continue
        key = (rec["method"], task, rec["seed"])
        old = chosen.get(key)
        if old is None:
            chosen[key] = rec
        elif rec["mtime"] >= old["mtime"]:
            left_out["superseded"].append(old)
            chosen[key] = rec
        else:
            left_out["superseded"].append(rec)
    return list(chosen.values()), left_out


def score_of(rec: dict, epoch: int | None) -> float | None:
    if epoch is None:
        return rec["curve"][-1]
    return rec["curve"][epoch - 1] if rec["epochs"] >= epoch else None


def mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def cell(scores: list[float], latex: bool) -> str:
    n = len(scores)
    if n == 0:
        return "---"
    m, s = mean_std(scores)
    if n == 1:
        return f"{m:.3f}" if latex else f"  {m:.3f}   (n=1)"
    return f"{m:.3f}$\\pm${s:.3f}" if latex else f"  {m:.3f}±{s:.3f} (n={n})"


def print_table(runs: list[dict], epoch: int | None, latex: bool) -> None:
    by = defaultdict(list)
    for r in runs:
        s = score_of(r, epoch)
        if s is not None:
            by[(r["method"], r["task"])].append(s)
    methods = [m for m in METHOD_ORDER if any(k[0] == m for k in by)]

    if latex:
        print(r"\begin{tabular}{l" + "c" * len(TASK_ORDER) + "}")
        print(r"\toprule")
        print(" & ".join(["Method"] + [t.upper() for t in TASK_ORDER]) + r" \\")
        print(r"\midrule")
    else:
        header = f"{'Method':<34}" + "".join(f"{t.upper():>20}" for t in TASK_ORDER)
        print(header)
        print("-" * len(header))
    for m in methods:
        cells = [cell(by.get((m, t), []), latex) for t in TASK_ORDER]
        if latex:
            print(METHOD_LABEL[m] + " & " + " & ".join(cells) + r" \\")
        else:
            print(f"{METHOD_LABEL[m]:<34}" + "".join(f"{c:>20}" for c in cells))
    if latex:
        print(r"\bottomrule")
        print(r"\end{tabular}")


def print_details(runs: list[dict], left_out: dict[str, list[dict]]) -> None:
    print("\nPer-run details (row, task, seed, alpha, epochs, final, loss first->last, M_rot):")
    for r in sorted(runs, key=lambda x: (x["method"], x["task"], x["seed"] if x["seed"] is not None else -1)):
        mrot = "" if r["m_rot"] is None else f"  M_rot={r['m_rot']:.3f}"
        print(f"  {r['method']:<22}{r['task']:<5} seed={r['seed']!s:<3} alpha={r['alpha']!s:<4}"
              f" ep={r['epochs']:<2} final={r['curve'][-1]:.4f}"
              f"  loss {r['loss'][0]:.3f}->{r['loss'][-1]:.3f}{mrot}")

    suspect = [r for r in runs if r["curve"][-1] <= SUSPECT_MAX.get(r["task"], -1)]
    stuck = [r for r in runs if r["loss"][-1] > 0.95 * r["loss"][0] and r["loss"][0] > 0.6]
    if suspect or stuck:
        print("\nSuspect runs (counted, but did not train well — check before reporting):")
        for r in suspect + [x for x in stuck if x not in suspect]:
            print(f"  {r['method']} {r['task']} seed={r['seed']} final={r['curve'][-1]:.4f} "
                  f"loss {r['loss'][0]:.3f}->{r['loss'][-1]:.3f}  ({r['path']})")

    titles = {
        "invalid": "INVALID runs excluded (all faces pruned; only the classifier head trained)",
        "superseded": "Superseded runs (same row/task/seed; the newer file is used)",
        "other_setting": "Runs outside the headline setting (other alpha / lr / epochs)",
    }
    for k, title in titles.items():
        if left_out[k]:
            print(f"\n{title}:")
            for r in left_out[k]:
                print(f"  {r['method']} {r['task']} seed={r['seed']} alpha={r['alpha']} lr={r['lr']} "
                      f"ep={r['epochs']} final={r['curve'][-1]:.4f}  ({r['path']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--epoch", type=int, default=None,
                    help="report the score at this epoch instead of the final epoch")
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs, left_out = load_runs(args.results_dir)
    where = "final epoch" if args.epoch is None else f"epoch {args.epoch}"
    print(f"Loaded {len(runs)} headline runs from {args.results_dir} "
          f"({len(left_out['invalid'])} invalid, {len(left_out['superseded'])} superseded, "
          f"{len(left_out['other_setting'])} other-setting)")
    print(f"=== Results summary (rank=8, alpha={HEADLINE_ALPHA:g}, {where}) ===\n")
    print_table(runs, args.epoch, args.latex)
    if not args.latex:
        print_details(runs, left_out)

    if args.out:
        serial: dict = {}
        for r in runs:
            s = score_of(r, args.epoch)
            if s is not None:
                serial.setdefault(r["method"], {}).setdefault(r["task"], {})[str(r["seed"])] = s
        with open(args.out, "w") as f:
            json.dump({"epoch": args.epoch or "final", "scores": serial}, f, indent=2)
        print(f"\nSaved JSON summary to {args.out}")


if __name__ == "__main__":
    main()
