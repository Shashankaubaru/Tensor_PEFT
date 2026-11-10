"""
collect_ablations.py — aggregate the CoLA ablations and the RTE learning-rate check
===================================================================================
Companion to collect_results.py (same final-epoch convention, same run reader).

  * CoLA ablations (abl_cola_*.pt, 5 epochs, rank 8, alpha 4, lr 5e-4):
    transform at p=12, n_faces with DCT, and W0-guided init at p=4.
    The Identity p=12 row comes from the headline runs (mtlora_cola_identity_*).
  * RTE learning-rate check (10 epochs, alpha/rank = 0.5): LoTR rank 8, LoTR rank 64
    (alpha 32) and Identity at every learning rate found in results/.

Usage:  python collect_ablations.py [--out results/ablations.json]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob

from collect_results import mean_std, read_run

W0_TAG = "w0init"


def fmt(v: list[float]) -> str:
    if not v:
        return "---"
    m, s = mean_std(v)
    return f"{m:.3f}" if len(v) == 1 else f"{m:.3f}±{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(p for p in glob(os.path.join(args.results_dir, "**", "*.pt"), recursive=True)
                   if os.path.isfile(p))

    abl: dict[str, dict[int, float]] = defaultdict(dict)
    extra: dict[str, dict] = defaultdict(dict)
    other: dict[str, dict[int, float]] = defaultdict(dict)   # MRPC / RTE mixing rows
    for p in files:
        base = os.path.basename(p)
        rec = read_run(p)
        if rec is None:
            continue
        final = rec["curve"][-1]
        if base.startswith("abl_cola_"):
            row = re.sub(r"_seed\d+\.pt$", "", base[len("abl_cola_"):])
            abl[row][rec["seed"]] = final
            extra[row]["epochs"] = rec["epochs"]
        elif base.startswith(("abl_mrpc_", "abl_rte_", "abl_sst2_")):
            row = re.sub(r"_seed\d+\.pt$", "", base[len("abl_"):])   # e.g. mrpc_dct_p12
            other[row][rec["seed"]] = final
        elif (rec["task"] == "cola" and rec["method"] == "mtlora_identity"
              and rec["n_faces"] == 12 and rec["rank"] == 8
              and abs(rec["lr"] - 5e-4) < 1e-12 and rec["epochs"] in (5, 10)
              and abs((rec["alpha"] or 0) - 4.0) < 1e-12):
            abl["identity_p12" if rec["epochs"] == 5 else "identity_p12_ep10"][rec["seed"]] = final
        elif (rec["task"] in ("mrpc", "rte") and rec["method"] == "mtlora_identity"
              and rec["n_faces"] == 12 and rec["rank"] == 8
              and abs((rec["alpha"] or 0) - 4.0) < 1e-12
              and abs(rec["lr"] - (1e-4 if rec["task"] == "rte" else 5e-4)) < 1e-12
              and rec["epochs"] == (10 if rec["task"] == "rte" else 5)):
            other[f"{rec['task']}_identity_p12"][rec["seed"]] = final
        elif (rec["task"] == "cola" and rec["method"] == "lora" and rec["n_faces"] == 12
              and rec["rank"] == 8 and abs(rec["lr"] - 5e-4) < 1e-12):
            abl["lora_p12" if rec["epochs"] == 5 else "lora_p12_ep10"][rec["seed"]] = final

    print("=== CoLA ablations (final-epoch MCC; 5 epochs unless the row says _ep10) ===")
    print("    *_lorainit rows use the Hu et al. initialisation (round 4); all others N(0,1) A, B=0")
    order = ["identity_p12", "lora_p12", "dct_p12", "random_p12", "dct_p1", "dct_p2", "dct_p4",
             "dct_p6", "dct_p4_w0init", "identity_p4", "identity_p6",
             "identity_p12_ep10", "lora_p12_ep10", "dct_p12_ep10", "random_p12_ep10",
             "dct_p12_lorainit", "random_p12_lorainit",
             "dct_p12_lorainit_ep10", "random_p12_lorainit_ep10",
             "dct_p6_lorainit", "identity_p6_lorainit", "lora_r4"]
    for row in order + sorted(set(abl) - set(order)):
        if row not in abl:
            print(f"  {row:<18} (no runs)")
            continue
        seeds = abl[row]
        vals = [seeds[s] for s in sorted(seeds)]
        print(f"  {row:<18} {fmt(vals):<14} n={len(vals)} seeds={sorted(seeds)}"
              f"  per-seed={[round(v, 3) for v in vals]}")

    print("\n=== Layer mixing beyond CoLA (final-epoch; MRPC 5 ep, RTE 10 ep at lr 1e-4) ===")
    for row in sorted(other):
        seeds = other[row]
        vals = [seeds[s] for s in sorted(seeds)]
        print(f"  {row:<18} {fmt(vals):<14} n={len(vals)} seeds={sorted(seeds)}"
              f"  per-seed={[round(v, 3) for v in vals]}")

    print("\n=== CoLA learning-rate check (5 epochs, final-epoch MCC, alpha/rank = 0.5) ===")
    cola_lr: dict[tuple, dict[int, float]] = defaultdict(dict)
    for p in files:
        rec = read_run(p)
        if (rec is None or rec["task"] != "cola" or rec["epochs"] != 5
                or abs((rec["alpha"] or 0) / rec["rank"] - 0.5) > 1e-12):
            continue
        if rec["rank"] == 64 and rec["method"] == "lotr":
            name = "lotr_r64"
        elif rec["rank"] == 8 and rec["method"] in ("mtlora_identity", "lora", "mtlora_dct_lorainit") \
                and rec["n_faces"] == 12:
            name = rec["method"]
        else:
            continue
        cola_lr[(name, rec["lr"])][rec["seed"]] = (rec["curve"][-1], rec["loss"][0], rec["loss"][-1])
    for (m, lr), seeds in sorted(cola_lr.items()):
        vals = [seeds[s][0] for s in sorted(seeds)]
        l0 = sum(seeds[s][1] for s in seeds) / len(seeds)
        l1 = sum(seeds[s][2] for s in seeds) / len(seeds)
        print(f"  {m:<16} lr={lr:<7g} {fmt(vals):<14} n={len(vals)} seeds={sorted(seeds)}"
              f"  loss {l0:.3f}->{l1:.3f}")

    print("\n=== RTE learning-rate check (final-epoch accuracy, 10 epochs, alpha/rank = 0.5) ===")
    rte: dict[tuple, dict[int, float]] = defaultdict(dict)
    for p in files:
        rec = read_run(p)
        if (rec is None or rec["task"] != "rte" or rec["rank"] not in (8, 64) or rec["epochs"] != 10
                or abs((rec["alpha"] or 0) / rec["rank"] - 0.5) > 1e-12):
            continue
        if rec["method"] in ("mtlora_identity", "lora", "mtlora_dct_lorainit") and rec["n_faces"] != 12:
            continue
        if rec["method"] not in ("lotr", "mtlora_identity", "lora", "mtlora_dct_lorainit"):
            continue
        if rec["rank"] == 64 and rec["method"] != "lotr":
            continue
        name = "lotr_r64" if rec["rank"] == 64 else rec["method"]
        rte[(name, rec["lr"])][rec["seed"]] = (rec["curve"][-1], rec["loss"][0], rec["loss"][-1])
    for (m, lr), seeds in sorted(rte.items()):
        vals = [seeds[s][0] for s in sorted(seeds)]
        l0 = sum(seeds[s][1] for s in seeds) / len(seeds)
        l1 = sum(seeds[s][2] for s in seeds) / len(seeds)
        print(f"  {m:<16} lr={lr:<7g} {fmt(vals):<14} n={len(vals)} seeds={sorted(seeds)}"
              f"  loss {l0:.3f}->{l1:.3f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"cola_ablations": {k: {str(s): v for s, v in d.items()} for k, d in abl.items()},
                       "mixing_other_tasks": {k: {str(s): v for s, v in d.items()} for k, d in other.items()},
                       "cola_lr": {f"{m}@{lr:g}": {str(s): v[0] for s, v in d.items()}
                                   for (m, lr), d in cola_lr.items()},
                       "rte_lr": {f"{m}@{lr:g}": {str(s): v[0] for s, v in d.items()}
                                  for (m, lr), d in rte.items()}}, f, indent=2)
        print(f"\nSaved JSON summary to {args.out}")


if __name__ == "__main__":
    main()
