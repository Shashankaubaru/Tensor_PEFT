"""
collect_regimes.py — aggregate the round-5 "regimes where coupling could matter" runs
=====================================================================================
Reads results/reg_{budget,data,llm15,llm7}_*.pt (final-epoch score, same convention as
collect_results.py) and prints, per setting, the mean±sd over seeds for every configuration
and the difference of each coupled configuration to the standard-LoRA configuration of the
SAME adapter budget, with a 95% Welch interval, Welch p and paired (same-seed) p.

Budget = rank-units p*r (LoRA rank-r on all L layers has p*r = L*r rank-units):
the adapters of all configurations inside one budget have the same number of parameters.

Usage:  python collect_regimes.py [--out results/regimes.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from collections import defaultdict
from glob import glob

import torch
from scipy import stats

NAME_RE = re.compile(r"^reg_(budget|data|llm15|llm7)_([a-z0-9]+)_(.+?)(?:_lr(\w+?))?_seed(\d+)\.pt$")


def read(path: str):
    try:
        r = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None
    return r if "history" in r else None


def units(cfg: str, L: int) -> int | None:
    """rank-units p*r from a config name such as lora_r1, dct_p2_r6, id_p2_r6, cayd3_p2_r6, dct_p28_r8."""
    m = re.match(r"^lora_r(\d+)$", cfg)
    if m:
        return L * int(m.group(1))
    m = re.match(r"^[a-z0-9]+_p(\d+)_r(\d+)$", cfg)
    if m:
        return int(m.group(1)) * int(m.group(2))
    return None


def welch(a: list[float], b: list[float]):
    """a-b in points with a 95% Welch CI and p."""
    if len(a) < 2 or len(b) < 2:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return dict(diff=100 * (ma - mb), lo=float("nan"), hi=float("nan"), p=float("nan"))
    df = se**4 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    p = 2 * stats.t.sf(abs((ma - mb) / se), df)
    h = stats.t.ppf(0.975, df) * se
    return dict(diff=100 * (ma - mb), lo=100 * (ma - mb - h), hi=100 * (ma - mb + h), p=float(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # (group, task, lr) -> cfg -> seed -> final score
    table: dict = defaultdict(lambda: defaultdict(dict))
    meta: dict = {}
    n_files = 0
    for p in sorted(glob(os.path.join(args.results_dir, "reg_*.pt"))):
        m = NAME_RE.match(os.path.basename(p))
        if not m:
            continue
        grp, task, cfg, _, seed = m.group(1), m.group(2), m.group(3), m.group(4), int(m.group(5))
        r = read(p)
        if r is None:
            continue
        n_files += 1
        key = (grp, task, f"{r['lr']:g}")
        table[key][cfg][seed] = float(r["history"][-1]["score"])
        L = r.get("n_layers") or (28 if grp.startswith("llm") else 12)
        meta[(key, cfg)] = dict(L=L, params=r.get("trainable_params"), epochs=r["epochs"])
    print(f"Loaded {n_files} regime runs\n")

    out = {}
    for key in sorted(table):
        grp, task, lr = key
        cfgs = table[key]
        L = meta[(key, next(iter(cfgs)))]["L"]
        by_budget: dict = defaultdict(list)
        for c in cfgs:
            by_budget[units(c, L)].append(c)
        print(f"=== {grp}  task={task}  lr={lr}  (L={L}) ===")
        res_key = f"{grp}|{task}|{lr}"
        out[res_key] = {}
        for u in sorted(by_budget, key=lambda x: (x is None, x)):
            ref = next((c for c in by_budget[u] if c.startswith("lora_r")), None)
            print(f"  budget {u} rank-units" if u else "  (unclassified)")
            for c in sorted(by_budget[u], key=lambda c: (c != ref, c)):
                seeds = cfgs[c]
                v = [seeds[s] for s in sorted(seeds)]
                mean = statistics.mean(v)
                sd = statistics.stdev(v) if len(v) > 1 else 0.0
                line = f"    {c:<14} n={len(v)}  {mean:.4f}±{sd:.4f}  params={meta[(key, c)]['params']}"
                rec = dict(n=len(v), mean=mean, sd=sd, scores=seeds, params=meta[(key, c)]["params"],
                           budget=u)
                if ref and c != ref:
                    w = welch(v, [cfgs[ref][s] for s in sorted(cfgs[ref])])
                    common = sorted(set(seeds) & set(cfgs[ref]))
                    pp = None
                    if len(common) >= 3:
                        x = [seeds[s] for s in common]
                        y = [cfgs[ref][s] for s in common]
                        if statistics.pstdev([a - b for a, b in zip(x, y)]) > 0:
                            pp = float(stats.ttest_rel(x, y).pvalue)
                    if w:
                        line += (f"   vs {ref}: {w['diff']:+.1f} [{w['lo']:+.1f}, {w['hi']:+.1f}] "
                                 f"Welch p={w['p']:.3f}" + (f" paired p={pp:.3f}" if pp is not None else ""))
                        rec.update(vs=ref, diff_pts=w["diff"], ci=[w["lo"], w["hi"]],
                                   welch_p=w["p"], paired_p=pp)
                dead = [s for s, x in seeds.items() if x <= 0.01]
                if dead:
                    line += f"   [constant-prediction seeds: {dead}]"
                print(line)
                out[res_key][c] = rec
        print()

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
