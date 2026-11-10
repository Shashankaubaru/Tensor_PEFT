"""
Compare tensor decompositions of the stacked fine-tuning update Delta W in R^{n1 x n2 x L}
(L = number of transformer layers) by *approximation error vs. number of stored parameters*.

Methods
-------
lora_layer    : independent rank-r SVD per layer (what per-layer LoRA could at best represent)
tucker2       : Tucker2 (LoTR):  shared A, B (n x r) + per-layer r x r core
tucker3       : Tucker (r, r, min(r, L))  (FLoRA / TensLoRA-"Depth" style)
cp            : CP / PARAFAC (LoRTA / CaRA style)
tsvd_dct_*    : M-product t-SVD with M = DCT, keeping the p lowest depth-frequencies with rank r
                (`lowfreq`), or the energy-optimal global rank allocation over all frequencies (`global`)
tsvd_rand_*   : same with a random orthogonal M (control: no depth structure)
tsvd_id_global: M = identity, i.e. optimal rank allocation across independent layers (no depth sharing)

Because M is orthogonal, Frobenius error is preserved by the transform (Parseval), so all
t-SVD errors are computed exactly from the singular values of the transform-domain faces.
Parameter count for the t-SVD variants is (n1 + n2) * sum_k r_k (A_k, B_k per kept face).

Also reports depth-structure diagnostics: energy spectrum along depth in the DCT / random basis,
and principal angles between the dominant subspaces of different layers.

Usage:
    python delta_w_compare.py --dw results/dw_cola_seed0.pt --proj q v --out results/cmp_cola
    python delta_w_compare.py --synthetic --out results/cmp_synth        # sanity check
"""
import argparse
import json
import os

import numpy as np
import torch

from tensor_utils import modek_product
from tensor_utils.transformation_matrices import dct_matrix

DT = torch.float64


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def fro2(X):
    return float((X ** 2).sum())


def transform(X, M):
    """Apply M along the depth mode: Xh[..., k] = sum_l M[k, l] X[..., l]."""
    return modek_product(X, M, k=2)


def inv_transform(Xh, M):
    """M orthogonal => M^{-1} = M^T."""
    return modek_product(Xh, M, k=2, transpose=True)


def face_svdvals(Xh):
    """Singular values of every frontal face, shape (L, min(n1, n2))."""
    return torch.stack([torch.linalg.svdvals(Xh[:, :, k]) for k in range(Xh.shape[2])])


def top_left(Mat, r):
    """Top-r left singular vectors via the Gram matrix."""
    evals, evecs = torch.linalg.eigh(Mat @ Mat.T)
    return evecs[:, -r:].flip(1)


def make_M(kind, L, seed=0):
    if kind == "dct":
        return dct_matrix(L, dtype=DT)
    if kind == "rand":
        g = torch.Generator().manual_seed(seed)
        q, _ = torch.linalg.qr(torch.randn(L, L, generator=g, dtype=DT))
        return q.contiguous()
    if kind == "id":
        return torch.eye(L, dtype=DT)
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# decompositions: each returns a list of dicts {method, params, rel_err, **config}
# --------------------------------------------------------------------------------------
def run_lora_layer(sv_id, n1, n2, nrm2, ranks):
    out = []
    for r in ranks:
        if r > sv_id.shape[1]:
            continue
        e2 = float((sv_id[:, r:] ** 2).sum())
        out.append(dict(method="lora_layer", params=sv_id.shape[0] * r * (n1 + n2),
                        rel_err=(e2 / nrm2) ** 0.5, r=r))
    return out


def run_tsvd_global(sv, n1, n2, nrm2, name, n_points=40):
    """Energy-optimal rank allocation: keep the m largest singular values over all faces."""
    flat = torch.sort((sv ** 2).flatten(), descending=True).values
    cum = torch.cumsum(flat, 0)
    ms = np.unique(np.round(np.geomspace(1, len(flat), n_points)).astype(int))
    out = []
    for m in ms:
        e2 = max(nrm2 - float(cum[m - 1]), 0.0)
        out.append(dict(method=f"tsvd_{name}_global", params=int(m) * (n1 + n2),
                        rel_err=(e2 / nrm2) ** 0.5, m=int(m)))
    return out


def run_tsvd_lowfreq(sv, n1, n2, nrm2, name, ps, ranks):
    """Keep the first p faces (lowest depth-frequencies), each at uniform rank r."""
    L = sv.shape[0]
    out = []
    for p in ps:
        p = min(p, L)
        for r in ranks:
            if r > sv.shape[1]:
                continue
            kept = float((sv[:p, :r] ** 2).sum())
            e2 = max(nrm2 - kept, 0.0)
            out.append(dict(method=f"tsvd_{name}_lowfreq", params=p * r * (n1 + n2),
                            rel_err=(e2 / nrm2) ** 0.5, p=p, r=r))
    return out


def tucker_hooi(X, ranks, iters=3, share_depth=False):
    """HOOI Tucker with orthonormal factors; error from ||X||^2 - ||core||^2.
    share_depth=True => Tucker2 (depth factor = identity, ranks[2] ignored)."""
    n1, n2, L = X.shape
    r1, r2, r3 = min(ranks[0], n1), min(ranks[1], n2), min(ranks[2], L)
    if share_depth:
        r3 = L
    A = top_left(X.reshape(n1, -1), r1)
    B = top_left(X.permute(1, 0, 2).reshape(n2, -1), r2)
    C = torch.eye(L, dtype=X.dtype) if share_depth else top_left(X.permute(2, 0, 1).reshape(L, -1), r3)
    for _ in range(iters):
        Y = torch.einsum("ijl,jb->ibl", X, B)
        Y = torch.einsum("ibl,lc->ibc", Y, C)
        A = top_left(Y.reshape(n1, -1), r1)
        Y = torch.einsum("ijl,ia->ajl", X, A)
        Y = torch.einsum("ajl,lc->ajc", Y, C)
        B = top_left(Y.permute(1, 0, 2).reshape(n2, -1), r2)
        if not share_depth:
            Y = torch.einsum("ijl,ia->ajl", X, A)
            Y = torch.einsum("ajl,jb->abl", Y, B)
            C = top_left(Y.permute(2, 0, 1).reshape(L, -1), r3)
    core = torch.einsum("ijl,ia->ajl", X, A)
    core = torch.einsum("ajl,jb->abl", core, B)
    core = torch.einsum("abl,lc->abc", core, C)
    params = r1 * n1 + r2 * n2 + r1 * r2 * (L if share_depth else r3) + (0 if share_depth else r3 * L)
    return fro2(core), params, (r1, r2, r3)


def run_tucker(X, nrm2, ranks, iters, depth_mode):
    out = []
    for r in ranks:
        if depth_mode == "tucker2":
            c2, params, rk = tucker_hooi(X, (r, r, X.shape[2]), iters, share_depth=True)
        else:
            c2, params, rk = tucker_hooi(X, (r, r, min(r, X.shape[2])), iters)
        out.append(dict(method=depth_mode, params=int(params),
                        rel_err=(max(nrm2 - c2, 0.0) / nrm2) ** 0.5, ranks=list(rk)))
    return out


def run_cp(X, nrm2, ranks, n_iter):
    import tensorly as tl
    from tensorly.decomposition import parafac
    tl.set_backend("pytorch")
    out = []
    n1, n2, L = X.shape
    for R in ranks:
        try:
            cp = parafac(X, rank=R, n_iter_max=n_iter, init="svd", tol=1e-7)
            e2 = fro2(X - tl.cp_to_tensor(cp))
            out.append(dict(method="cp", params=R * (n1 + n2 + L), rel_err=(e2 / nrm2) ** 0.5, R=R))
        except Exception as e:  # keep going; CP can fail for some ranks/inits
            print(f"  [cp R={R}] failed: {type(e).__name__}: {e}")
    return out


# --------------------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------------------
def depth_energy_spectrum(X, M):
    Xh = transform(X, M)
    e = (Xh ** 2).sum(dim=(0, 1))
    return (e / e.sum()).tolist()


def layer_subspace_overlap(X, k=16, seed=0):
    """Mean cosine of principal angles between top-k left/right subspaces of different layers."""
    n1, n2, L = X.shape
    Us, Vs = [], []
    for l in range(L):
        U, S, Vh = torch.linalg.svd(X[:, :, l], full_matrices=False)
        Us.append(U[:, :k])
        Vs.append(Vh[:k].T)

    def mean_cos(F):
        by_dist = {}
        for i in range(L):
            for j in range(i + 1, L):
                by_dist.setdefault(j - i, []).append(float(torch.linalg.svdvals(F[i].T @ F[j]).mean()))
        return {d: float(np.mean(v)) for d, v in by_dist.items()}

    g = torch.Generator().manual_seed(seed)

    def rand_baseline(n):
        a = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=DT))[0]
        b = torch.linalg.qr(torch.randn(n, k, generator=g, dtype=DT))[0]
        return float(torch.linalg.svdvals(a.T @ b).mean())

    return dict(k=k, left_by_distance=mean_cos(Us), right_by_distance=mean_cos(Vs),
                random_left=rand_baseline(n1), random_right=rand_baseline(n2))


def layer_cosine_matrix(X):
    L = X.shape[2]
    F = X.reshape(-1, L)
    F = F / F.norm(dim=0, keepdim=True).clamp_min(1e-30)
    return (F.T @ F).tolist()


def pareto(points):
    pts = sorted(points, key=lambda d: d["params"])
    best, out = float("inf"), []
    for d in pts:
        if d["rel_err"] < best:
            best = d["rel_err"]
            out.append(d)
    return out


def params_to_reach(points, thr):
    ok = [d["params"] for d in points if d["rel_err"] <= thr]
    return min(ok) if ok else None


# --------------------------------------------------------------------------------------
def analyse(X, args, label):
    X = X.to(DT)
    n1, n2, L = X.shape
    nrm2 = fro2(X)
    print(f"\n=== {label}: shape {tuple(X.shape)}  ||X||_F = {nrm2 ** 0.5:.4f} ===")
    results = []

    Ms = {"dct": make_M("dct", L), "rand": make_M("rand", L, args.seed), "id": make_M("id", L)}
    svs = {}
    for name, M in Ms.items():
        Xh = transform(X, M)
        # Parseval sanity check (orthogonal M)
        assert abs(fro2(Xh) - nrm2) / nrm2 < 1e-8, "transform is not norm preserving"
        svs[name] = face_svdvals(Xh)

    ranks = args.ranks
    results += run_lora_layer(svs["id"], n1, n2, nrm2, ranks)
    for name in ("dct", "rand", "id"):
        results += run_tsvd_global(svs[name], n1, n2, nrm2, name)
    ps = [p for p in args.faces if p <= L] or [L]
    for name in ("dct", "rand"):
        results += run_tsvd_lowfreq(svs[name], n1, n2, nrm2, name, ps, ranks)

    print("tucker2 ...")
    results += run_tucker(X, nrm2, ranks, args.hooi_iters, "tucker2")
    print("tucker3 ...")
    results += run_tucker(X, nrm2, ranks, args.hooi_iters, "tucker3")
    if not args.skip_cp:
        print("cp ...")
        results += run_cp(X, nrm2, args.cp_ranks, args.cp_iters)

    methods = sorted({d["method"] for d in results})
    fronts = {m: pareto([d for d in results if d["method"] == m]) for m in methods}

    thresholds = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    print(f"\nMin #params to reach relative Frobenius error <= thr   (LoRA r=8 on all layers = {L * 8 * (n1 + n2)})")
    print(f"{'method':<22}" + "".join(f"{t:>12}" for t in thresholds))
    table = {}
    for m in methods:
        row = [params_to_reach(fronts[m], t) for t in thresholds]
        table[m] = dict(zip(map(str, thresholds), row))
        print(f"{m:<22}" + "".join(f"{('-' if v is None else v):>12}" for v in row))

    spec = {k: depth_energy_spectrum(X, Ms[k]) for k in ("dct", "rand")}
    p_lo = max(1, L // 4)
    print(f"\nEnergy in lowest {p_lo}/{L} depth-frequencies:  DCT = {sum(spec['dct'][:p_lo]):.3f}   "
          f"random-M = {sum(spec['rand'][:p_lo]):.3f}   (uniform would be {p_lo / L:.3f})")

    ov = layer_subspace_overlap(X, k=args.subspace_k, seed=args.seed)
    adj = ov["left_by_distance"].get(1)
    far = ov["left_by_distance"].get(L - 1)
    print(f"Mean cos(principal angles), top-{ov['k']} left subspaces: adjacent layers = {adj:.3f}, "
          f"layers 0 vs {L - 1} = {far:.3f}, random baseline = {ov['random_left']:.3f}")

    return dict(label=label, shape=[n1, n2, L], points=results, table=table, depth_spectrum=spec,
                subspace_overlap=ov, layer_cosine=layer_cosine_matrix(X))


def plot(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({d["method"] for d in res["points"]})
    style = {"lora_layer": ("k", "o"), "tucker2": ("tab:blue", "s"), "tucker3": ("tab:cyan", "s"), "cp": ("tab:green", "^"),
             "tsvd_dct_global": ("tab:red", "D"), "tsvd_dct_lowfreq": ("tab:orange", "D"),
             "tsvd_rand_global": ("tab:gray", "x"), "tsvd_rand_lowfreq": ("tab:pink", "x"),
             "tsvd_id_global": ("tab:purple", "v")}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for m in methods:
        pts = pareto([d for d in res["points"] if d["method"] == m])
        c, mk = style.get(m, ("k", "."))
        ax[0].plot([d["params"] for d in pts], [d["rel_err"] for d in pts], color=c, marker=mk, label=m, ms=4)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("# stored parameters")
    ax[0].set_ylabel(r"$\|\Delta W-\hat{\Delta W}\|_F / \|\Delta W\|_F$")
    ax[0].set_title(f"{res['label']}: error vs parameters")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)
    L = res["shape"][2]
    ax[1].plot(range(L), res["depth_spectrum"]["dct"], "r-D", label="DCT basis")
    ax[1].plot(range(L), res["depth_spectrum"]["rand"], "g-x", label="random orthogonal M")
    ax[1].set_xlabel("depth-frequency index k")
    ax[1].set_ylabel("fraction of energy")
    ax[1].set_title("Energy spectrum of Delta W along depth")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def synthetic(n=256, L=12, seed=0, noise=0.05):
    """Planted structure: DCT-domain faces 0..2 are rank-4 matrices, all other faces are zero."""
    g = torch.Generator().manual_seed(seed)
    M = make_M("dct", L)
    Xh = torch.zeros(n, n, L, dtype=DT)
    for f in range(3):
        U = torch.randn(n, 4, generator=g, dtype=DT)
        V = torch.randn(n, 4, generator=g, dtype=DT)
        Xh[:, :, f] = (U @ V.T) / (n ** 0.5) / (f + 1)
    X = inv_transform(Xh, M)
    X = X + noise * X.std() * torch.randn(X.shape, generator=g, dtype=DT)
    return X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dw", help="file produced by delta_w_extract.py")
    ap.add_argument("--proj", nargs="+", default=["q", "v"])
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--out", default="results/cmp")
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--faces", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12])
    ap.add_argument("--cp_ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--cp_iters", type=int, default=100)
    ap.add_argument("--hooi_iters", type=int, default=3)
    ap.add_argument("--subspace_k", type=int, default=16)
    ap.add_argument("--skip_cp", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    jobs, meta = [], None
    if args.synthetic:
        jobs.append(("synthetic", synthetic(seed=args.seed)))
    else:
        if not args.dw:
            ap.error("--dw is required unless --synthetic")
        blob = torch.load(args.dw, map_location="cpu")
        meta = blob.get("meta")
        print("meta:", meta)
        for p in args.proj:
            jobs.append((p, blob[p]))

    for label, X in jobs:
        res = analyse(X, args, label)
        res["meta"] = meta
        with open(os.path.join(args.out, f"{label}.json"), "w") as f:
            json.dump(res, f)
        plot(res, os.path.join(args.out, f"{label}.png"))
        print("wrote", os.path.join(args.out, f"{label}.json / .png"))


if __name__ == "__main__":
    main()
