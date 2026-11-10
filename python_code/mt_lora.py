"""
MTLoRA — M-product Tensor LoRA
==============================
A parameter-efficient fine-tuning adapter that operates in the M-product algebra.

Core idea
---------
Stack the adapters for *L* linear layers of the same type (e.g. all Q-projections) along a
third axis to form order-3 tensors  A ∈ R^{d_out × r × L}  and  B ∈ R^{r × d_in × L}.
The update to layer ℓ is the ℓ-th frontal slice of the M-product

    ΔW_ℓ  =  (A ⋆_M B)[:, :, ℓ]

The M-product `A ⋆_M B` is defined as

    A ⋆_M B  =  M⁻¹ ×₃ facewise( M ×₃ A,  M ×₃ B )

where M is an orthogonal L×L matrix (DCT by default) and ×₃ denotes the mode-3 product.
In the *transform domain* every face is an independent r×d_in → d_out linear map, so the
forward pass of layer ℓ reduces to a sum over p ≤ L transform-domain pairs:

    y_ℓ  =  W₀_ℓ x  +  α/r · Σ_f  (M⁻¹)_{ℓ,f} · Â_f (B̂_f x)

where  Â_f = Â[:, :, f],  B̂_f = B̂[:, :, f]  live in the transform domain.

Initialisation (Phase 1)
------------------------
  - Apply tSVD-M( W₀, M, rank=r ) to the stacked pretrained weights.
  - Init  Â_f = U_f (left M-singular tensor, shape d_out × r)  for each face f.
  - Init  B̂_f = 0  (zero start → ΔW = 0 at step 0, unaltered pretrained behaviour).

Dynamic rank pruning
---------------------
After each optimiser step call  adapter.prune_dead_faces(threshold)  which zeroes out any
face f where  ‖B̂_f‖_F < threshold.  This is principled because the M-singular values of
the update tensor are monotone-decreasing within each face after init, analogously to matrix SVD.

Usage
-----
See MTLoRALinear, which wraps an existing nn.Linear, and mt_lora_demo.py for a full example.

References
----------
- Kilmer & Martin (2011).  Factorization strategies for third-order tensors.
- Kernfeld, Kilmer & Aeron (2015).  Tensor–tensor products with invertible linear transforms.
- Hu et al. (2021).  LoRA: Low-Rank Adaptation of Large Language Models.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from tensor_utils.transformation_matrices import dct_matrix, random_orthogonal


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_M(kind: str, L: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return an orthogonal L×L transform matrix as a non-differentiable buffer."""
    if kind == "dct":
        M = dct_matrix(L, dtype=dtype, device=device)
    elif kind == "random":
        M = random_orthogonal(L, dtype=dtype, device=device)
    elif kind == "identity":
        M = torch.eye(L, dtype=dtype, device=device)
    else:
        raise ValueError(f"Unknown transform kind '{kind}'. Use 'dct', 'random' or 'identity'.")
    return M


def _facewise_bmm(A_hat: torch.Tensor, B_hat: torch.Tensor) -> torch.Tensor:
    """
    Batched matmul over the L (last) axis.
    A_hat : (d_out, r, L)
    B_hat : (r, d_in, L)
    returns (d_out, d_in, L)
    """
    # permute to (L, d_out, r)  and  (L, r, d_in)  for torch.bmm
    A = A_hat.permute(2, 0, 1)   # (L, d_out, r)
    B = B_hat.permute(2, 0, 1)   # (L, r, d_in)
    C = torch.bmm(A, B)          # (L, d_out, d_in)
    return C.permute(1, 2, 0)    # (d_out, d_in, L)


def _modek3(X: torch.Tensor, M: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Mode-3 product along the last axis.
    X : (*, L),   M : (L, L)  =>  result: (*, L)
    If inverse and M is orthogonal, multiply by M^T.
    """
    if inverse:
        return X @ M           # X @ Mᵀᵀ = X @ M  when we stored Mᵀ here... see below
    return X @ M.T


# NOTE: modek_product(X, M, k=2) computes  X ×₃ M  which is defined as
#   output[i,j,k] = Σ_l M[k,l] X[i,j,l]
# In matrix form along the last axis: output = X @ M^T
# So to transform:   X_hat = X @ M^T  =>  _modek3(X, M, inverse=False)
# To invert (M orth): X = X_hat @ M   =>  _modek3(X_hat, M, inverse=True)
# This matches the modek_product convention used in tSVD_M.py.


# ---------------------------------------------------------------------------
# MTLoRA core
# ---------------------------------------------------------------------------

class MTLoRA(nn.Module):
    """
    Manages one family of M-product tensor adapters: all projections of type `proj_name`
    across `L` layers.  Each layer ℓ gets its update  ΔW_ℓ = (A ⋆_M B)[:,:,ℓ].

    Parameters
    ----------
    d_out, d_in : int
        Weight matrix dimensions (output features × input features).
    L : int
        Number of layers sharing this adapter.
    rank : int
        M-tubal rank r (number of transform-domain columns kept per face).
    n_faces : int or None
        Number of transform-domain faces to keep (p ≤ L).  If None, keep all L.
    transform : str
        'dct' | 'random' | 'identity'
    alpha : float
        LoRA scaling factor (applied as alpha/rank in forward).
    init_from_weights : bool
        If True, initialise Â from tSVD-M of the stacked pretrained weights.
        Requires calling  .init_from_stacked(W0_stacked)  after construction.
        If False, Â is random Gaussian, B̂ = 0  (standard LoRA init).
    dtype : torch.dtype
    """

    def __init__(
        self,
        d_out: int,
        d_in: int,
        L: int,
        rank: int = 8,
        n_faces: Optional[int] = None,
        transform: str = "dct",
        alpha: float = 1.0,
        init_from_weights: bool = False,
        dtype: torch.dtype = torch.float32,
        a_init: str = "normal",
    ) -> None:
        super().__init__()
        if a_init not in ("normal", "lora"):
            raise ValueError(f"a_init must be 'normal' or 'lora', got {a_init!r}")
        self.a_init = a_init
        self.d_out = d_out
        self.d_in = d_in
        self.L = L
        self.rank = rank
        self.n_faces = min(n_faces, L) if n_faces is not None else L
        self.alpha = alpha
        self.scaling = alpha / rank
        self.init_from_weights = init_from_weights

        # M is a fixed buffer (not a parameter) unless you later swap it for nn.Parameter
        M = _make_M(transform, L, dtype=dtype, device=torch.device("cpu"))
        self.register_buffer("M", M)          # (L, L)
        # Also cache M^T = M^{-1} (M is orthogonal)
        self.register_buffer("Mt", M.T.contiguous())  # (L, L)

        # Transform-domain parameters Â ∈ R^{d_out × r × L} and B̂ ∈ R^{r × d_in × L}
        # We only allocate the *kept* faces (p faces), not all L.
        p = self.n_faces
        self.A_hat = nn.Parameter(torch.empty(d_out, rank, p, dtype=dtype))
        self.B_hat = nn.Parameter(torch.empty(rank, d_in, p, dtype=dtype))

        # face_mask: boolean buffer, shape (p,).  False = face is pruned (zeroed).
        self.register_buffer("face_mask", torch.ones(p, dtype=torch.bool))
        # Track face energies for rank diagnostics.
        self.register_buffer("face_energy", torch.zeros(p, dtype=dtype))

        if not init_from_weights:
            self._random_init()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _random_init(self) -> None:
        """
        a_init='normal' (default): A_hat ~ N(0,1), B_hat = 0.
        a_init='lora': the Hu et al. LoRA initialisation. The random factor is the
            INPUT-side one, B_hat (r x d_in) ~ U(-1/sqrt(d_in), 1/sqrt(d_in)), which is
            nn.init.kaiming_uniform_(a=sqrt(5)) on an (r, d_in) matrix, and the OUTPUT-side
            factor A_hat (d_out x r) = 0. With transform='identity' and n_faces=L this is
            exactly per-layer LoRA.
        """
        if self.a_init == "lora":
            bound = 1.0 / math.sqrt(self.d_in)
            nn.init.uniform_(self.B_hat, -bound, bound)
            nn.init.zeros_(self.A_hat)
        else:
            nn.init.normal_(self.A_hat, 0.0, 1.0)
            nn.init.zeros_(self.B_hat)

    @torch.no_grad()
    def init_from_stacked(self, W0: torch.Tensor) -> None:
        """
        Phase-1 initialisation from the stacked pretrained weight tensor.

        W0 : (d_out, d_in, L)  — stacked pretrained weights (frozen).

        Steps:
          1. Transform W0 to the M-domain: Ŵ = W0 ×₃ M
          2. For each kept face f ∈ {0 … p-1}: SVD Ŵ[:,:,f] = U Σ Vᵀ, truncate to rank r.
             Set  Â_f ← U[:, :r]  (d_out × r)
             Set  B̂_f ← 0         (r × d_in)  — ensures ΔW = 0 at step 0.
          3. The geometric interpretation: Â is aligned to the M-singular subspace of W0,
             B̂ is zero so the adapter starts silent.  Adaptation optimises B̂ first
             (similar to LoRA where A is frozen for the first few steps by initialisation).
        """
        assert W0.shape == (self.d_out, self.d_in, self.L), (
            f"Expected W0 shape ({self.d_out}, {self.d_in}, {self.L}), got {tuple(W0.shape)}"
        )
        W0 = W0.to(dtype=self.M.dtype, device=self.M.device)
        # Transform along depth axis
        W_hat = _modek3(W0, self.M, inverse=False)  # (d_out, d_in, L)

        p = self.n_faces
        A_init = torch.zeros_like(self.A_hat)
        for f in range(p):
            face = W_hat[:, :, f]                          # (d_out, d_in)
            U, S, Vh = torch.linalg.svd(face, full_matrices=False)
            r_eff = min(self.rank, S.shape[0])
            A_init[:, :r_eff, f] = U[:, :r_eff]

        self.A_hat.data.copy_(A_init)
        nn.init.zeros_(self.B_hat)
        print(f"[MTLoRA] init_from_stacked: used top-{p} faces of W0 tSVD-M "
              f"(d_out={self.d_out}, d_in={self.d_in}, L={self.L}, rank={self.rank})")

    # ------------------------------------------------------------------
    # Forward: compute the per-layer update tensors ΔW[:,:,ℓ] for all ℓ
    # ------------------------------------------------------------------

    def compute_delta_hat(self) -> torch.Tensor:
        """
        Compute the transform-domain update  ΔŴ = facewise(Â, B̂)
        with dead faces zeroed by the face_mask.

        Returns  ΔŴ : (d_out, d_in, p)  in the transform domain (kept faces only).
        """
        # Apply mask: zero out pruned faces without breaking grad flow
        mask = self.face_mask.to(self.A_hat.dtype)        # (p,)
        # broadcast over d_out × rank  and  rank × d_in
        A = self.A_hat * mask[None, None, :]               # (d_out, r, p)
        B = self.B_hat * mask[None, None, :]               # (r, d_in, p)
        return _facewise_bmm(A, B)                         # (d_out, d_in, p)

    def get_delta_W(self) -> torch.Tensor:
        """
        Return ΔW : (d_out, d_in, L)  in the original (spatial) domain.

        If n_faces < L, the missing faces are treated as zero in the transform domain
        before the inverse transform — i.e. we embed the p kept faces into a zero-padded
        (d_out, d_in, L) transform-domain tensor and then apply M⁻¹.
        """
        Delta_hat_p = self.compute_delta_hat()             # (d_out, d_in, p)
        if self.n_faces == self.L:
            Delta_hat = Delta_hat_p
        else:
            # Embed into full L-face tensor (zero-pad the unused high-freq faces)
            Delta_hat = torch.zeros(
                self.d_out, self.d_in, self.L,
                dtype=Delta_hat_p.dtype, device=Delta_hat_p.device
            )
            Delta_hat[:, :, : self.n_faces] = Delta_hat_p
        # Inverse transform: ΔW = ΔŴ ×₃ M⁻¹ = ΔŴ @ M  (M orthogonal)
        return _modek3(Delta_hat, self.M, inverse=True)    # (d_out, d_in, L)

    # ------------------------------------------------------------------
    # Dynamic rank pruning
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune_dead_faces(self, threshold: float = 1e-4) -> int:
        """
        Zero out faces whose B̂_f Frobenius norm falls below `threshold`.
        Updates face_mask and face_energy in-place.
        Returns the number of newly pruned faces.
        """
        n_pruned = 0
        for f in range(self.n_faces):
            if not self.face_mask[f]:
                continue
            e = float(self.B_hat[:, :, f].norm(p="fro"))
            self.face_energy[f] = e
            if e < threshold:
                self.face_mask[f] = False
                self.A_hat[:, :, f].zero_()
                self.B_hat[:, :, f].zero_()
                n_pruned += 1
        return n_pruned

    def active_face_count(self) -> int:
        return int(self.face_mask.sum())

    def parameter_count(self) -> int:
        active = self.active_face_count()
        return active * self.rank * (self.d_out + self.d_in)

    def extra_repr(self) -> str:
        return (f"d_out={self.d_out}, d_in={self.d_in}, L={self.L}, "
                f"rank={self.rank}, n_faces={self.n_faces}, alpha={self.alpha}")


# ---------------------------------------------------------------------------
# Per-layer wrapper
# ---------------------------------------------------------------------------

class MTLoRALinear(nn.Module):
    """
    Wraps one nn.Linear (layer ℓ of a group) and adds an MTLoRA update.

    Instead of instantiating MTLoRA per layer, the shared MTLoRA object is
    passed in and this wrapper just holds the layer index ℓ.

    Forward:
        y = W₀ x  +  scaling · ΔW_ℓ x
    where  ΔW_ℓ = get_delta_W()[:, :, layer_idx].
    """

    def __init__(self, linear: nn.Linear, adapter: MTLoRA, layer_idx: int) -> None:
        super().__init__()
        # Freeze the pretrained weight
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        # Get ΔW for this layer: (d_out, d_in)
        delta_W = self.adapter.get_delta_W()[:, :, self.layer_idx]
        # Scale and apply
        return base + self.adapter.scaling * (x @ delta_W.T)

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}"


# ---------------------------------------------------------------------------
# Convenience: wrap all same-type projections of a HuggingFace model
# ---------------------------------------------------------------------------

def wrap_roberta_attention(
    model: nn.Module,
    proj_name: str = "query",
    rank: int = 8,
    n_faces: Optional[int] = None,
    transform: str = "dct",
    alpha: float = 1.0,
    init_from_weights: bool = False,
) -> MTLoRA:
    """
    Find all  encoder.layer.*.attention.self.{proj_name}  linear layers in `model`,
    collect them into an MTLoRA adapter, optionally initialise from W₀, and
    replace each layer with an MTLoRALinear wrapper in-place.

    Returns the shared MTLoRA adapter (needed to call prune_dead_faces etc.).
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
        raise ValueError(
            f"No layers matched pattern for proj_name='{proj_name}'. "
            f"Check the model architecture."
        )

    layers_sorted = sorted(found)
    L = len(layers_sorted)
    sample_linear = found[layers_sorted[0]][1]
    d_out, d_in = sample_linear.weight.shape   # (out_features, in_features)

    adapter = MTLoRA(
        d_out=d_out, d_in=d_in, L=L, rank=rank, n_faces=n_faces,
        transform=transform, alpha=alpha, init_from_weights=init_from_weights,
    )

    if init_from_weights:
        # Stack pretrained weights into (d_out, d_in, L)
        W0 = torch.stack(
            [found[idx][1].weight.detach().float() for idx in layers_sorted], dim=2
        )
        adapter.init_from_stacked(W0)

    # Replace each linear in-place
    for local_idx, layer_idx in enumerate(layers_sorted):
        full_name, linear = found[layer_idx]
        wrapper = MTLoRALinear(linear, adapter, layer_idx=local_idx)
        # Navigate to parent and swap attribute
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    print(
        f"[MTLoRA] wrapped {L} '{proj_name}' layers "
        f"(rank={rank}, n_faces={n_faces or L}, transform={transform}, "
        f"init_from_weights={init_from_weights}). "
        f"Trainable params: {sum(p.numel() for p in adapter.parameters()):,}"
    )
    return adapter


# ---------------------------------------------------------------------------
# Efficient forward: avoid materialising ΔW when d is large
# ---------------------------------------------------------------------------

class MTLoRALinearFast(nn.Module):
    """
    Like MTLoRALinear but never materialises the full d_out × d_in ΔW matrix.
    Instead it computes  y_adapter = Σ_f c_{ℓ,f} · Â_f (B̂_f x)  directly,
    where c_{ℓ,f} = (Mᵀ)_{ℓ,f} = M_{f,ℓ}.

    FLOPs per token: p × r × (d_out + d_in)  vs  d_out × d_in for materialising.
    Preferred when  p × r ≪ min(d_out, d_in).
    """

    def __init__(self, linear: nn.Linear, adapter: MTLoRA, layer_idx: int) -> None:
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        adapter = self.adapter
        # Coefficients c_{ℓ, f} = M^T_{ℓ, f} = Mt[ℓ, :]  (shape p or L)
        # We use Mt (cached as buffer) and slice the kept faces.
        c = adapter.Mt[self.layer_idx, : adapter.n_faces]  # (p,)
        mask = adapter.face_mask.to(c.dtype)               # (p,)
        c = c * mask                                        # zero out pruned faces
        scaling = adapter.scaling

        # x  : (..., d_in)
        # B̂ : (r, d_in, p)  =>  B̂_f @ x.T : (r, batch)  for each f
        # Â : (d_out, r, p)  =>  Â_f @ (B̂_f x) : (d_out, batch)

        # Batch over faces with einsum for clarity and fused CUDA ops:
        # Bx[..., f] = x @ B̂[:,:,f].T  =>  (..., r, p)
        Bx = torch.einsum("...i,rip->...rp", x, adapter.B_hat)  # (..., r, p)
        # ABx[..., f] = Â[:,:,f] @ Bx[...,f]  =>  (..., d_out, p)
        ABx = torch.einsum("orp,...rp->...op", adapter.A_hat, Bx)  # (..., d_out, p)
        # Sum over faces with coefficients
        delta = torch.einsum("...op,p->...o", ABx, c * scaling)  # (..., d_out)
        return base + delta

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}, fast=True"


def wrap_roberta_attention_fast(
    model: nn.Module,
    proj_name: str = "query",
    rank: int = 8,
    n_faces: Optional[int] = None,
    transform: str = "dct",
    alpha: float = 1.0,
    init_from_weights: bool = False,
    a_init: str = "normal",
) -> MTLoRA:
    """Same as wrap_roberta_attention but uses MTLoRALinearFast wrappers."""
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
        raise ValueError(f"No layers matched pattern for proj_name='{proj_name}'.")

    layers_sorted = sorted(found)
    L = len(layers_sorted)
    sample_linear = found[layers_sorted[0]][1]
    d_out, d_in = sample_linear.weight.shape

    adapter = MTLoRA(
        d_out=d_out, d_in=d_in, L=L, rank=rank, n_faces=n_faces,
        transform=transform, alpha=alpha, init_from_weights=init_from_weights,
        a_init=a_init,
    )

    if init_from_weights:
        W0 = torch.stack(
            [found[idx][1].weight.detach().float() for idx in layers_sorted], dim=2
        )
        adapter.init_from_stacked(W0)

    for local_idx, layer_idx in enumerate(layers_sorted):
        full_name, linear = found[layer_idx]
        wrapper = MTLoRALinearFast(linear, adapter, layer_idx=local_idx)
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    print(
        f"[MTLoRA-fast] wrapped {L} '{proj_name}' layers "
        f"(rank={rank}, n_faces={n_faces or L}, transform={transform}, "
        f"init_from_weights={init_from_weights}). "
        f"Trainable params: {sum(p.numel() for p in adapter.parameters()):,}"
    )
    return adapter
