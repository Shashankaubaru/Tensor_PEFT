"""
MTLoRA with Learnable Orthogonal M  (Cayley parametrisation)
=============================================================

This module extends MTLoRA with a **trainable** orthogonal depth-transform matrix M,
parametrised via the Cayley map:

    M(S)  =  (I − S)(I + S)⁻¹

where S is a skew-symmetric matrix  (S = −Sᵀ).  We store the upper-triangle
of S as a free parameter vector of length  L*(L−1)/2  and reconstruct S on every
forward call.  Because M is always orthogonal by construction, Parseval's theorem
holds throughout training and the tSVD-M theoretical guarantees remain intact.

Key design choices
------------------
1.  **Cayley map over exponential map**: the Cayley map avoids the matrix-exponential
    computation (no series truncation, no eigendecomposition).  A single
    `torch.linalg.solve((I+S), (I−S))` gives M exactly in O(L³) operations.
    For L=12 (RoBERTa-base) this is negligible.

2.  **S is initialised to zero** → M(0) = I (identity transform) at step 0.
    This mirrors the zero-delta start of standard LoRA and enables warm-starting
    from any fixed-M checkpoint.

3.  **Shared across layers**: One CayleyM instance is shared by all MTLoRALinear
    wrappers in a family.  Gradients from all L layers propagate through the same S,
    giving a well-conditioned signal.

4.  **Optional freeze schedule**: call  `adapter.freeze_M(True/False)`  to freeze/
    unfreeze M mid-training.  This enables a two-phase schedule:
        Phase A (epochs 1–k):  train Â, B̂ with M=I (warm-up, no Cayley overhead)
        Phase B (epochs k+1…): unfreeze S, jointly optimise Â, B̂, M.

Usage
-----
    from mt_lora_learnable_M import wrap_roberta_learnable_M
    adapter = wrap_roberta_learnable_M(model, proj_name="query", rank=8)
    # adapter.cayley gives access to the CayleyM module
    # adapter.cayley.S is the skew-symmetric parameter

References
----------
- Kilmer & Martin (2011). Factorization strategies for third-order tensors.
- Kernfeld, Kilmer & Aeron (2015). Tensor–tensor products with invertible linear transforms.
- Wen & Yin (2013).  A feasible method for optimization with orthogonality constraints.
  (Cayley transform on the Stiefel manifold)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from mt_lora import MTLoRA, _modek3


# ---------------------------------------------------------------------------
# Cayley parametrisation of SO(L)
# ---------------------------------------------------------------------------

class CayleyM(nn.Module):
    """
    Trainable orthogonal L×L matrix via Cayley map: M = (I−S)(I+S)⁻¹.

    S is skew-symmetric.  We store its upper-triangle entries as a free
    parameter and build S on every forward call.

    The matrix M and its transpose Mt = M⊤ are returned as plain tensors
    (not Parameters) so they can be used as buffers in einsum operations.

    Parameters
    ----------
    L : int
        Size of the square matrix.
    init : str
        'zero' (default) → S=0, M=I at init.
        'random' → small random skew-symmetric init (σ=0.01).
    dtype : torch.dtype
    """

    def __init__(
        self,
        L: int,
        init: str = "zero",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.L = L
        n_params = L * (L - 1) // 2        # upper-triangle entries of S

        # Store the upper-triangle of S as a flat parameter vector
        if init == "zero":
            data = torch.zeros(n_params, dtype=dtype)
        elif init == "random":
            data = torch.randn(n_params, dtype=dtype) * 0.01
        else:
            raise ValueError(f"Unknown init '{init}'. Use 'zero' or 'random'.")

        self.S_upper = nn.Parameter(data)

        # Pre-compute index arrays for building S from S_upper (cheaper than triu_indices
        # on every forward call when L is small; recomputed once at init)
        rows, cols = torch.triu_indices(L, L, offset=1)
        self.register_buffer("rows", rows)  # (n_params,)
        self.register_buffer("cols", cols)  # (n_params,)

        # Cache identity for convenience
        self.register_buffer("eye", torch.eye(L, dtype=dtype))

    def build_skew(self) -> torch.Tensor:
        """Reconstruct the full skew-symmetric matrix S from S_upper."""
        S = torch.zeros(self.L, self.L, dtype=self.S_upper.dtype,
                        device=self.S_upper.device)
        S[self.rows, self.cols] = self.S_upper
        S = S - S.T                          # enforce exact skew-symmetry
        return S                             # (L, L)

    def forward(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute M = (I−S)(I+S)⁻¹  via a linear solve.

        Returns
        -------
        M  : (L, L)  orthogonal
        Mt : (L, L)  = M⊤ = M⁻¹  (orthogonal)
        """
        S = self.build_skew()                        # (L, L)
        I = self.eye.to(S.device)                    # (L, L)
        IpS = I + S                                  # (I + S)
        ImS = I - S                                  # (I − S)
        # M = ImS @ IpS⁻¹  ⟺  M @ IpS = ImS  ⟺  IpS⊤ @ M⊤ = ImS⊤
        # torch.linalg.solve(A, B) solves A X = B, so:
        #   solve(IpS⊤, ImS⊤)  returns M⊤
        Mt = torch.linalg.solve(IpS.T, ImS.T)        # (L, L) = M⊤
        M  = Mt.T                                     # (L, L)
        return M, Mt

    def extra_repr(self) -> str:
        return f"L={self.L}, n_params={self.S_upper.numel()}"


def cayley_coords_from_orthogonal(M: torch.Tensor) -> torch.Tensor:
    """
    Upper-triangle of the skew-symmetric S with (I - S)(I + S)^{-1} = M, i.e. S = (I - M)(I + M)^{-1}.
    Used to start the learnable M at a fixed transform (e.g. the DCT) instead of at the identity.

    The Cayley map reaches only matrices with det = +1 and no eigenvalue -1.  If det(M) = -1 the
    last row is negated (a sign flip of the last face, which only matters when that face is kept).
    Raises if I + M is too ill-conditioned for a float32 solve.
    """
    M = M.double()
    L = M.shape[0]
    I = torch.eye(L, dtype=M.dtype, device=M.device)
    if torch.linalg.det(M) < 0:
        M = M.clone()
        M[-1] = -M[-1]
    if torch.linalg.cond(I + M) > 1e4:
        raise ValueError("I + M is ill-conditioned; M is too close to the Cayley-map boundary")
    S = (I - M) @ torch.linalg.inv(I + M)
    rows, cols = torch.triu_indices(L, L, offset=1)
    return S[rows, cols]


# ---------------------------------------------------------------------------
# Adapter module with learnable M
# ---------------------------------------------------------------------------

class MTLoRALearnableM(nn.Module):
    """
    MTLoRA with a learnable orthogonal depth-transform M (Cayley parametrisation).

    The interface mirrors MTLoRA closely.  The main difference is that M is not a
    fixed buffer but a CayleyM sub-module.  The forward pass calls cayley() to get
    the current M and Mt, then proceeds exactly as in MTLoRALinearFast.

    Parameters
    ----------
    d_out, d_in : int
    L : int          Number of layers.
    rank : int       M-tubal rank.
    n_faces : int    Kept transform-domain faces (p ≤ L).
    alpha : float
    init_from_weights : bool
    freeze_M_epochs : int
        If > 0, M is frozen (S_upper.requires_grad = False) for the first
        `freeze_M_epochs` epochs.  Use adapter.step_epoch() to advance the counter.
    dtype : torch.dtype
    """

    def __init__(
        self,
        d_out: int,
        d_in: int,
        L: int,
        rank: int = 8,
        n_faces: Optional[int] = None,
        alpha: float = 1.0,
        init_from_weights: bool = False,
        freeze_M_epochs: int = 2,
        dtype: torch.dtype = torch.float32,
        a_init: str = "normal",
        m_init: str = "identity",
    ) -> None:
        super().__init__()
        if a_init not in ("normal", "lora"):
            raise ValueError(f"a_init must be 'normal' or 'lora', got {a_init!r}")
        if m_init not in ("identity", "dct"):
            raise ValueError(f"m_init must be 'identity' or 'dct', got {m_init!r}")
        self.a_init = a_init
        self.m_init = m_init
        self.d_out = d_out
        self.d_in = d_in
        self.L = L
        self.rank = rank
        self.n_faces = min(n_faces, L) if n_faces is not None else L
        self.alpha = alpha
        self.scaling = alpha / rank
        self.init_from_weights = init_from_weights
        self.freeze_M_epochs = freeze_M_epochs
        self._epoch = 0

        # Learnable M: S = 0 (M = I) by default, or the Cayley coordinates of the DCT matrix.
        self.cayley = CayleyM(L, init="zero", dtype=dtype)
        if m_init == "dct":
            from mt_lora import _make_M
            M0 = _make_M("dct", L, dtype=torch.float64, device=torch.device("cpu"))
            self.cayley.S_upper.data.copy_(cayley_coords_from_orthogonal(M0).to(dtype))
        with torch.no_grad():
            self.register_buffer("M_init", self.cayley()[0].detach().clone())

        # Adapter parameters (transform-domain faces)
        p = self.n_faces
        self.A_hat = nn.Parameter(torch.empty(d_out, rank, p, dtype=dtype))
        self.B_hat = nn.Parameter(torch.empty(rank, d_in, p, dtype=dtype))

        # face_mask: active faces (for pruning)
        self.register_buffer("face_mask", torch.ones(p, dtype=torch.bool))
        self.register_buffer("face_energy", torch.zeros(p, dtype=dtype))

        if not init_from_weights:
            self._random_init()

        # Apply initial M freeze if requested
        if freeze_M_epochs > 0:
            self.freeze_M(True)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _random_init(self) -> None:
        """
        a_init='normal' (default): A_hat ~ N(0,1), B_hat = 0.
        a_init='lora': Hu et al. LoRA init, identical to MTLoRA(a_init='lora') in mt_lora.py:
            B_hat ~ U(+-1/sqrt(d_in)), A_hat = 0.
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
        Phase-1 init from stacked pretrained weights W0 : (d_out, d_in, L).

        Uses M = I (since CayleyM starts at zero → M=I) for the initial transform.
        After M begins to rotate during training, the original alignment is softly
        preserved by the gradient signal.
        """
        assert W0.shape == (self.d_out, self.d_in, self.L), (
            f"Expected W0 shape ({self.d_out}, {self.d_in}, {self.L}), got {tuple(W0.shape)}"
        )
        W0_f = W0.float()
        # At init, M=I so the transform domain = spatial domain
        p = self.n_faces
        A_init = torch.zeros_like(self.A_hat)
        for f in range(p):
            face = W0_f[:, :, f]
            U, S, Vh = torch.linalg.svd(face, full_matrices=False)
            r_eff = min(self.rank, U.shape[1])
            A_init[:, :r_eff, f] = U[:, :r_eff].to(self.A_hat.dtype)

        self.A_hat.data.copy_(A_init)
        nn.init.zeros_(self.B_hat)
        print(f"[MTLoRALearnableM] init_from_stacked done (L={self.L}, rank={self.rank})")

    # ------------------------------------------------------------------
    # M freeze / unfreeze schedule
    # ------------------------------------------------------------------

    def freeze_M(self, freeze: bool = True) -> None:
        """Freeze (True) or unfreeze (False) the Cayley S_upper parameter."""
        self.cayley.S_upper.requires_grad_(not freeze)

    def step_epoch(self) -> None:
        """
        Call once per epoch to advance the epoch counter and potentially
        unfreeze M after `freeze_M_epochs` warm-up epochs.
        """
        self._epoch += 1
        if self._epoch >= self.freeze_M_epochs:
            self.freeze_M(False)
            print(f"[MTLoRALearnableM] epoch {self._epoch}: M unfrozen (Cayley S now trainable)")

    # ------------------------------------------------------------------
    # Current M (recomputed on each forward call)
    # ------------------------------------------------------------------

    def get_M(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (M, Mt) from the current Cayley state."""
        return self.cayley()

    # ------------------------------------------------------------------
    # Compute ΔW in the spatial domain
    # ------------------------------------------------------------------

    def get_delta_W(self) -> torch.Tensor:
        """
        Compute the full ΔW : (d_out, d_in, L) for all layers.
        Uses the current M from the Cayley parametrisation.
        """
        M, Mt = self.get_M()
        mask = self.face_mask.to(self.A_hat.dtype)

        A = self.A_hat * mask[None, None, :]   # (d_out, r, p)
        B = self.B_hat * mask[None, None, :]   # (r, d_in, p)

        # Facewise product: ΔŴ[:,:,f] = A[:,:,f] @ B[:,:,f]
        # A permuted: (p, d_out, r),  B permuted: (p, r, d_in)
        A_p = A.permute(2, 0, 1)               # (p, d_out, r)
        B_p = B.permute(2, 0, 1)               # (p, r, d_in)
        DW_hat_p = torch.bmm(A_p, B_p)         # (p, d_out, d_in)
        DW_hat_p = DW_hat_p.permute(1, 2, 0)   # (d_out, d_in, p)

        # Embed into L-face tensor
        if self.n_faces < self.L:
            DW_hat = torch.zeros(
                self.d_out, self.d_in, self.L,
                dtype=DW_hat_p.dtype, device=DW_hat_p.device
            )
            DW_hat[:, :, : self.n_faces] = DW_hat_p
        else:
            DW_hat = DW_hat_p

        # Inverse transform: ΔW = ΔŴ ×₃ M⁻¹ = ΔŴ ×₃ Mᵀ
        # _modek3(X, M, inverse=True) computes X @ M  (because Mᵀᵀ = M)
        # We need  X @ Mᵀ  which is _modek3(X, M, inverse=False)
        # Re-check convention: _modek3(X, M, inverse=False) → X @ Mᵀ
        #                      _modek3(X, M, inverse=True)  → X @ M
        # For the inverse transform we want ΔŴ ×₃ Mᵀ = ΔŴ @ Mᵀ along axis 2
        # i.e. reshape to (..., L) and post-multiply by Mᵀ.
        # Vectorised: DW_hat.reshape(-1, L) @ Mt
        L = self.L
        DW = DW_hat.reshape(-1, L) @ Mt         # (..., L) @ (L, L) = (..., L)
        DW = DW.reshape(self.d_out, self.d_in, L)
        return DW

    # ------------------------------------------------------------------
    # Dynamic face pruning (same as MTLoRA)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def prune_dead_faces(self, threshold: float = 1e-4) -> int:
        n_pruned = 0
        for f in range(self.n_faces):
            if not self.face_mask[f]:
                continue
            e = float(self.B_hat[:, :, f].detach().norm(p="fro"))
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
        AB_params = active * self.rank * (self.d_out + self.d_in)
        M_params = self.cayley.S_upper.numel() if self.cayley.S_upper.requires_grad else 0
        return AB_params + M_params

    def extra_repr(self) -> str:
        return (
            f"d_out={self.d_out}, d_in={self.d_in}, L={self.L}, "
            f"rank={self.rank}, n_faces={self.n_faces}, alpha={self.alpha}, "
            f"freeze_M_epochs={self.freeze_M_epochs}"
        )


# ---------------------------------------------------------------------------
# Per-layer wrapper (fast, no ΔW materialisation)
# ---------------------------------------------------------------------------

class MTLoRALinearLearnableM(nn.Module):
    """
    Wraps one nn.Linear layer with an MTLoRALearnableM adapter.

    The forward pass computes  y = W₀x + α/r · Σ_f (Mt)_{ℓ,f} · Â_f(B̂_f x)
    where Mt = M⊤ is recomputed from the Cayley module on each forward call.

    The CayleyM.forward() is called once per *adapter family* (not once per layer)
    by passing the (M, Mt) tensors from the adapter into each MTLoRALinearLearnableM
    wrapper.  This is done in the companion `MTLoRAFamilyWrapper` class below,
    but for simplicity this class recomputes M on every call.  For L ≤ 32 the
    overhead is negligible (L³ ≈ 32,000 FLOPs).
    """

    def __init__(
        self,
        linear: nn.Linear,
        adapter: MTLoRALearnableM,
        layer_idx: int,
    ) -> None:
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.adapter = adapter
        self.layer_idx = layer_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        adapter = self.adapter
        M, Mt = adapter.get_M()                               # (L, L), (L, L)

        # Coefficients for layer ℓ: c_f = Mt[ℓ, f]  for f in kept faces
        c = Mt[self.layer_idx, : adapter.n_faces]             # (p,)
        mask = adapter.face_mask.to(c.dtype)
        c = c * mask                                          # zero pruned faces

        # Batched einsum over faces (same as MTLoRALinearFast)
        Bx = torch.einsum("...i,rip->...rp", x, adapter.B_hat)    # (..., r, p)
        ABx = torch.einsum("orp,...rp->...op", adapter.A_hat, Bx)  # (..., d_out, p)
        delta = torch.einsum("...op,p->...o", ABx, c * adapter.scaling)  # (..., d_out)
        return base + delta

    def extra_repr(self) -> str:
        return f"layer_idx={self.layer_idx}, learnable_M=True"


# ---------------------------------------------------------------------------
# Convenience wrapper: replace RoBERTa attention projections
# ---------------------------------------------------------------------------

def wrap_roberta_learnable_M(
    model: nn.Module,
    proj_name: str = "query",
    rank: int = 8,
    n_faces: Optional[int] = None,
    alpha: float = 1.0,
    init_from_weights: bool = False,
    freeze_M_epochs: int = 2,
    a_init: str = "normal",
    m_init: str = "identity",
) -> MTLoRALearnableM:
    """
    Replace all `proj_name` projections in a RoBERTa encoder with
    MTLoRALinearLearnableM wrappers backed by a single MTLoRALearnableM adapter.

    Parameters
    ----------
    model : nn.Module
        A RoBERTa model (HuggingFace AutoModelForSequenceClassification or similar).
    proj_name : str
        One of 'query', 'key', 'value', 'dense' (output projection).
    rank : int
    n_faces : int or None
    alpha : float
    init_from_weights : bool
        If True, initialise Â from tSVD of W₀ (with M=I at init).
    freeze_M_epochs : int
        Freeze M for this many epochs before jointly optimising.
        0 = always train M; 1 = freeze only epoch 0; etc.

    Returns
    -------
    adapter : MTLoRALearnableM
        The shared adapter (contains CayleyM and A_hat, B_hat).
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
            f"No layers matched the RoBERTa pattern for proj_name='{proj_name}'. "
            f"Available linear modules: "
            + str([n for n, _ in model.named_modules() if isinstance(_, nn.Linear)])[:300]
        )

    layers_sorted = sorted(found)
    L = len(layers_sorted)
    sample_linear = found[layers_sorted[0]][1]
    d_out, d_in = sample_linear.weight.shape

    adapter = MTLoRALearnableM(
        d_out=d_out,
        d_in=d_in,
        L=L,
        rank=rank,
        n_faces=n_faces,
        alpha=alpha,
        init_from_weights=init_from_weights,
        freeze_M_epochs=freeze_M_epochs,
        a_init=a_init,
        m_init=m_init,
    )

    if init_from_weights:
        W0 = torch.stack(
            [found[idx][1].weight.detach().float() for idx in layers_sorted], dim=2
        )
        adapter.init_from_stacked(W0)

    # Replace each linear with its wrapper
    for local_idx, layer_idx in enumerate(layers_sorted):
        full_name, linear = found[layer_idx]
        wrapper = MTLoRALinearLearnableM(linear, adapter, layer_idx=local_idx)
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], wrapper)

    # Summary
    ab_params = sum(p.numel() for p in [adapter.A_hat, adapter.B_hat])
    m_params = adapter.cayley.S_upper.numel()
    print(
        f"[MTLoRA-LearnableM] wrapped {L} '{proj_name}' layers "
        f"(rank={rank}, n_faces={n_faces or L}, alpha={alpha}, "
        f"freeze_M_epochs={freeze_M_epochs}).\n"
        f"  A+B params : {ab_params:,}\n"
        f"  Cayley S   : {m_params:,}  (frozen for first {freeze_M_epochs} epoch(s))\n"
        f"  Total      : {ab_params + m_params:,}"
    )
    return adapter
