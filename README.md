# ⋆<sub>M</sub>-LoRA: M-Product Tensor Low-Rank Adaptation

This project is about **⋆<sub>M</sub>-LoRA**, a parameter-efficient fine-tuning (PEFT) method that lifts the LoRA
adapter from a per-layer matrix factorization to a **cross-layer tensor factorization** based on the tensor **M-product** algebra.

The adapters of the *L* same-type attention projections (e.g. all Q-projections) are stacked along a depth axis into order-3 tensors
`A ∈ R^{d_out × r × L}` and `B ∈ R^{r × d_in × L}`. The update of layer ℓ is the ℓ-th frontal slice of

```
ΔW = A ⋆_M B = M⁻¹ ×₃ facewise( M ×₃ A, M ×₃ B )
```

where `M` is an orthogonal `L × L` transform (DCT by default). Every transform-domain *face* gets its own low-rank factor pair, and the
truncated tensor M-SVD (tSVD-M) is used for initialization and face-level rank pruning. A **learnable orthogonal M** (Cayley parametrisation) is also provided.
Setting `M = I` with one face per layer recovers standard per-layer LoRA.

The method is evaluated on GLUE (CoLA, MRPC, RTE, SST-2, plus QNLI/MNLI) with RoBERTa-base, and on deeper decoder models (Qwen2.5-1.5B / 7B),
against LoRA and a LoTR (Tucker-2) baseline. The write-up (`paper/`) reports largely **negative** results: coupling layers through a fixed DCT,
a random orthogonal `M`, or a learned `M` is statistically indistinguishable from standard LoRA at equal budget.

## Repository layout

```
.
├── README.md
├── requirements.txt          Python dependencies
├── paper/                    LaTeX source of the paper (ICLR 2025 style)
└── python_code/              All code (run scripts from inside this folder)
    ├── tensor_utils/         Tensor algebra library
    └── ...                   Method, experiments, analysis, legacy tests
```

### `paper/`

| File | Description |
|---|---|
| `main.tex`, `references.bib` | Paper source and bibliography |
| `iclr2025_conference.sty`, `.bst` | ICLR 2025 style files |
| `Makefile` | Builds `main.pdf` |
| `main.pdf` | Compiled paper |
| `figures/` | `depth_structure.png`, `pareto_curves.png` |

### `python_code/tensor_utils/` — tensor algebra

| File | Description |
|---|---|
| `modek_product.py` | Mode-k product, mode-k fold / unfold |
| `m_product.py` | M-product (`mprod`), tensor transpose (`mtran`), facewise product, identity tensor |
| `transformation_matrices.py` | Orthogonal transforms `M`: DCT, DFT, random orthogonal |
| `tSVD.py` | Tensor SVD (t-SVD) and projector construction |
| `tSVD_M.py` | Tensor SVD under an arbitrary orthogonal `M` (tSVD-M) |
| `tensorPOD.py`, `tensorPOD_M.py` | Tensor proper orthogonal decomposition built on t-SVD / tSVD-M |
| `Tucker_projector.py` | Truncated Tucker projector |
| `utils.py` | Placeholder (empty) |
| `__init__.py` | Exports `modek_product`, `mprod`, `facewise`, `dct_matrix`, `random_orthogonal`, ... |

### `python_code/` — method implementation

| File | Description |
|---|---|
| `mt_lora.py` | Core ⋆<sub>M</sub>-LoRA adapter (`MTLoRA`, `MTLoRALinear`), tSVD-M initialization from pretrained weights, dynamic face pruning, fast batched RoBERTa attention wrapping |
| `mt_lora_learnable_M.py` | Variant with a trainable orthogonal `M` via the Cayley map `M(S) = (I−S)(I+S)⁻¹`, with optional freeze schedule |

### `python_code/` — experiments (GLUE fine-tuning)

| File | Description |
|---|---|
| `mt_lora_demo.py` | Fine-tune RoBERTa-base on GLUE with ⋆<sub>M</sub>-LoRA; `--transform identity --n_faces L` gives standard LoRA |
| `mt_lora_learnable_M_demo.py` | Same, with the learnable Cayley `M` (separate learning rate for `M`, optional freeze phase) |
| `mt_lora_llm.py` | ⋆<sub>M</sub>-LoRA / LoRA on decoder-only LLMs (Llama / Qwen2 / Mistral naming) for GLUE classification |
| `lotr_baseline.py` | LoTR-style Tucker-2 PEFT baseline ([Bershatsky et al., 2024](https://arxiv.org/abs/2402.01376)) |

### `python_code/` — ΔW structure analysis

| File | Description |
|---|---|
| `delta_w_extract.py` | Full fine-tune RoBERTa on a GLUE task and save the stacked updates `ΔW = W_ft − W_pre` for q/k/v/o |
| `delta_w_compare.py` | Compare decompositions of `ΔW` (per-layer SVD, Tucker-2, Tucker-3, CP, t-SVD with DCT / random / identity `M`) by approximation error vs. parameter count; depth-structure diagnostics |
| `MTLoRA_Analysis.ipynb` | Analysis notebook: ΔW decomposition benchmark, depth-structure plots, adapter tests |

### `python_code/` — result aggregation

| File | Description |
|---|---|
| `collect_results.py` | Aggregate result `.pt` files into mean ± std tables per (method, task); optional LaTeX output |
| `collect_ablations.py` | Aggregate the CoLA ablations and the RTE learning-rate check |
| `collect_regimes.py` | Aggregate the "regimes where coupling could matter" runs (small adapters, larger datasets, deeper models) |

### `python_code/` — earlier exploration / legacy

| File | Description |
|---|---|
| `TSVD_test.py`, `TSVD_M_test.py` | Tests of t-SVD / tSVD-M on stacked attention weight tensors (expect `tensor_Q.pt`) |
| `CP_test.py`, `Tucker_test.py` | CP and Tucker decompositions of stacked weight tensors (expect `tensor.pt`) |
| `utils.py` | Helper plotting / unfolding utilities for the tests above |
| `main.py` | Load an LLM and inspect its parameters |
| `vanilla_attention_bert.py` | Baseline BERT attention classifier on IMDB reviews |
| `HyperCube_low_rank_demo.ipynb` | Low-rank structure via weight decay on a HyperCube model |
| `Untitled.ipynb`, `Untitled1.ipynb` | Scratch notebooks (IMDB loading, weight SVD inspection) |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run from inside `python_code/`. Results are written to `results/` (git-ignored).

```bash
cd python_code

# ⋆_M-LoRA with DCT transform, 3 kept faces, W0-guided init
python mt_lora_demo.py --task cola --transform dct --n_faces 3 --rank 8 --init_from_weights

# Standard LoRA (identity transform, one face per layer)
python mt_lora_demo.py --task cola --transform identity --n_faces 12 --rank 8

# Learnable (Cayley) M
python mt_lora_learnable_M_demo.py --task cola --rank 8 --n_faces 12 --lr_M 1e-3 --freeze_M_epochs 2

# LoTR baseline
python lotr_baseline.py --task cola --rank 8 --epochs 10 --lr 5e-4 --seed 0

# Deeper model
python mt_lora_llm.py --model Qwen/Qwen2.5-1.5B --task cola --transform dct --n_faces 28 --rank 8 --out results/x.pt

# ΔW analysis
python delta_w_extract.py --task cola --epochs 3 --out results/dw_cola.pt
python delta_w_compare.py --dw results/dw_cola.pt --proj q v --out results/cmp_cola

# Aggregate results
python collect_results.py
```
