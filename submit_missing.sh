#!/bin/bash
# =============================================================================
# submit_missing.sh — submit the remaining experiments for the paper (GPU on CCC)
#
# Audited 2026-10-25. The 4-task headline table (LoTR r8 / Identity / Cayley) is
# complete. These groups add the experiments that the claims still depend on:
#
#   matched   LoTR rank 64 on CoLA/MRPC/RTE/SST-2, seeds 0-2 (12 jobs).
#             LoTR r=64 has 147,456 params per adapter = 294,912 total, exactly the
#             adapter size of Identity p=12 r=8, so the comparison is parameter-matched.
#             alpha=32 keeps scaling alpha/rank = 0.5, the same as alpha=4 at r=8.
#   seeds     Seeds 3,4 for Identity and Cayley on CoLA (5 ep and 10 ep) and RTE, plus
#             Identity CoLA 10 ep seed 42 to match Cayley's seed set (13 jobs).
#             Tests whether the small Cayley-vs-Identity gaps are real.
#   rtelr     RTE learning-rate check: LoTR r8 at 3e-4 and 1e-3, Identity at 3e-4,
#             seeds 0-2 (9 jobs). LoTR barely trains at the headline lr of 1e-4.
#   ablation  CoLA 5-epoch ablations with 3 seeds each (21 jobs): transform
#             (DCT, random at p=12), n_faces p in {1,2,4,6} (DCT), and W0-guided init
#             (p=4). Identity p=12 already has 3 seeds in the headline table.
#   extra     Follow-ups after the first ablation results (28 jobs):
#             - CoLA 5-epoch seeds 3,4 for DCT p=12, random p=12, DCT p=4 and DCT p=4 with
#               W0-guided init (8 jobs). Tests the borderline DCT/random-vs-Identity gap and
#               gives the init comparison seeds 0-4 like the Identity row. (The W0-guided
#               seed 0 job is part of "ablation".)
#             - LoTR r64 (alpha 32) CoLA 10 epochs, seeds 0-4 (5 jobs): counterpart to the
#               10-epoch Identity and Cayley rows.
#             - RTE learning-rate grid extension (all seeds 0-2): LoTR r8 at 3e-3 and 1e-2
#               (6 jobs; it was still improving at 1e-3), LoTR r64 at 3e-4 and 1e-3 (6 jobs),
#               Identity at 1e-3 (3 jobs).
#             Total 8 + 5 + 6 + 6 + 3 = 28 jobs.
#
#   round3    Third round (76 jobs), added after the second sync. NOT part of "all":
#             you must name it. Every job here fills a gap listed in the paper's Limitations.
#             A. lora    Standard LoRA baseline = Identity (p=L, M=I) with the Hu et al. init
#                        (--a_init lora: B Kaiming-uniform, A = 0). Headline settings, so it is
#                        directly comparable with the Identity rows: CoLA 5 ep seeds 0-4, MRPC
#                        0-2, RTE (10 ep, lr 1e-4) 0-4, SST-2 0-2, CoLA 10 ep 0-4 (21 jobs).
#             B. cola10  DCT and random M at p=12, CoLA 10 epochs, seeds 0-4 (10 jobs): does the
#                        5-epoch advantage over Identity (10 ep: 0.548) survive convergence?
#             C. mix     DCT and random M at p=12 on MRPC (5 ep) and RTE (10 ep, lr 1e-4),
#                        seeds 0-2 (12 jobs): does layer mixing help beyond CoLA?
#             D. lr      Learning-rate checks (27 jobs). CoLA 5 ep, seeds 0-2: LoTR rank 64 and
#                        Identity at 2e-4, 1e-3, 2e-3 (18 jobs; the 5-point CoLA gap to LoTR-64 uses
#                        one lr). RTE 10 ep, seeds 0-2: Identity at 3e-5 and 5e-5, LoTR rank 64 at
#                        5e-4 (9 jobs; Identity's best lr may lie below 1e-4).
#             E. fewl    Identity (M = I) with only the first p layers adapted, p in {4,6}, CoLA
#                        5 ep, seeds 0-2 (6 jobs). Same adapter size as DCT p=4 / p=6, so it
#                        separates "fewer faces" from "DCT low-pass". NOTE: with M = I, face f is
#                        layer f, so layers >= p get no adapter.
#             Total 21 + 10 + 12 + 27 + 6 = 76 jobs. Results:  python collect_results.py
#             (LoRA rows) and  python collect_ablations.py  (everything else).
#
#   round4    Initialisation-matched comparison (NOT part of "all"; name it). Round 3 showed
#             that the N(0,1)-initialised Identity is a weak baseline: standard LoRA (Hu et al.
#             init, i.e. the SAME M=I, p=L adapter) is 3.9-4.2 points better on CoLA, so every
#             DCT / random / Cayley row so far was compared at an unfair initialisation. In round4
#             every method uses --a_init lora (B ~ U(+-1/sqrt(d_in)), A = 0), the same lr,
#             epochs, alpha/rank = 0.5 and seeds as the existing LoRA rows (LoTR already uses
#             Kaiming-uniform A, B). The question is simply: does coupling layers through M
#             beat standard LoRA when nothing else differs? The runs can come out either way.
#      round4core   (45 jobs)  CoLA only, seeds 0-4
#             A. DCT and random M, p=12, --a_init lora, 5 and 10 epochs (20 jobs). Compare with
#                lora_p12 / lora_p12_ep10 (0.552 / 0.590).
#             B. Cayley M, p=12, --a_init lora, 5 and 10 epochs (10 jobs).
#             C. Equal-budget test (15 jobs): DCT p=6 and Identity p=6 (first 6 layers only), both
#                --a_init lora, vs standard LoRA rank 4 (alpha 2; same 147K adapter params as
#                p=6). This is the setting where a cross-layer method has the best chance of
#                helping: fewer parameters than LoRA-r8, and it separates "fewer faces" from
#                "low-pass DCT" (the N(0,1) Identity p<L runs did not train at all).
#      round4tasks  (33 jobs)  does it generalise? DCT, random, Cayley with --a_init lora on
#             MRPC (5 ep, seeds 0-2), RTE (10 ep, lr 1e-4, seeds 0-4), SST-2 (5 ep, seeds 0-2).
#             Compare with the LoRA rows of the headline table.
#      round4lr     (39 jobs)  learning-rate controls, so no method is compared at a
#             learning rate that suits another method better.
#             CoLA 5 ep, seeds 0-2: LoRA and DCT (lora init) at 2e-4, 1e-3, 2e-3 (18 jobs).
#             RTE 10 ep, seeds 0-2: LoRA and DCT (lora init) at 3e-4, 1e-3 (12 jobs); LoRA at
#             1e-4 was under-trained (loss 0.61 vs 0.49 for Identity).
#             CoLA 5 ep, seeds 0-2: N(0,1) Identity at 1e-5, 3e-5, 1e-4 (9 jobs): tests whether
#             Identity's deficit is just an oversized step (one Adam step at 5e-4 changes DeltaW
#             about 50x more than standard LoRA's does).
#             Total 45 + 33 + 39 = 117 jobs.  Results:  python collect_results.py (Cayley-lora
#             rows) and  python collect_ablations.py  (everything else).
#
#   round5    Regimes where coupling could matter (NOT part of "all"; name it). Rounds 3-4 found no
#             difference between DCT / random / Cayley M and standard LoRA at matched init, lr and
#             parameter count for RoBERTa-base (L=12) at rank 8. The one place a cross-layer
#             adapter has a structural advantage is a SMALL budget: DCT with p faces and rank r has
#             p*r*(d_out+d_in) parameters but each layer's DeltaW has rank up to p*r, whereas
#             per-layer LoRA with the same parameters has rank (p*r)/L per layer. E.g. DCT p=2, r=6
#             and LoRA r=1 have the same size (12 rank-units), but 12 vs 1 per-layer rank.
#             All runs use --a_init lora and alpha/rank = 0.5.
#      round5budget (162 jobs)  RoBERTa-base, CoLA (5 ep, lr 1e-3) and RTE (10 ep, lr 1e-3), seeds
#             0-4 (150 jobs). Budgets B = 12, 24, 48 rank-units (= LoRA rank 1, 2, 4). Per budget:
#             LoRA (M=I, p=12), DCT p=2 and p=6, random M p=2, Identity p=2 (first two layers only;
#             control). r = B/p so every configuration in a budget has the same number of adapter
#             parameters. Plus a learning-rate check at the smallest budget (12 jobs): CoLA, seeds
#             0-2, LoRA r1 and DCT p2 r6 at 5e-4 and 2e-3 (small adapters often want a larger lr).
#      round5data   (36 jobs)   More data: MNLI (393k, 3 ep) and QNLI (105k, 3 ep), RoBERTa-base,
#             lr 5e-4, seeds 0-2: LoRA r8, DCT p12 r8, LoRA r1, DCT p2 r6, LoRA r2, DCT p2 r12.
#      round5llm    (108 jobs)  Deeper models via mt_lora_llm.py (decoder-only; frozen bf16 base,
#             fp32 adapters). Qwen2.5-1.5B (L=28), CoLA/RTE/MRPC, seeds 0-2, 3 epochs, lr 2e-4:
#             LoRA r8, DCT p28 r8, random p28 r8; LoRA r1 (28 units), DCT p2 r14, DCT p4 r7,
#             random p2 r14, Identity p2 r14 (72 jobs). LR check on CoLA: LoRA r8 and DCT p28 r8 at
#             1e-4 and 5e-4 (12 jobs). Qwen2.5-7B (L=28; needs ~40 GB GPU memory), CoLA/RTE,
#             seeds 0-2: LoRA r8, DCT p28 r8, LoRA r1, DCT p2 r14 (24 jobs).
#             (Llama-3-8B is gated and needs a Hugging Face token; Qwen2.5-7B has the same depth
#             L=28 plus GQA and is openly licensed, so it is used instead.)
#             Total 162 + 36 + 108 = 306 jobs.  Results:  python collect_regimes.py
#
#   round6    Learned (Cayley) M in the regimes of round5 (NOT part of "all"; name it). Round 5 found
#             that fixed DCT / random M never beat LoRA of the same size. A learned M could in
#             principle find a depth basis that suits the model, so we repeat the small-budget and
#             deeper-model comparisons with Cayley M. Same settings as round5 (standard init,
#             alpha/r = 0.5, same lr / epochs / seeds), trained with mt_lora_demo.py / mt_lora_llm.py
#             (--transform cayley), so the lr schedule is the same linear decay as the round5 rows.
#             Variant tags in the output files: cay<i|d><e> where i / d = M starts at the identity /
#             at the DCT matrix and e = 3 or 2 means lr_M = 1e-3 or 1e-2 (S is frozen for the first
#             2 epochs on RoBERTa, 1 epoch on the LLMs). With a DCT start and S frozen, a run is
#             identical to the fixed-DCT run of round5 for the same seed, so the two are paired.
#             In earlier Cayley runs M moved only 0.2-0.5 (Frobenius) from its start, so lr_M = 1e-2
#             is included to let it move.
#      round6budget (90 jobs)  RoBERTa-base, CoLA (5 ep) and RTE (10 ep), lr 1e-3, seeds 0-4,
#             p = 2 at budgets 12, 24, 48 rank-units (r = 6, 12, 24): cayd3, cayd2, cayi2.
#             Compare with LoRA / DCT p=2 / Identity p=2 of round5budget.
#      round6llm    (57 jobs)  Qwen2.5-1.5B, CoLA / RTE / MRPC, seeds 0-2 (45 jobs): cayd3, cayd2, cayi2
#             at p=2, r=14 (28 units) and cayi3, cayd3 at p=28, r=8 (224 units). Qwen2.5-7B, CoLA /
#             RTE, seeds 0-2 (12 jobs): cayd2 at p=2, r=14 and cayi3 at p=28, r=8.
#             Total 90 + 57 = 147 jobs.   Results:  python collect_regimes.py
#
#   GPU jobs run with REQUIRE_CUDA=1: a job placed on a node without a GPU exits at once instead
#   of running on the CPU until the wall limit (as happened to rl15_cola_lora_r1_s1).
#
# Usage on CCC (GPU):
#   cd ~/Tensor_PEFT
#   unset HF_HOME
#   export CCC_USE_GPU=1
#   bash submit_missing.sh round3           # or: matched | seeds | rtelr | ablation | extra | all
#   DRY_RUN=1 bash submit_missing.sh round3 # print the bsub commands only
#   ("all" = matched+seeds+rtelr+ablation+extra, already run; it does NOT include round3.)
#   ONLY="lotr64_cola_s0 lotr64_cola_s1" bash submit_missing.sh matched   # named jobs only
#
# Prerequisites:
#   pip install -r requirements.txt  (once)
#   HF_HOME must be unset (default ~/.cache/huggingface) or point to a writable
#   directory. /scratch is NOT writable for this account.
# =============================================================================

set -euo pipefail

GROUP="${1:-}"
case "$GROUP" in
    matched|seeds|rtelr|ablation|extra|round3|round4|round4core|round4tasks|round4lr|round5|round5budget|round5data|round5llm|round6|round6budget|round6llm|all) ;;
    *)
        echo "Usage: bash submit_missing.sh matched|seeds|rtelr|ablation|extra|round3|round4[core|tasks|lr]|round5[budget|data|llm]|round6[budget|llm]|all" >&2
        exit 1
        ;;
esac
# "all" covers the first five groups only; round3 must be requested by name.
want() { [[ "$GROUP" == "$1" || ( "$GROUP" == "all" && "$1" != "round3" ) ]]; }
# round4 sub-groups: "round4" runs all three; round4core / round4tasks / round4lr run one.
want4() { [[ "$GROUP" == "round4" || "$GROUP" == "round4$1" ]]; }
want5() { [[ "$GROUP" == "round5" || "$GROUP" == "round5$1" ]]; }
want6() { [[ "$GROUP" == "round6" || "$GROUP" == "round6$1" ]]; }
# GPU memory requested per job (-gpu gmem=...). 12G fits RoBERTa-base; round5llm raises it.
JOB_GMEM="12G"
GPU_ENV=""   # "REQUIRE_CUDA=1 " on GPU runs: a job on a CPU-only node then exits at once

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_DIR/results"
LOGS_DIR="$REPO_DIR/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Jobs inherit this shell's environment: fail fast if the HF cache is not writable.
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
if ! mkdir -p "$HF_CACHE_DIR" 2>/dev/null || [[ ! -w "$HF_CACHE_DIR" ]]; then
    echo "ERROR: HF cache '$HF_CACHE_DIR' is not writable. Run: unset HF_HOME" >&2
    exit 1
fi

# CCC has no dedicated GPU queue: GPUs are requested on the normal queue via -gpu.
QUEUE="normal"
USE_GPU="${CCC_USE_GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
if [[ "$USE_GPU" == "1" ]]; then
    GPU_ENV="REQUIRE_CUDA=1 "
    NCPUS=4
    MEM="16GB"
    WALLTIME_SHORT="02:00"   # cola / mrpc / rte on an A100 take ~5-15 min
    WALLTIME_LONG="04:00"    # sst2 (67k examples) is the longest
else
    NCPUS=8
    MEM="32GB"
    WALLTIME_SHORT="08:00"
    WALLTIME_LONG="16:00"
fi

PYTHON="${CCC_PYTHON:-/usr/bin/python3}"

# ---------------------------------------------------------------------------
submit_lsf() {
    local name="$1"; local walltime="$2"; shift 2
    local logfile="$LOGS_DIR/${name}.log"
    local gpu_args=()
    if [[ "$USE_GPU" == "1" ]]; then
        # gmem: only place the job on a GPU with enough free memory. Another job
        # filled most of a shared 80GB card once and caused a CUDA OOM.
        gpu_args=(-gpu "num=1:mode=exclusive_process:gmem=${JOB_GMEM}")
    fi
    if [[ -n "${ONLY:-}" && " $ONLY " != *" $name "* ]]; then
        return
    fi
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "bsub ${gpu_args[*]-} -J $name -q $QUEUE -n $NCPUS -W $walltime ... $PYTHON $*"
        return
    fi
    # bsub -o appends, so clear any old log first
    rm -f "$logfile"
    bsub \
        ${gpu_args[@]+"${gpu_args[@]}"} \
        -J  "$name" \
        -q  "$QUEUE" \
        -n  "$NCPUS" \
        -R  "rusage[mem=$MEM]" \
        -W  "$walltime" \
        -o  "$logfile" \
        -e  "$logfile" \
        "cd $REPO_DIR && ${GPU_ENV}$PYTHON -u $*"
    echo "  Submitted: $name  (wall=$walltime)  →  $logfile"
}

# ---------------------------------------------------------------------------
# matched: LoTR rank 64 (parameter-matched to Identity p=12, r=8)
# ---------------------------------------------------------------------------
if want matched; then
    echo "=== LoTR rank 64, alpha 32 (parameter-matched) ==="
    for seed in 0 1 2; do
        submit_lsf "lotr64_cola_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
            --task cola --rank 64 --alpha 32.0 --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lotr_cola_r64_a32_seed${seed}.pt"
        submit_lsf "lotr64_mrpc_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
            --task mrpc --rank 64 --alpha 32.0 --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lotr_mrpc_r64_a32_seed${seed}.pt"
        submit_lsf "lotr64_rte_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
            --task rte --rank 64 --alpha 32.0 --epochs 10 --lr 1e-4 --seed "$seed" \
            --out "results/lotr_rte_r64_a32_seed${seed}.pt"
        submit_lsf "lotr64_sst2_s${seed}" "$WALLTIME_LONG" lotr_baseline.py \
            --task sst2 --rank 64 --alpha 32.0 --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lotr_sst2_r64_a32_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# seeds: more seeds for the Cayley-vs-Identity comparison
#   mt_lora_learnable_M_demo.py takes --out_dir (not --out) and writes
#   <out_dir>/learnable_M_<task>_r8_nf12_fm2_seed<seed>.pt
# ---------------------------------------------------------------------------
if want seeds; then
    echo "=== Extra seeds: Identity and Cayley ==="
    for seed in 3 4; do
        # CoLA, 5 epochs
        submit_lsf "mtlora_cola_id_r8_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/mtlora_cola_identity_r8_seed${seed}.pt"
        submit_lsf "cayley_cola_ep5_r8_s${seed}" "$WALLTIME_SHORT" mt_lora_learnable_M_demo.py \
            --task cola --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 5 --lr 5e-4 --lr_M 1e-3 --freeze_M_epochs 2 --seed "$seed" \
            --out_dir "results/cayley_cola_ep5_seed${seed}"
        # CoLA, 10 epochs
        submit_lsf "mtlora_cola_id_ep10_r8_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 5e-4 --seed "$seed" \
            --out "results/mtlora_cola_identity_ep10_r8_seed${seed}.pt"
        submit_lsf "cayley_cola_ep10_r8_s${seed}" "$WALLTIME_SHORT" mt_lora_learnable_M_demo.py \
            --task cola --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 5e-4 --lr_M 1e-3 --freeze_M_epochs 2 --seed "$seed" \
            --out_dir "results/cayley_cola_ep10_seed${seed}"
        # RTE, 10 epochs
        submit_lsf "mtlora_rte_id_r8_lr1e4_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 1e-4 --seed "$seed" \
            --out "results/mtlora_rte_identity_r8_lr1e4_seed${seed}.pt"
        submit_lsf "cayley_rte_v2_r8_s${seed}" "$WALLTIME_SHORT" mt_lora_learnable_M_demo.py \
            --task rte --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 1e-4 --lr_M 1e-3 --freeze_M_epochs 2 --seed "$seed" \
            --out_dir "results/cayley_rte_ep10_seed${seed}"
    done
    # Identity CoLA 10 epochs seed 42, so Identity has the same seeds as Cayley (0,1,2,42,3,4)
    submit_lsf "mtlora_cola_id_ep10_r8_s42" "$WALLTIME_SHORT" mt_lora_demo.py \
        --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
        --epochs 10 --lr 5e-4 --seed 42 \
        --out "results/mtlora_cola_identity_ep10_r8_seed42.pt"
fi

# ---------------------------------------------------------------------------
# rtelr: RTE learning-rate check (other-setting rows, not part of the headline table)
# ---------------------------------------------------------------------------
if want rtelr; then
    echo "=== RTE learning-rate check ==="
    for lr in 3e-4 1e-3; do
        tag="${lr/e-/e}"      # 3e-4 -> 3e4, 1e-3 -> 1e3
        for seed in 0 1 2; do
            submit_lsf "lotr_rte_a4_lr${tag}_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
                --task rte --rank 8 --alpha 4.0 --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/lotr_rte_r8_a4_lr${tag}_seed${seed}.pt"
        done
    done
    for seed in 0 1 2; do
        submit_lsf "mtlora_rte_id_lr3e4_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 3e-4 --seed "$seed" \
            --out "results/mtlora_rte_identity_r8_lr3e4_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# ablation: CoLA, 5 epochs, rank 8, alpha 4, lr 5e-4, 3 seeds per row
#   file names start with abl_ so the headline collector ignores them
# ---------------------------------------------------------------------------
if want ablation; then
    echo "=== CoLA ablations (3 seeds) ==="
    for seed in 0 1 2; do
        # transform comparison at p=12 (Identity p=12 is already in the headline table)
        for tr in dct random; do
            submit_lsf "abl_${tr}_p12_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform "$tr" --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_${tr}_p12_seed${seed}.pt"
        done
        # n_faces ablation (DCT, random init)
        for p in 1 2 4 6; do
            submit_lsf "abl_dct_p${p}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform dct --rank 8 --n_faces "$p" --alpha 4.0 \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_dct_p${p}_seed${seed}.pt"
        done
        # W0-guided initialisation (p=4; the random-init side is abl_dct_p4 above)
        submit_lsf "abl_dct_p4_w0init_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform dct --rank 8 --n_faces 4 --alpha 4.0 --init_from_weights \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_dct_p4_w0init_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# extra: follow-ups (see header). Not part of "all" re-runs you have already done:
# "all" re-submits every group, so prefer naming a group.
# ---------------------------------------------------------------------------
if want extra; then
    echo "=== Extra: CoLA ablation seeds 3,4 ==="
    for seed in 3 4; do
        for tr in dct random; do
            submit_lsf "abl_${tr}_p12_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform "$tr" --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_${tr}_p12_seed${seed}.pt"
        done
        submit_lsf "abl_dct_p4_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform dct --rank 8 --n_faces 4 --alpha 4.0 \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_dct_p4_seed${seed}.pt"
        submit_lsf "abl_dct_p4_w0init_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform dct --rank 8 --n_faces 4 --alpha 4.0 --init_from_weights \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_dct_p4_w0init_seed${seed}.pt"
    done

    echo "=== Extra: LoTR rank 64, CoLA 10 epochs ==="
    for seed in 0 1 2 3 4; do
        submit_lsf "lotr64_cola_ep10_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
            --task cola --rank 64 --alpha 32.0 --epochs 10 --lr 5e-4 --seed "$seed" \
            --out "results/lotr_cola_r64_a32_ep10_seed${seed}.pt"
    done

    echo "=== Extra: RTE learning-rate grid extension ==="
    for lr in 3e-3 1e-2; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "lotr_rte_a4_lr${tag}_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
                --task rte --rank 8 --alpha 4.0 --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/lotr_rte_r8_a4_lr${tag}_seed${seed}.pt"
        done
    done
    for lr in 3e-4 1e-3; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "lotr64_rte_lr${tag}_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
                --task rte --rank 64 --alpha 32.0 --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/lotr_rte_r64_a32_lr${tag}_seed${seed}.pt"
        done
    done
    for seed in 0 1 2; do
        submit_lsf "mtlora_rte_id_lr1e3_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 1e-3 --seed "$seed" \
            --out "results/mtlora_rte_identity_r8_lr1e3_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# round3: see header. File naming keeps the headline collector clean:
#   lora_*            -> reported as the "LoRA" rows (a_init = lora is stored in the file)
#   abl_<task>_*      -> collect_ablations.py
#   *_lr<tag>_*       -> "other setting" (lr differs from the headline)
# ---------------------------------------------------------------------------
if want round3; then
    echo "=== round3 A: standard LoRA baseline (Identity, Hu et al. init) ==="
    for seed in 0 1 2 3 4; do
        submit_lsf "lora_cola_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lora_cola_r8_seed${seed}.pt"
        submit_lsf "lora_cola_ep10_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
            --epochs 10 --lr 5e-4 --seed "$seed" \
            --out "results/lora_cola_ep10_r8_seed${seed}.pt"
        submit_lsf "lora_rte_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
            --epochs 10 --lr 1e-4 --seed "$seed" \
            --out "results/lora_rte_r8_seed${seed}.pt"
    done
    for seed in 0 1 2; do
        submit_lsf "lora_mrpc_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task mrpc --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lora_mrpc_r8_seed${seed}.pt"
        submit_lsf "lora_sst2_s${seed}" "$WALLTIME_LONG" mt_lora_demo.py \
            --task sst2 --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/lora_sst2_r8_seed${seed}.pt"
    done

    echo "=== round3 B: DCT / random M at p=12, CoLA 10 epochs ==="
    for seed in 0 1 2 3 4; do
        for tr in dct random; do
            submit_lsf "abl_${tr}_p12_ep10_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform "$tr" --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 10 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_${tr}_p12_ep10_seed${seed}.pt"
        done
    done

    echo "=== round3 C: DCT / random M at p=12 on MRPC and RTE ==="
    for seed in 0 1 2; do
        for tr in dct random; do
            submit_lsf "abl_mrpc_${tr}_p12_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task mrpc --transform "$tr" --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_mrpc_${tr}_p12_seed${seed}.pt"
            submit_lsf "abl_rte_${tr}_p12_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task rte --transform "$tr" --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 10 --lr 1e-4 --seed "$seed" \
                --out "results/abl_rte_${tr}_p12_seed${seed}.pt"
        done
    done

    echo "=== round3 D: learning-rate checks ==="
    for lr in 2e-4 1e-3 2e-3; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "lotr64_cola_lr${tag}_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
                --task cola --rank 64 --alpha 32.0 --epochs 5 --lr "$lr" --seed "$seed" \
                --out "results/lotr_cola_r64_a32_lr${tag}_seed${seed}.pt"
            submit_lsf "mtlora_cola_id_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 5 --lr "$lr" --seed "$seed" \
                --out "results/mtlora_cola_identity_r8_lr${tag}_seed${seed}.pt"
        done
    done
    for lr in 3e-5 5e-5; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "mtlora_rte_id_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/mtlora_rte_identity_r8_lr${tag}_seed${seed}.pt"
        done
    done
    for seed in 0 1 2; do
        submit_lsf "lotr64_rte_lr5e4_s${seed}" "$WALLTIME_SHORT" lotr_baseline.py \
            --task rte --rank 64 --alpha 32.0 --epochs 10 --lr 5e-4 --seed "$seed" \
            --out "results/lotr_rte_r64_a32_lr5e4_seed${seed}.pt"
    done

    echo "=== round3 E: Identity with only the first p layers adapted ==="
    for seed in 0 1 2; do
        for p in 4 6; do
            submit_lsf "abl_id_p${p}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform identity --rank 8 --n_faces "$p" --alpha 4.0 \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_identity_p${p}_seed${seed}.pt"
        done
    done
fi

# ---------------------------------------------------------------------------
# round4: initialisation-matched comparison. See the header for the rationale.
#   File naming keeps collectors clean:
#     abl_<task>_<transform>_p12_lorainit[_ep10]_seed<k>.pt   -> collect_ablations.py rows
#     abl_cola_{dct,identity}_p6_lorainit_seed<k>.pt, abl_cola_lora_r4_seed<k>.pt
#     cayley_lorainit_<task>_ep<E>_seed<k>/learnable_M_*.pt   -> collect_results.py (cayley_lora)
#     lr_<method>_<task>_lr<tag>_seed<k>.pt                   -> lr tables in collect_ablations.py
# ---------------------------------------------------------------------------
if want4 core; then
    echo "=== round4 A: DCT / random M with LoRA init, CoLA 5 and 10 epochs ==="
    for seed in 0 1 2 3 4; do
        for tf in dct random; do
            submit_lsf "r4_${tf}_cola_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform "$tf" --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 5 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_${tf}_p12_lorainit_seed${seed}.pt"
            submit_lsf "r4_${tf}_cola_ep10_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform "$tf" --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 10 --lr 5e-4 --seed "$seed" \
                --out "results/abl_cola_${tf}_p12_lorainit_ep10_seed${seed}.pt"
        done
    done

    echo "=== round4 B: Cayley M with LoRA init, CoLA 5 and 10 epochs ==="
    for seed in 0 1 2 3 4; do
        for ep in 5 10; do
            submit_lsf "r4_cayley_cola_ep${ep}_s${seed}" "$WALLTIME_SHORT" mt_lora_learnable_M_demo.py \
                --task cola --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs "$ep" --lr 5e-4 --lr_M 1e-3 --freeze_M_epochs 2 --seed "$seed" \
                --out_dir "results/cayley_lorainit_cola_ep${ep}_seed${seed}"
        done
    done

    echo "=== round4 C: equal-budget test (p=6 vs LoRA rank 4), LoRA init ==="
    for seed in 0 1 2 3 4; do
        submit_lsf "r4_dct_p6_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform dct --rank 8 --n_faces 6 --alpha 4.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_dct_p6_lorainit_seed${seed}.pt"
        submit_lsf "r4_id_p6_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 8 --n_faces 6 --alpha 4.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_identity_p6_lorainit_seed${seed}.pt"
        submit_lsf "r4_lora_r4_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
            --task cola --transform identity --rank 4 --n_faces 12 --alpha 2.0 --a_init lora \
            --epochs 5 --lr 5e-4 --seed "$seed" \
            --out "results/abl_cola_lora_r4_seed${seed}.pt"
    done
fi

if want4 tasks; then
    echo "=== round4 tasks: DCT / random / Cayley with LoRA init on MRPC, RTE, SST-2 ==="
    # task:epochs:lr:seeds  (same settings as the LoRA rows)
    for spec in "mrpc:5:5e-4:0 1 2" "rte:10:1e-4:0 1 2 3 4" "sst2:5:5e-4:0 1 2"; do
        task="${spec%%:*}"; rest="${spec#*:}"
        ep="${rest%%:*}"; rest="${rest#*:}"
        lr="${rest%%:*}"; seeds="${rest#*:}"
        wt="$WALLTIME_SHORT"; [[ "$task" == "sst2" ]] && wt="$WALLTIME_LONG"
        for seed in $seeds; do
            for tf in dct random; do
                submit_lsf "r4_${tf}_${task}_s${seed}" "$wt" mt_lora_demo.py \
                    --task "$task" --transform "$tf" --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                    --epochs "$ep" --lr "$lr" --seed "$seed" \
                    --out "results/abl_${task}_${tf}_p12_lorainit_seed${seed}.pt"
            done
            submit_lsf "r4_cayley_${task}_s${seed}" "$wt" mt_lora_learnable_M_demo.py \
                --task "$task" --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs "$ep" --lr "$lr" --lr_M 1e-3 --freeze_M_epochs 2 --seed "$seed" \
                --out_dir "results/cayley_lorainit_${task}_ep${ep}_seed${seed}"
        done
    done
fi

if want4 lr; then
    echo "=== round4 lr: CoLA learning-rate check for LoRA and DCT (LoRA init) ==="
    for lr in 2e-4 1e-3 2e-3; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "r4lr_lora_cola_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 5 --lr "$lr" --seed "$seed" \
                --out "results/lr_lora_cola_lr${tag}_seed${seed}.pt"
            submit_lsf "r4lr_dct_cola_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform dct --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 5 --lr "$lr" --seed "$seed" \
                --out "results/lr_dctlorainit_cola_lr${tag}_seed${seed}.pt"
        done
    done

    echo "=== round4 lr: RTE learning-rate check for LoRA and DCT (LoRA init) ==="
    for lr in 3e-4 1e-3; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "r4lr_lora_rte_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task rte --transform identity --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/lr_lora_rte_lr${tag}_seed${seed}.pt"
            submit_lsf "r4lr_dct_rte_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task rte --transform dct --rank 8 --n_faces 12 --alpha 4.0 --a_init lora \
                --epochs 10 --lr "$lr" --seed "$seed" \
                --out "results/lr_dctlorainit_rte_lr${tag}_seed${seed}.pt"
        done
    done

    echo "=== round4 lr: N(0,1) Identity at small learning rates (step-size hypothesis) ==="
    for lr in 1e-5 3e-5 1e-4; do
        tag="${lr/e-/e}"
        for seed in 0 1 2; do
            submit_lsf "r4lr_id_cola_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                --task cola --transform identity --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs 5 --lr "$lr" --seed "$seed" \
                --out "results/mtlora_cola_identity_r8_lr${tag}_seed${seed}.pt"
        done
    done
fi

# ---------------------------------------------------------------------------
# round5: regimes where coupling could matter. See the header for the rationale.
# ---------------------------------------------------------------------------
if want5 budget; then
    echo "=== round5 budget: small adapter budgets, RoBERTa-base, CoLA + RTE ==="
    # spec "name:transform:p:rank" ; alpha = rank/2 so that alpha/rank = 0.5 everywhere.
    # LoRA = identity with p=12 (every layer). id_p2 adapts only layers 0 and 1.
    for B in 12 24 48; do
        rl=$((B / 12))
        cfgs=("lora_r${rl}:identity:12:${rl}"
              "dct_p2_r$((B / 2)):dct:2:$((B / 2))"
              "dct_p6_r$((B / 6)):dct:6:$((B / 6))"
              "rand_p2_r$((B / 2)):random:2:$((B / 2))"
              "id_p2_r$((B / 2)):identity:2:$((B / 2))")
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r name tf p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for task_spec in "cola:5" "rte:10"; do
                task="${task_spec%%:*}"; ep="${task_spec#*:}"
                for seed in 0 1 2 3 4; do
                    submit_lsf "rb_${task}_${name}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                        --task "$task" --transform "$tf" --rank "$r" --n_faces "$p" --alpha "$alpha" \
                        --a_init lora --epochs "$ep" --lr 1e-3 --seed "$seed" \
                        --out "results/reg_budget_${task}_${name}_seed${seed}.pt"
                done
            done
        done
    done
fi

if want5 budget; then
    echo "=== round5 budget: learning-rate check at the smallest budget (CoLA) ==="
    for lr in 5e-4 2e-3; do
        tag="${lr/e-/e}"
        for cfg in "lora_r1:identity:12:1" "dct_p2_r6:dct:2:6"; do
            IFS=: read -r name tf p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rb_cola_${name}_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                    --task cola --transform "$tf" --rank "$r" --n_faces "$p" --alpha "$alpha" \
                    --a_init lora --epochs 5 --lr "$lr" --seed "$seed" \
                    --out "results/reg_budget_cola_${name}_lr${tag}_seed${seed}.pt"
            done
        done
    done
fi

if want5 data; then
    echo "=== round5 data: MNLI and QNLI, RoBERTa-base ==="
    cfgs=("lora_r8:identity:12:8" "dct_p12_r8:dct:12:8"
          "lora_r1:identity:12:1" "dct_p2_r6:dct:2:6"
          "lora_r2:identity:12:2" "dct_p2_r12:dct:2:12")
    for task in qnli mnli; do
        wt="04:00"; [[ "$task" == "mnli" ]] && wt="08:00"
        [[ "$USE_GPU" != "1" ]] && wt="48:00"
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r name tf p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rd_${task}_${name}_s${seed}" "$wt" mt_lora_demo.py \
                    --task "$task" --transform "$tf" --rank "$r" --n_faces "$p" --alpha "$alpha" \
                    --a_init lora --epochs 3 --lr 5e-4 --seed "$seed" \
                    --out "results/reg_data_${task}_${name}_seed${seed}.pt"
            done
        done
    done
fi

if want5 llm; then
    echo "=== round5 llm: Qwen2.5-1.5B (L=28), CoLA / RTE / MRPC ==="
    JOB_GMEM="24G"
    M15="Qwen/Qwen2.5-1.5B"
    cfgs=("lora_r8:identity:28:8" "dct_p28_r8:dct:28:8" "rand_p28_r8:random:28:8"
          "lora_r1:identity:28:1" "dct_p2_r14:dct:2:14" "dct_p4_r7:dct:4:7"
          "rand_p2_r14:random:2:14" "id_p2_r14:identity:2:14")
    for task in cola rte mrpc; do
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r name tf p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rl15_${task}_${name}_s${seed}" "$WALLTIME_SHORT" mt_lora_llm.py \
                    --model "$M15" --task "$task" --transform "$tf" --rank "$r" --n_faces "$p" \
                    --alpha "$alpha" --a_init lora --epochs 3 --lr 2e-4 --seed "$seed" \
                    --out "results/reg_llm15_${task}_${name}_lr2e4_seed${seed}.pt"
            done
        done
    done

    echo "=== round5 llm: learning-rate check, Qwen2.5-1.5B, CoLA ==="
    for lr in 1e-4 5e-4; do
        tag="${lr/e-/e}"
        for cfg in "lora_r8:identity:28:8" "dct_p28_r8:dct:28:8"; do
            IFS=: read -r name tf p r <<< "$cfg"
            for seed in 0 1 2; do
                submit_lsf "rl15_cola_${name}_lr${tag}_s${seed}" "$WALLTIME_SHORT" mt_lora_llm.py \
                    --model "$M15" --task cola --transform "$tf" --rank "$r" --n_faces "$p" \
                    --alpha 4.0 --a_init lora --epochs 3 --lr "$lr" --seed "$seed" \
                    --out "results/reg_llm15_cola_${name}_lr${tag}_seed${seed}.pt"
            done
        done
    done

    echo "=== round5 llm: Qwen2.5-7B (L=28), CoLA / RTE ==="
    JOB_GMEM="40G"
    M7="Qwen/Qwen2.5-7B"
    cfgs=("lora_r8:identity:28:8" "dct_p28_r8:dct:28:8" "lora_r1:identity:28:1" "dct_p2_r14:dct:2:14")
    for task in cola rte; do
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r name tf p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rl7_${task}_${name}_s${seed}" "$WALLTIME_LONG" mt_lora_llm.py \
                    --model "$M7" --task "$task" --transform "$tf" --rank "$r" --n_faces "$p" \
                    --alpha "$alpha" --a_init lora --epochs 3 --lr 1e-4 --grad_ckpt --seed "$seed" \
                    --out "results/reg_llm7_${task}_${name}_lr1e4_seed${seed}.pt"
            done
        done
    done
    JOB_GMEM="12G"
fi

# ---------------------------------------------------------------------------
# round6: learned (Cayley) M in the round5 regimes. See the header for the rationale.
#   variant tag cay<i|d><e>: start at identity / DCT, lr_M = 1e-<e>.
# ---------------------------------------------------------------------------
if want6 budget; then
    echo "=== round6 budget: Cayley M at small budgets, RoBERTa-base, CoLA + RTE ==="
    for B in 12 24 48; do
        r=$((B / 2)); alpha=$(python3 -c "print($r / 2)")
        for var in "cayd3:dct:1e-3" "cayd2:dct:1e-2" "cayi2:identity:1e-2"; do
            IFS=: read -r vname minit lrm <<< "$var"
            for task_spec in "cola:5" "rte:10"; do
                task="${task_spec%%:*}"; ep="${task_spec#*:}"
                for seed in 0 1 2 3 4; do
                    submit_lsf "rc_${task}_${vname}_p2_r${r}_s${seed}" "$WALLTIME_SHORT" mt_lora_demo.py \
                        --task "$task" --transform cayley --m_init "$minit" --lr_M "$lrm" \
                        --freeze_M_epochs 2 --rank "$r" --n_faces 2 --alpha "$alpha" \
                        --a_init lora --epochs "$ep" --lr 1e-3 --seed "$seed" \
                        --out "results/reg_budget_${task}_${vname}_p2_r${r}_seed${seed}.pt"
                done
            done
        done
    done
fi

if want6 llm; then
    echo "=== round6 llm: Cayley M, Qwen2.5-1.5B (L=28), CoLA / RTE / MRPC ==="
    JOB_GMEM="24G"
    M15="Qwen/Qwen2.5-1.5B"
    cfgs=("cayd3:dct:1e-3:2:14" "cayd2:dct:1e-2:2:14" "cayi2:identity:1e-2:2:14"
          "cayi3:identity:1e-3:28:8" "cayd3:dct:1e-3:28:8")
    for task in cola rte mrpc; do
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r vname minit lrm p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rcl15_${task}_${vname}_p${p}_r${r}_s${seed}" "$WALLTIME_SHORT" mt_lora_llm.py \
                    --model "$M15" --task "$task" --transform cayley --m_init "$minit" --lr_M "$lrm" \
                    --freeze_M_epochs 1 --rank "$r" --n_faces "$p" --alpha "$alpha" \
                    --a_init lora --epochs 3 --lr 2e-4 --seed "$seed" \
                    --out "results/reg_llm15_${task}_${vname}_p${p}_r${r}_lr2e4_seed${seed}.pt"
            done
        done
    done

    echo "=== round6 llm: Cayley M, Qwen2.5-7B (L=28), CoLA / RTE ==="
    JOB_GMEM="40G"
    M7="Qwen/Qwen2.5-7B"
    cfgs=("cayd2:dct:1e-2:2:14" "cayi3:identity:1e-3:28:8")
    for task in cola rte; do
        for cfg in "${cfgs[@]}"; do
            IFS=: read -r vname minit lrm p r <<< "$cfg"
            alpha=$(python3 -c "print($r / 2)")
            for seed in 0 1 2; do
                submit_lsf "rcl7_${task}_${vname}_p${p}_r${r}_s${seed}" "$WALLTIME_LONG" mt_lora_llm.py \
                    --model "$M7" --task "$task" --transform cayley --m_init "$minit" --lr_M "$lrm" \
                    --freeze_M_epochs 1 --rank "$r" --n_faces "$p" --alpha "$alpha" \
                    --a_init lora --epochs 3 --lr 1e-4 --grad_ckpt --seed "$seed" \
                    --out "results/reg_llm7_${task}_${vname}_p${p}_r${r}_lr1e4_seed${seed}.pt"
            done
        done
    done
    JOB_GMEM="12G"
fi

echo ""
echo "Done. Monitor:  bjobs -w"
echo "Tail a log:     tail -f logs/<jobname>.log"
echo "Results:        ls -lh $RESULTS_DIR/*.pt $RESULTS_DIR/*/*.pt"
