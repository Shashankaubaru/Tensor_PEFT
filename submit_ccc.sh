#!/bin/bash
# =============================================================================
# submit_ccc.sh — LSF array job submission for star_M-LoRA experiments on CCC
#
# Usage:
#   bash submit_ccc.sh              # submits all experiment groups
#   bash submit_ccc.sh mtlora       # only MTLoRA identity multi-seed sweep
#   bash submit_ccc.sh lotr         # only LoTR baseline sweep
#   bash submit_ccc.sh cayley       # only Cayley learnable-M sweep
#
# Prerequisites on CCC:
#   module load anaconda3/latest (or python/3.11)
#   pip install -r requirements.txt  (run once before submitting)
#   export HF_HOME=/path/to/shared/scratch/.cache/huggingface
# =============================================================================

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$REPO_DIR/results"
LOGS_DIR="$REPO_DIR/logs"
mkdir -p "$RESULTS_DIR" "$LOGS_DIR"

# Queue — set GPU_QUEUE env var to override (e.g. export CCC_GPU_QUEUE=normal)
GPU_QUEUE="${CCC_GPU_QUEUE:-normal}"
USE_GPU="${CCC_USE_GPU:-0}"     # set to 1 if submitting to a GPU queue
NCPUS=8
MEM="32GB"
WALLTIME="08:00"       # hh:mm

# Python — /usr/bin/python3 confirmed available on this cluster
PYTHON="${CCC_PYTHON:-/usr/bin/python3}"

run_job="${1:-all}"

# ---------------------------------------------------------------------------
# Helper: submit one LSF job
# Usage: submit_lsf <job_name> <cmd...>
# ---------------------------------------------------------------------------
submit_lsf() {
    local name="$1"; shift
    local logfile="$LOGS_DIR/${name}.log"
    if [[ "$USE_GPU" == "1" ]]; then
        bsub \
            -J "$name" \
            -q "$GPU_QUEUE" \
            -n "$NCPUS" \
            -R "rusage[mem=$MEM]" \
            -gpu "num=1:mode=exclusive_process" \
            -W "$WALLTIME" \
            -o "$logfile" \
            -e "$logfile" \
            "cd $REPO_DIR && $PYTHON $*"
    else
        bsub \
            -J "$name" \
            -q "$GPU_QUEUE" \
            -n "$NCPUS" \
            -R "rusage[mem=$MEM]" \
            -W "$WALLTIME" \
            -o "$logfile" \
            -e "$logfile" \
            "cd $REPO_DIR && $PYTHON $*"
    fi
    echo "  Submitted: $name  →  $logfile"
}

# ---------------------------------------------------------------------------
# 1. MTLoRA Identity — multi-seed sweep (3 seeds × 4 tasks)
# ---------------------------------------------------------------------------
if [[ "$run_job" == "all" || "$run_job" == "mtlora" ]]; then
    echo "=== Submitting MTLoRA Identity multi-seed sweep ==="
    for task in cola mrpc sst2; do
        epochs=5; lr="5e-4"
        for seed in 0 1 2; do
            name="mtlora_${task}_id_r8_s${seed}"
            submit_lsf "$name" mt_lora_demo.py \
                --task "$task" --transform identity \
                --rank 8 --n_faces 12 --alpha 4.0 \
                --epochs "$epochs" --lr "$lr" --seed "$seed" \
                --out "results/mtlora_${task}_identity_r8_seed${seed}.pt"
        done
    done
    # RTE needs lower LR and more epochs
    for seed in 0 1 2; do
        name="mtlora_rte_id_r8_s${seed}"
        submit_lsf "$name" mt_lora_demo.py \
            --task rte --transform identity \
            --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 1e-4 --seed "$seed" \
            --out "results/mtlora_rte_identity_r8_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# 2. LoTR baseline — multi-seed sweep (3 seeds × 4 tasks)
# ---------------------------------------------------------------------------
if [[ "$run_job" == "all" || "$run_job" == "lotr" ]]; then
    echo "=== Submitting LoTR baseline sweep ==="
    for task in cola mrpc sst2; do
        epochs=5; lr="5e-4"
        for seed in 0 1 2; do
            name="lotr_${task}_r8_s${seed}"
            submit_lsf "$name" lotr_baseline.py \
                --task "$task" --rank 8 --alpha 1.0 \
                --epochs "$epochs" --lr "$lr" --seed "$seed" \
                --out "results/lotr_${task}_r8_seed${seed}.pt"
        done
    done
    for seed in 0 1 2; do
        name="lotr_rte_r8_s${seed}"
        submit_lsf "$name" lotr_baseline.py \
            --task rte --rank 8 --alpha 1.0 \
            --epochs 10 --lr 1e-4 --seed "$seed" \
            --out "results/lotr_rte_r8_seed${seed}.pt"
    done
fi

# ---------------------------------------------------------------------------
# 3. star_M-LoRA-Cayley — 3 seeds on CoLA + 1 seed on MRPC/RTE/SST-2
# ---------------------------------------------------------------------------
if [[ "$run_job" == "all" || "$run_job" == "cayley" ]]; then
    echo "=== Submitting Cayley learnable-M sweep ==="
    # CoLA multi-seed
    for seed in 0 1 2; do
        name="cayley_cola_r8_s${seed}"
        submit_lsf "$name" mt_lora_learnable_M_demo.py \
            --task cola --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 10 --lr 5e-4 --lr_M 1e-3 \
            --freeze_M_epochs 2 --seed "$seed" \
            --out "results/cayley_cola_r8_seed${seed}.pt"
    done
    # MRPC / SST-2
    for task in mrpc sst2; do
        name="cayley_${task}_r8_s0"
        submit_lsf "$name" mt_lora_learnable_M_demo.py \
            --task "$task" --rank 8 --n_faces 12 --alpha 4.0 \
            --epochs 5 --lr 5e-4 --lr_M 1e-3 \
            --freeze_M_epochs 2 --seed 0 \
            --out "results/cayley_${task}_r8_seed0.pt"
    done
    # RTE
    name="cayley_rte_r8_s0"
    submit_lsf "$name" mt_lora_learnable_M_demo.py \
        --task rte --rank 8 --n_faces 12 --alpha 4.0 \
        --epochs 10 --lr 1e-4 --lr_M 1e-3 \
        --freeze_M_epochs 2 --seed 0 \
        --out "results/cayley_rte_r8_seed0.pt"
fi

echo ""
echo "All jobs submitted. Monitor with:  bjobs -w"
echo "Results will appear in: $RESULTS_DIR"
