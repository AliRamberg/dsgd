#!/bin/bash
# Distributed SGD Experiment Runner
# 
# This script helps run multiple experiments with different configurations
# for easy comparison. Modify the arrays below to define your experiments.
#
# Usage examples:
#   ./run_test.sh                    # Run experiments defined in arrays below
#   
# To compare different configs, modify the arrays:
#   - Add multiple values: WORKERS=(2 4 8) to test different worker counts
#   - Add multiple modes: MODES=("ssp" "ssgd" "asgd") to compare algorithms
#   - Add multiple staleness: SSP_STALENESS=(5 10 20) to test SSP windows
#
# Each experiment creates:
#   - A .conf file with the exact command used
#   - A directory with results (config.json, events.jsonl, meta.json)

set -e  # Exit on error

# ============================================================================
# Configuration: Modify these arrays to define your experiments
# ============================================================================

# Experiment run ID (creates experiments/run-${RUN_ID}/)
RUN_ID=17

# Training modes to test
MODES=("asgd" "ssp" "localsgd")

# Number of workers
WORKERS=(2)

# Total updates/steps/epochs (for SSGD)
STEPS=(100)

# SSP staleness windows (only used for ssp mode)
SSP_STALENESS=(5)

# LocalSGD local steps (only used for localsgd mode)
LOCAL_K=(5)

# Common hyperparameters
LR=0.1
HETERO_BASE=0.05
HETERO_JITTER=0.02
HETERO_STRAGGLER_EVERY=5

# ============================================================================
# Script logic (usually no need to modify below)
# ============================================================================

OUTDIR="experiments/run-${RUN_ID}"
mkdir -p "${OUTDIR}"

# Counter for tracking experiments
EXPERIMENT_COUNT=0

echo "=========================================="
echo "Starting experiment batch: run-${RUN_ID}"
echo "=========================================="
echo ""

# Common hyperparameters (constant across all experiments)
COMMON_ARGS="--lr ${LR} --hetero-base ${HETERO_BASE} --hetero-jitter ${HETERO_JITTER} --hetero-straggler-every ${HETERO_STRAGGLER_EVERY} --outdir ${OUTDIR}/"

# Base command template - builds common command with variable parameters
build_base_cmd() {
    local mode=$1
    local workers=$2
    local steps=$3
    local name=$4
    
    echo "uv run --active main.py --mode ${mode} --total-updates ${steps} --num-workers ${workers} ${COMMON_ARGS} --run-name=${name}"
}

for mode in "${MODES[@]}"; do
    for workers in "${WORKERS[@]}"; do
        for steps in "${STEPS[@]}"; do
            # Build experiment name and command based on mode and parameters
            case "$mode" in
                ssp)
                    for staleness in "${SSP_STALENESS[@]}"; do
                        NAME="${mode}-w${workers}-stale${staleness}-u${steps}"
                        BASE_CMD=$(build_base_cmd "$mode" "$workers" "$steps" "$NAME")
                        CMD="${BASE_CMD} --ssp-staleness ${staleness}"
                        
                        echo "[$((++EXPERIMENT_COUNT))] Running: ${NAME}"
                        echo "${CMD}" > "${OUTDIR}/${NAME}.conf"
                        eval "${CMD}"
                        echo ""
                    done
                    ;;
                localsgd)
                    for local_k in "${LOCAL_K[@]}"; do
                        NAME="${mode}-w${workers}-k${local_k}-u${steps}"
                        BASE_CMD=$(build_base_cmd "$mode" "$workers" "$steps" "$NAME")
                        CMD="${BASE_CMD} --local-k ${local_k}"
                        
                        echo "[$((++EXPERIMENT_COUNT))] Running: ${NAME}"
                        echo "${CMD}" > "${OUTDIR}/${NAME}.conf"
                        eval "${CMD}"
                        echo ""
                    done
                    ;;
                ssgd|asgd)
                    NAME="${mode}-w${workers}-u${steps}"
                    CMD=$(build_base_cmd "$mode" "$workers" "$steps" "$NAME")
                    
                    echo "[$((++EXPERIMENT_COUNT))] Running: ${NAME}"
                    echo "${CMD}" > "${OUTDIR}/${NAME}.conf"
                    eval "${CMD}"
                    echo ""
                    ;;
            esac
        done
    done
done

echo "=========================================="
echo "Completed ${EXPERIMENT_COUNT} experiments"
echo "Results saved to: ${OUTDIR}/"
echo "=========================================="
