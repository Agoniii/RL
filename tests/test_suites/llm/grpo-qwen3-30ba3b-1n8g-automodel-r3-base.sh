#!/bin/bash
# Experimental GPU acceptance driver. Not yet enrolled in nightly CI.
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
source "$SCRIPT_DIR/common.env"

MAX_STEPS=${R3_SMOKE_STEPS:-2}
echo "[INFO] R3 smoke: ${MAX_STEPS} steps, fixed recipe seed, single turn, TP1/CP1/EP1."
echo "[INFO] This checks the execution path, not convergence or speed."
cd "$PROJECT_ROOT"
uv run examples/run_grpo.py \
    --config "$CONFIG_PATH" \
    grpo.max_num_steps="$MAX_STEPS" \
    logger.log_dir="$LOG_DIR" \
    logger.wandb_enabled=False \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=False \
    "$@" 2>&1 | tee "$RUN_LOG"

uv run tests/json_dump_tb_logs.py "$LOG_DIR" --output_path "$JSON_METRICS"
uv run tests/check_metrics.py "$JSON_METRICS" \
    "len(data['train/loss']) >= $MAX_STEPS" \
    "all_finite(data['train/loss'])" \
    "all_finite(data['train/grad_norm'])" \
    "min(data['train/grad_norm']) > 0"
