#!/bin/bash
# Post-orchestration: re-run the imbalance conditions with the richer victim set
# + purity gate, add the moderate (rho=10) imbalance level, then re-analyze.
# Waits for the main orchestrator's ORCH_DONE marker so the GPU is free.
cd /root/autodl-tmp/ChangeDullahanStealthySourceCode
source /root/miniconda3/etc/profile.d/conda.sh && conda activate cdssc
LOG=exp_robust/logs/post_orch.log
: > "$LOG"
echo "[post] $(date) waiting for ORCH_DONE..." >> "$LOG"
while [ ! -f exp_robust/logs/ORCH_DONE ]; do sleep 20; done
echo "[post] $(date) ORCH_DONE. re-running lt (rho=100) condition..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run lt >> "$LOG" 2>&1
echo "[post] $(date) training lt10 (rho=10)..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main train lt10 >> "$LOG" 2>&1
echo "[post] $(date) running lt10 condition..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run lt10 >> "$LOG" 2>&1
echo "[post] $(date) re-running analyze..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.analyze >> "$LOG" 2>&1
echo "[post] $(date) POST DONE" >> "$LOG"
touch exp_robust/logs/POST_DONE
