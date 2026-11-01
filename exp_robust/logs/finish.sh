#!/bin/bash
# Finish the remaining experiments after the disk-full interruption:
#   1. dynamic (3-phase, reuses baseline model)
#   2. re-run lt (rho=100) with richer victims + purity gate
#   3. train + run lt10 (rho=10, moderate imbalance)
#   4. final analyze
# Each condition auto-frees its feature cache when done.
cd /root/autodl-tmp/ChangeDullahanStealthySourceCode
source /root/miniconda3/etc/profile.d/conda.sh && conda activate cdssc
LOG=exp_robust/logs/finish.log
: > "$LOG"

echo "[finish] $(date) running dynamic..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run dynamic >> "$LOG" 2>&1

echo "[finish] $(date) re-running lt (rho=100)..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run lt >> "$LOG" 2>&1

echo "[finish] $(date) training lt10 (rho=10)..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main train lt10 >> "$LOG" 2>&1

echo "[finish] $(date) running lt10 condition..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run lt10 >> "$LOG" 2>&1

echo "[finish] $(date) final analyze..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.analyze >> "$LOG" 2>&1

echo "[finish] $(date) ALL DONE" >> "$LOG"
touch exp_robust/logs/FINISH_DONE
