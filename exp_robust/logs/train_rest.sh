#!/bin/bash
# Chain-trains lt, noise20, noise40 AFTER the baseline model finishes, so the
# single GPU is never over-subscribed. Polls for the baseline nor_last snapshot
# + the training process to have exited.
cd /root/autodl-tmp/ChangeDullahanStealthySourceCode
source /root/miniconda3/etc/profile.d/conda.sh && conda activate cdssc
LOG=exp_robust/logs/train_rest.log
echo "[chain] waiting for baseline to finish..." > "$LOG"
while true; do
  if [ -f exp_robust/outputs/models/baseline/server_nor_last.pt ] && \
     ! pgrep -f "exp_robust.main train baseline" >/dev/null; then
    break
  fi
  sleep 30
done
echo "[chain] baseline done at $(date). starting lt..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main train lt >> "$LOG" 2>&1
echo "[chain] lt done at $(date). starting noise20..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main train noise20 >> "$LOG" 2>&1
echo "[chain] noise20 done at $(date). starting noise40..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main train noise40 >> "$LOG" 2>&1
echo "[chain] ALL TRAINING DONE at $(date)" >> "$LOG"
