#!/bin/bash
# Master orchestrator. Waits for the baseline model to finish training, then
# pipelines each remaining model's TRAIN + CONDITION run, then runs dynamic and
# analysis. Keeps the single GPU continuously busy with no over-subscription.
cd /root/autodl-tmp/ChangeDullahanStealthySourceCode
source /root/miniconda3/etc/profile.d/conda.sh && conda activate cdssc
LOG=exp_robust/logs/orchestrate.log
: > "$LOG"

wait_model () {  # $1 = tag
  while true; do
    if [ -f exp_robust/outputs/models/$1/server_nor_last.pt ] && \
       ! pgrep -f "exp_robust.main train $1" >/dev/null; then
      break
    fi
    sleep 20
  done
}

echo "[orch] $(date) waiting for baseline model..." >> "$LOG"
wait_model baseline
echo "[orch] $(date) baseline trained. running baseline condition..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run baseline >> "$LOG" 2>&1

for cond in lt noise20 noise40; do
  echo "[orch] $(date) training $cond ..." >> "$LOG"
  PYTHONUNBUFFERED=1 python -m exp_robust.main train $cond >> "$LOG" 2>&1
  echo "[orch] $(date) $cond trained. running $cond condition..." >> "$LOG"
  PYTHONUNBUFFERED=1 python -m exp_robust.main run $cond >> "$LOG" 2>&1
done

echo "[orch] $(date) running dynamic..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.main run dynamic >> "$LOG" 2>&1

echo "[orch] $(date) running analysis..." >> "$LOG"
PYTHONUNBUFFERED=1 python -m exp_robust.analyze >> "$LOG" 2>&1
echo "[orch] $(date) ALL DONE" >> "$LOG"
touch exp_robust/logs/ORCH_DONE
