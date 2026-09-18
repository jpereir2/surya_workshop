#!/usr/bin/env bash
# One unattended 256-flare run: purge cache during training, stop purging,
# then plot. Safe to launch and go to sleep.
#
#   chmod +x big_run.sh
#   nohup ./big_run.sh > logs/big_run.log 2>&1 &
#   tail -f logs/big_run.log
#
# Morning:  cat logs/BIG_STATUS.txt

set -u
mkdir -p logs figures
STATUS=logs/BIG_STATUS.txt
: > "$STATUS"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

CKPT=ckpt_big256.ckpt
CACHE=~/scratch_space/surya_cache

say "starting 256-flare run"
rm -rf "$CACHE"/*
say "cache cleared, $(df -h /home | awk 'NR==2{print $4}') free"

# --- background purger: delete files untouched for 10+ min ------------------
# Valid only at 1 epoch, where each file is read exactly once.
(
  while true; do
      find "$CACHE" -type f -amin +10 -delete 2>/dev/null
      find "$CACHE" -type d -empty -delete 2>/dev/null
      echo "[$(date +%H:%M)] purge: $(df -h /home | awk 'NR==2{print $4}') free"
      sleep 300
  done
) > logs/purge.log 2>&1 &
PURGE_PID=$!
say "purger running as PID $PURGE_PID"

# --- train ------------------------------------------------------------------
python run_train.py --n-train 256 --n-val 32 --n-test 32 --epochs 1 \
    --max-minutes 90 --input-minutes=-60,-36 --out "$CKPT" --no-wandb \
    >> logs/train_big256.log 2>&1

# --- stop purging BEFORE plotting ------------------------------------------
kill "$PURGE_PID" 2>/dev/null
say "purger stopped"

if [ ! -f "$CKPT" ]; then
    say "FAILED: no checkpoint. See logs/train_big256.log"
    exit 1
fi

# A checkpoint truncated by a full disk loads as a corrupt zip, so check the
# size before spending time plotting. A good one is ~1.8G.
SIZE=$(stat -c %s "$CKPT")
say "checkpoint written: $(du -h "$CKPT" | cut -f1)"
if [ "$SIZE" -lt 1500000000 ]; then
    say "FAILED: checkpoint is too small, probably truncated. Not plotting."
    exit 1
fi

# --- plot -------------------------------------------------------------------
python compare_runs.py --ckpt "$CKPT" --split test --goes-class MX --n-show 2 \
    --input-minutes=-60,-36 --outdir figures/big256_test \
    >> logs/plot_big256.log 2>&1 \
    && say "plotted -> figures/big256_test/" \
    || say "plotting FAILED, see logs/plot_big256.log"

say "done. $(df -h /home | awk 'NR==2{print $4}') free"
