#!/usr/bin/env bash
# Overnight 2x2 for the flip experiment.
#
# Two variables, four runs:
#
#                      inputs [-60,-36]      inputs [-120,-60]
#   quick (n64, e1)        run 1                  run 2
#   deeper (n256, e2)      run 3                  run 4
#
# The quick pair lands within ~1.5 h, so you have a complete 2x2 on the input
# window early. The deeper pair repeats it with 4x the data and 2 epochs, which
# is the "does more training change the answer" axis.
#
# Every run is independent: trains, checkpoints, then plots before the next one
# starts. A failure logs and the chain continues. Every run has a hard
# wall-clock cap, so a slow S3 pull cannot eat the night on one rung.
#
# Evaluation always scores BOTH orientations (--both), so every run answers the
# flip question regardless of what it trained on.
#
# Usage:
#   chmod +x overnight.sh
#   nohup ./overnight.sh > logs/overnight.log 2>&1 &
#   tail -f logs/overnight.log
#
# Morning:
#   cat logs/STATUS.txt
#   ls -d figures/*/

set -u   # deliberately NOT -e: a failed run must not kill the chain

mkdir -p logs figures
STATUS=logs/STATUS.txt
: > "$STATUS"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$STATUS"; }

# name  ntrain nval ntest epochs mins  input_window  flip_prob
run_one() {
    local name=$1 ntrain=$2 nval=$3 ntest=$4 epochs=$5 mins=$6 win=$7 flip=$8
    local ckpt="ckpt_${name}.ckpt"

    say "=== ${name}: train=${ntrain} epochs=${epochs} cap=${mins}m inputs=[${win}] flip_p=${flip} ==="

    python run_train.py \
        --n-train "$ntrain" --n-val "$nval" --n-test "$ntest" \
        --epochs "$epochs" --max-minutes "$mins" \
        --input-minutes="$win" --flip-prob "$flip" \
        --out "$ckpt" --no-wandb \
        >> "logs/train_${name}.log" 2>&1

    if [ ! -f "$ckpt" ]; then
        say "${name}: FAILED, no checkpoint. See logs/train_${name}.log"
        return 1
    fi
    say "${name}: trained -> ${ckpt}"

    # Evaluation MUST use the same input window the model trained on.
    for split in test val; do
        python compare_runs.py \
            --ckpt "$ckpt" --split "$split" --goes-class MX --n-show 2 \
            --input-minutes="$win" \
            --outdir "figures/${name}_${split}" \
            >> "logs/plot_${name}_${split}.log" 2>&1 \
            && say "${name}: plotted ${split} -> figures/${name}_${split}/" \
            || say "${name}: plotting ${split} FAILED, see logs/plot_${name}_${split}.log"
    done

    say "${name}: COMPLETE"
    return 0
}

say "2x2 ladder started"
say "free RAM: $(free -g | awk '/^Mem:/{print $7}') GB | disk: $(df -h /home | awk 'NR==2{print $4}') free"

# ---- quick pair: the input-window comparison, done within ~1.5 h -------------
run_one "q1_win60_n64"   64 32 32 1  45 "-60,-36"  0.5
run_one "q2_win120_n64"  64 32 32 1  45 "-120,-60" 0.5

# ---- deeper pair: same comparison, 4x the data, 2 epochs --------------------
run_one "d3_win60_n256"  256 64 64 2 120 "-60,-36"  0.5
run_one "d4_win120_n256" 256 64 64 2 120 "-120,-60" 0.5

say "=== ladder finished ==="
grep "COMPLETE" "$STATUS" || say "nothing completed -- check logs/train_*.log"
ls -d figures/*/ 2>/dev/null | tee -a "$STATUS"
