#!/usr/bin/env python
"""
Fine-tune the flip experiment as a detachable script.

Cells 20/32/34/35 of the notebook, with the knobs moved to the command line so a
run can be launched and left alone. StepProgress writes one durable line per step
to a log file, so `tail -f` shows progress from another terminal.

Examples
--------
# same as the notebook run you already did, to confirm parity
python run_train.py --n-train 32 --epochs 1 --max-minutes 45 \
                    --out finetuned_32steps.ckpt

# the bigger run: more flares, more epochs, longer wall-clock budget
python run_train.py --n-train 512 --n-val 128 --epochs 3 --max-minutes 720 \
                    --out finetuned_512x3.ckpt

# train, then immediately plot the same event with the new weights
python run_train.py --n-train 512 --epochs 3 --out ft512.ckpt --plot-after

Leave it running:
    mkdir -p logs
    nohup python run_train.py --n-train 512 --epochs 3 --out ft512.ckpt \
        > logs/train512.log 2>&1 &
    tail -f logs/train512.log

IMPORTANT: check `max_samples` in configs/config_script.yaml. The workshop config
ships with `max_samples: 10`, which caps the dataset regardless of --n-train. Set
it to `null` before a real training run.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger

import surya_setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=32, help="training flares to draw")
    p.add_argument("--n-val", type=int, default=16)
    p.add_argument("--n-test", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-minutes", type=int, default=45,
                   help="hard wall-clock stop, so a slow network cannot eat the night")
    p.add_argument("--flip-prob", type=float, default=0.5,
                   help="training mix: 0.0 upright | 0.5 mixed | 1.0 flipped")
    p.add_argument("--lr", type=float, default=None, help="override cfg.learning_rate")
    p.add_argument("--seed", type=int, default=42,
                   help="also changes WHICH flares are drawn")
    p.add_argument("--out", default=None,
                   help="checkpoint path. Default encodes the settings.")
    p.add_argument("--logdir", default="logs")
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--val-during-fit", action="store_true",
                   help="run validation passes during training (costs extra reads)")
    p.add_argument("--plot-after", action="store_true",
                   help="call run_plots.py with the new checkpoint when fit finishes")
    p.add_argument("--input-minutes", default=None,
                   help="override the input window, e.g. '-120,-60'. Must have the "
                        "same number of entries as the config (2).")
    p.add_argument("--config", default="./configs/config_script.yaml")
    return p.parse_args()


class StepProgress(L.Callback):
    """One durable line per step, to stdout and to a log file."""

    def __init__(self, logfile):
        self.logfile = logfile

    def _say(self, m):
        line = f"[{dt.datetime.now():%H:%M:%S}] {m}"
        print(line, flush=True)
        with open(self.logfile, "a") as fh:
            fh.write(line + "\n")

    def on_train_start(self, tr, pl):
        self.total = tr.max_epochs * len(tr.train_dataloader)
        self.t0, self.prev = time.perf_counter(), None
        self._say(f"START {self.total} steps -> {self.logfile}")

    def on_train_batch_start(self, tr, pl, batch, i):
        now = time.perf_counter()
        self.wait = 0.0 if self.prev is None else now - self.prev
        self.begin = now

    def on_train_batch_end(self, tr, pl, outputs, batch, i):
        self.prev = time.perf_counter()
        step, el = tr.global_step, self.prev - self.t0
        per = el / max(step, 1)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs
        self._say(f"step {step:4d}/{self.total}  loss {float(loss):.4f}  "
                  f"compute {self.prev-self.begin:5.1f}s  data-wait {self.wait:5.1f}s  "
                  f"{per:5.1f}s/step  ETA {(self.total-step)*per/60:5.1f}m")

    def on_train_end(self, tr, pl):
        self._say(f"END status={tr.state.status} step={tr.global_step}/{self.total}")


def main():
    args = parse_args()
    os.makedirs(args.logdir, exist_ok=True)

    stamp = f"{dt.datetime.now():%m%d_%H%M}"
    out_ckpt = args.out or f"finetuned_n{args.n_train}_e{args.epochs}_{stamp}.ckpt"
    progress_log = os.path.join(args.logdir, f"progress_{stamp}.log")

    # The sample counts live in surya_setup as module-level knobs, so set them
    # before build_everything reads them.
    surya_setup.N_TRAIN = args.n_train
    surya_setup.N_VAL = args.n_val
    surya_setup.N_TEST = args.n_test
    surya_setup.SUBSET_SEED = args.seed
    surya_setup.FLIP_PROBABILITY = args.flip_prob

    print("=" * 70)
    print(f"train {args.n_train} | val {args.n_val} | test {args.n_test}")
    print(f"epochs {args.epochs} | max {args.max_minutes} min | flip_p {args.flip_prob}")
    print(f"checkpoint -> {out_ckpt}")
    print(f"progress   -> {progress_log}")
    print("=" * 70)

    input_minutes = None
    if args.input_minutes:
        input_minutes = [int(x) for x in args.input_minutes.split(",")]

    objs = surya_setup.build_everything(args.config, input_minutes=input_minutes)
    cfg = objs["cfg"]
    lit_model = objs["lit_model"]
    train_data_loader = objs["train_data_loader"]
    val_data_loader = objs["val_data_loader"]
    test_data_loader = objs["test_data_loader"]

    # The workshop config caps the dataset for quick experiments. If it is still
    # set, --n-train above it silently does nothing, so say so loudly.
    max_samples = getattr(cfg.data, "max_samples", None)
    if max_samples is not None and max_samples < args.n_train:
        print(f"\n*** WARNING: cfg max_samples = {max_samples} but --n-train = "
              f"{args.n_train}. The cap wins. Set `max_samples: null` in "
              f"{args.config} before a real run. ***\n")

    if args.lr is not None:
        lit_model.lr = args.lr
        print(f"learning rate overridden -> {args.lr}")

    # ---- loggers ----
    loggers = [CSVLogger("runs", name=cfg.wandb_project)]
    if not args.no_wandb:
        loggers.append(WandbLogger(
            entity=cfg.wandb_entity,
            project=cfg.wandb_project,
            name=f"flare_flip_finetune_{stamp}",
            log_model=False,
            save_dir="./wandb_tmp",
        ))

    trainer = L.Trainer(
        max_epochs=args.epochs,
        max_time={"minutes": args.max_minutes},
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        logger=loggers,
        # No validation during fit by default -- the evaluate()/plot passes do the
        # scoring, and a mid-training val pass re-reads samples for nothing.
        limit_val_batches=(1.0 if args.val_during_fit else 0),
        # save_last writes last.ckpt, so an interrupted run is not lost.
        callbacks=[ModelCheckpoint(save_top_k=0, save_last=True),
                   StepProgress(progress_log)],
        log_every_n_steps=1,
    )
    steps = args.epochs * len(train_data_loader)
    print(f"{args.epochs} epochs x {len(train_data_loader)} steps = {steps} optimizer steps")

    # Only the TRAINING set flips randomly. val and test stay at 0.0 because the
    # evaluation flips them by hand to score each sample in both orientations.
    train_data_loader.dataset.flip_probability = args.flip_prob
    val_data_loader.dataset.flip_probability = 0.0
    test_data_loader.dataset.flip_probability = 0.0

    if args.val_during_fit:
        trainer.fit(lit_model, train_data_loader, val_data_loader)
    else:
        trainer.fit(lit_model, train_data_loader)

    # ---- what actually happened ----
    status = str(trainer.state.status)
    print("\n" + "=" * 70)
    print("status         :", status)
    print("epochs done    :", trainer.current_epoch, "of", trainer.max_epochs)
    print("optimizer steps:", trainer.global_step, "of", steps)
    if "INTERRUPTED" in status.upper():
        print("  ^^ interrupted -- max_time did NOT fire")
    elif trainer.global_step < steps:
        print("  ^^ stopped early -- the max_time guard fired")
    else:
        print("  ^^ completed all steps")

    trainer.save_checkpoint(out_ckpt)
    print("saved          :", out_ckpt)

    mx, verdict = surya_setup.weight_provenance(lit_model)
    print(f"max|B|         : {mx:.3e}  ->  {verdict}")
    if verdict == "STOCK PRETRAINED":
        print("  ^^ the adapters never moved. Either zero steps ran, or the LoRA "
              "params are not in the optimizer. Check the step count above.")

    try:
        import wandb
        if wandb.run is not None:
            wandb.run.finish()
    except ImportError:
        pass

    print("=" * 70)

    if args.plot_after:
        print("\nplotting with the new weights ...")
        for split in ("val", "test"):
            cmd = [sys.executable, "run_plots.py",
                   "--weights", out_ckpt, "--split", split,
                   "--goes-class", "MX", "--both"]
            if args.input_minutes:
                cmd += ["--input-minutes", args.input_minutes]
            subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
