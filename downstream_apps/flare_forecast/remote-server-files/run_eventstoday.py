# -*- coding: utf-8 -*-
"""
Score specific flares by peak time (e.g. the 2017 X-class events), upright and
flipped, with several sets of weights in one go.

Why this exists: run_plots.py only searches the CAPPED test/val subsets, and the
test subset is 2023-only, so 2017 events can never be found there. This script
keeps the UNCAPPED train and val pools, finds each event in whichever pool holds
it, and reuses run_plots.report_and_plot so figures and numbers match the
earlier runs exactly.

It also checks for leakage: whether each event (or its neighbours from the same
week, e.g. AR 12673 in Sept 2017) was drawn into the training subset.

Run from:  <repo_root>/downstream_apps/flare_forecast/

Step 1, cheap sanity check (no images read, no plotting):
    python run_events.py --dry-run

Step 2, the real thing, one call per input window:
    nohup python run_events.py \
        --weights pretrained,ckpt_q1_win60_n64.ckpt,ckpt_big256.ckpt \
        --input-minutes=-60,-36 --n-train 256 \
        --outdir figures/events2017_w60 > logs/events2017_w60.log 2>&1 &
"""
import argparse
import copy
import gc
import os

import pandas as pd
import torch

import surya_setup
import run_plots

DEFAULT_EVENTS = "2017-04-02T08:02,2017-09-10T16:06,2017-09-06T12:02"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--events", default=DEFAULT_EVENTS,
                   help="comma list of flare peak times")
    p.add_argument("--weights", default="pretrained",
                   help="comma list: 'pretrained' and/or .ckpt paths. All must "
                        "have been trained with the same --input-minutes.")
    p.add_argument("--input-minutes", default="-60,-36",
                   help="MUST match the window the checkpoints were trained on")
    p.add_argument("--n-train", type=int, default=256,
                   help="training-set size of the LARGEST checkpoint in --weights; "
                        "used only to rebuild its training draw for the leakage check")
    p.add_argument("--seed", type=int, default=42, help="training draw seed")
    p.add_argument("--tol-minutes", type=int, default=10,
                   help="how close a catalog peak_time must be to count as a match")
    p.add_argument("--near-days", type=float, default=7.0,
                   help="report training flares within this many days of each event")
    p.add_argument("--dry-run", action="store_true",
                   help="locate events + leakage check only, no model runs")
    p.add_argument("--label", default="2017", help="goes into filenames")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--channels", default="aia131,aia171")
    p.add_argument("--scales", default="normal")
    p.add_argument("--outdir", default="figures/events2017")
    p.add_argument("--config", default="./configs/config_script.yaml")
    return p.parse_args()


def peaks_of(ds):
    return pd.to_datetime(ds.df_valid_indices["peak_time"].astype(str))


def find_event(pools, target, tol):
    """Return (pool_name, position, row) of the closest match, or None."""
    best = None
    for name, ds in pools.items():
        gap = (peaks_of(ds) - target).abs()
        if len(gap) == 0:
            continue
        pos = int(gap.values.argmin())
        g = gap.iloc[pos]
        if g <= tol and (best is None or g < best[3]):
            best = (name, pos, ds.df_valid_indices.iloc[pos], g)
    return best


def nearby(ds, target, days):
    pk = peaks_of(ds)
    mask = (pk - target).abs() <= pd.Timedelta(days=days)
    df = ds.df_valid_indices[mask.values]
    return df[["GOES_class", "peak_time"]]


def write_html(df, outdir):
    rows = df.drop(columns=["figures"]).to_html(index=False, float_format="%.4f")
    figs = []
    for _, r in df.iterrows():
        for f in str(r["figures"]).split(";"):
            if f:
                b = os.path.basename(f)
                cap = "{} &middot; {} {} &middot; {} &middot; MSE {:.4f} &middot; skill {:+.1%}".format(
                    r["weights"], r["GOES_class"], r["peak_time"], r["orientation"],
                    r["mse_model"], r["skill"])
                figs.append('<figure><figcaption>{}</figcaption><a href="{}">'
                            '<img src="{}"></a></figure>'.format(cap, b, b))
    html = ("<!doctype html><meta charset='utf-8'><title>Event comparison</title>"
            "<style>:root{{color-scheme:light dark}}body{{font:15px/1.5 system-ui,sans-serif;margin:2rem}}"
            "table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #8884;padding:.25rem .5rem}}"
            "figure{{margin:1.5rem 0}}figcaption{{font-size:13px;opacity:.8}}"
            "img{{width:100%;border:1px solid #8884}}</style>"
            "<h1>Selected events, upright vs flipped</h1>{}{}").format(rows, "".join(figs))
    path = os.path.join(outdir, "events_comparison.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tol = pd.Timedelta(minutes=args.tol_minutes)
    targets = [pd.Timestamp(t.strip()) for t in args.events.split(",") if t.strip()]
    weights_list = [w.strip() for w in args.weights.split(",") if w.strip()]
    input_minutes = [int(x) for x in args.input_minutes.split(",")]

    # Reproduce the largest checkpoint's training draw, so the leakage check is real.
    surya_setup.N_TRAIN = args.n_train
    surya_setup.SUBSET_SEED = args.seed

    # build_everything shrinks the pools in place via take(). Wrap take() so a
    # shallow copy of each UNCAPPED pool is kept first. Order of calls inside
    # build_everything: train_full, val_full, test_full.
    stash = []
    orig_take = surya_setup.take

    def take_and_stash(ds, positions):
        stash.append(copy.copy(ds))
        return orig_take(ds, positions)

    surya_setup.take = take_and_stash
    objs = surya_setup.build_everything(args.config, input_minutes=input_minutes)
    surya_setup.take = orig_take

    cfg, lit_model = objs["cfg"], objs["lit_model"]
    pools = {"train-pool": stash[0], "val-pool": stash[1]}
    train_sub = objs["train_ds"]
    for ds in pools.values():
        ds.flip_probability = 0.0

    for name, ds in pools.items():
        yrs = peaks_of(ds).dt.year.value_counts().sort_index().to_dict()
        print("{}: {} flares, years {}".format(name, len(ds), yrs))

    # ---- locate events + leakage check ------------------------------------
    found = []
    print("\n" + "=" * 90)
    for t in targets:
        hit = find_event(pools, t, tol)
        if hit is None:
            print("{}  NOT FOUND within {} min in either pool.".format(t, args.tol_minutes))
            for name, ds in pools.items():
                nb = nearby(ds, t, 1)
                if len(nb):
                    print("  closest in {} (+-1 day):\n{}".format(name, nb.to_string(index=False)))
            continue
        name, pos, row, gap = hit
        print("{}  -> {} in {} (pos {}, peak {}, off by {})".format(
            t, row["GOES_class"], name, pos, row["peak_time"], gap))

        same = (peaks_of(train_sub) - t).abs() <= tol
        near = nearby(train_sub, t, args.near_days)
        if same.any():
            print("  *** LEAK: this exact flare is IN the n={} training draw ***".format(args.n_train))
        else:
            print("  not in the n={} training draw".format(args.n_train))
        if len(near):
            print("  training flares within {} days (same-region risk):\n{}".format(
                args.near_days, near.to_string(index=False)))
        found.append((t, name, pos))
    print("=" * 90)

    if args.dry_run or not found:
        print("\ndry run / nothing to plot. Done.")
        return

    # ---- score every event with every weights set -------------------------
    ch_list = [c.strip() for c in args.channels.split(",")]
    ch_idx = {c: cfg.data.channels.index(c) for c in ch_list}
    scale_map = {"normal": 1, "enhanced": 10}
    scale_versions = [(s.strip(), scale_map[s.strip()]) for s in args.scales.split(",")]

    # Snapshot the stock weights so each checkpoint starts from a clean model.
    pretrained_sd = {k: v.detach().cpu().clone() for k, v in lit_model.state_dict().items()}

    summary = []
    for w in weights_list:
        lit_model.load_state_dict(pretrained_sd, strict=True)
        if w == "pretrained":
            tag = "pretrained"
        else:
            surya_setup.load_finetuned(lit_model, w)
            tag = os.path.splitext(os.path.basename(w))[0]
        mx, verdict = surya_setup.weight_provenance(lit_model)
        print("\n[{}] max|B| = {:.3e} -> {}".format(tag, mx, verdict))
        if (w == "pretrained") != (verdict == "STOCK PRETRAINED"):
            print("  WARNING: weights provenance does not match what was asked for")

        for t, name, pos in found:
            for orient in ("UPRIGHT", "FLIPPED"):
                r = run_plots.report_and_plot(pools[name], pos, args.label, lit_model, cfg,
                                              orient, args, ch_idx, ch_list,
                                              scale_versions, tag)
                r["pool"] = name
                summary.append(r)
                gc.collect()
                torch.cuda.empty_cache()

    df = pd.DataFrame(summary)
    csv_path = os.path.join(args.outdir, "summary_events.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + df.drop(columns=["figures"]).to_string(index=False))

    piv = df.pivot_table(index=["weights", "GOES_class"], columns="orientation",
                         values="mse_model")
    if {"UPRIGHT", "FLIPPED"}.issubset(piv.columns):
        piv["flip_minus_upright"] = piv["FLIPPED"] - piv["UPRIGHT"]
        print("\nflipped minus upright MSE (positive = flipping hurts):")
        print(piv.to_string(float_format="{:.4f}".format))

    print("\nsummary -> {}".format(csv_path))
    print("html    -> {}".format(write_html(df, args.outdir)))


if __name__ == "__main__":
    main()

