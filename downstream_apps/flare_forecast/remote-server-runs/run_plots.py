#!/usr/bin/env python
"""
Stage 4b as a script: GT/PREDICTION panels saved to disk instead of shown inline.

Same figure layout as the notebook. What changed:
  * plt.savefig instead of plt.show, so it runs headless under nohup
  * filenames encode weights/split/class/orientation, so pretrained and fine-tuned
    runs never overwrite each other and land side by side for comparison
  * --goes-class picks the event by flare class instead of by position, so you can
    ask for an X-class directly
  * a summary CSV per run

Examples
--------
# stock pretrained weights, first M/X-class event in test, both orientations
python run_plots.py --weights pretrained --split test --goes-class MX --both

# your fine-tuned checkpoint, same event, for a like-for-like comparison
python run_plots.py --weights finetuned_32steps.ckpt --split test --goes-class MX --both

# a specific X-class event by date, full resolution
python run_plots.py --weights pretrained --split val --goes-class X \
                    --peak-time 2017-09-10 --stride 1

Leave it running detached:
    nohup python run_plots.py --weights pretrained --both > plots.log 2>&1 &
    tail -f plots.log
"""
import argparse
import gc
import os

import matplotlib
matplotlib.use("Agg")          # headless: no display on the server

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sunpy.visualization.colormaps as cm          # registers 'sdoaia131', 'hmimag'
from torch.utils.data._utils.collate import default_collate

import surya_setup


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default="pretrained",
                   help="'pretrained' for stock Surya, or a path to a .ckpt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--goes-class", default=None,
                   help="filter to events whose class starts with these letters, "
                        "e.g. X, M, MX. Omit to take samples by position.")
    p.add_argument("--peak-time", default=None,
                   help="further filter to events whose peak_time starts with this "
                        "string, e.g. 2017-09-10")
    p.add_argument("--n-show", type=int, default=1, help="how many events to plot")
    p.add_argument("--both", action="store_true",
                   help="render each event UPRIGHT and FLIPPED (the experiment)")
    p.add_argument("--stride", type=int, default=4,
                   help="1 = full 4096 (~872 MB per frame). 4 = ~55 MB. Use 1 only "
                        "for a final figure.")
    p.add_argument("--channels", default="aia131,aia171")
    p.add_argument("--scales", default="normal",
                   help="comma list of normal,enhanced")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--input-minutes", default=None,
                   help="override the input window, e.g. '-120,-60'")
    p.add_argument("--config", default="./configs/config_script.yaml")
    return p.parse_args()


# ==================================================================
# PLOTTING HELPERS  (unchanged from the notebook)
# ==================================================================
def _lim(a, factor):
    nz = a[a != 0]
    return (np.percentile(nz, 99) * factor) if nz.size else 1.0


def _show(ax, img, channel, label, factor, lim=None):
    # lim=None scales each panel to itself. Pass an explicit lim to force two panels
    # onto one scale -- required for GT vs PREDICTION, or a prediction that is
    # uniformly 10x too bright looks identical to the truth.
    if lim is None:
        lim = _lim(img, factor)
    if "hmi" not in channel:
        ax.imshow(img, cmap=f"sdo{channel}", vmin=0, vmax=lim)
        fc = "w"
    else:
        fc = "k"
        ax.imshow(img, cmap="coolwarm" if "_v" in channel else "hmimag",
                  vmin=-lim, vmax=lim)
    ax.text(0.01, 0.99, label, transform=ax.transAxes, ha="left", va="top",
            color=fc, fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])


def report_and_plot(ds, idx, split, lit, cfg, orient, args, ch_idx,
                    channels_to_plot, scale_versions, weights_tag):
    item = ds[idx]

    if orient == "FLIPPED":
        # Flip input AND target together -- the same contract the dataset's own flip
        # honours. Flipping only the input would ask the model for a mirror image.
        item = dict(item)
        item["ts"]       = np.ascontiguousarray(np.flip(item["ts"], axis=-2))
        item["forecast"] = np.ascontiguousarray(np.flip(item["forecast"], axis=-2))

    n_inputs = item["ts"].shape[1]
    n_lead   = item["forecast"].shape[1]
    assert n_lead == 1, f"this plot assumes one target frame; got {n_lead}. Set rollout_steps: 0."
    channel_order = cfg.data.channels

    # ---- 0. RUN THE MODEL ----
    lit = lit.to("cuda").eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _b = default_collate([item])
        _b = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in _b.items()}
        pred_norm = lit(_b)[0].float().cpu().numpy()
    del _b
    torch.cuda.empty_cache()

    # ---- 1. TRUE FRAME TIMESTAMPS ----
    reference_timestep = pd.Timestamp(ds.valid_indices[idx])
    now_offset_min = sorted(cfg.data.time_delta_input_minutes)[-1]
    now_time = reference_timestep + pd.Timedelta(minutes=now_offset_min)
    input_times  = [now_time - pd.Timedelta(minutes=round(float(h) * 60))
                    for h in item["time_delta_input"]]
    target_times = [now_time - pd.Timedelta(minutes=round(float(h) * 60))
                    for h in item["lead_time_delta"]]

    # ---- 2. PROVENANCE ----
    row = ds.df_valid_indices.iloc[idx]
    anchor_col   = cfg.data.ds_time_column
    anchor_event = pd.Timestamp(item["ds_index"])
    gt_time      = target_times[0]
    gap          = abs(gt_time - anchor_event)

    print("\n" + "=" * 100)
    print(f"{split.upper()}  sample {idx}  |  {orient}  |  weights = {weights_tag}")
    print("=" * 100)
    print(f"FLARE EVENT      : {row['GOES_class']}")
    print(f"  start_time     : {row['start_time']}")
    print(f"  peak_time      : {row['peak_time']}")
    print(f"  end_time       : {row['end_time']}")
    print(f"Ground Truth     : {gt_time}   (gap from {anchor_col}: "
          f"{gap.total_seconds()/60:.0f} min)")

    # ---- 2b. PREDICTION QUALITY ----
    _gt_n  = item["forecast"][:, 0]      # normalized space -- what the loss sees
    _now_n = item["ts"][:, -1]           # persistence: copy the "now" frame forward
    mse_model = float(np.mean([((pred_norm[k] - _gt_n[k]) ** 2).mean()
                               for k in range(len(channel_order))]))
    mse_persist = float(np.mean([((_now_n[k] - _gt_n[k]) ** 2).mean()
                                 for k in range(len(channel_order))]))
    print("\nNORMALIZED-SPACE MSE (all 13 channels)")
    print(f"  model              : {mse_model:.5f}")
    print(f"  persistence        : {mse_persist:.5f}   <- copy 'now' forward, no model")
    print(f"  skill vs persist.  : {1 - mse_model/mse_persist:+.1%}   (>0 = model helps)")
    print("\nPER-CHANNEL RMSE, physical units")
    print(f"  {'channel':<10} {'model':>12} {'persistence':>14}")
    for c, n in ch_idx.items():
        rm = float(np.sqrt(((pred_norm[n] - _gt_n[n]) ** 2).mean()))
        rp = float(np.sqrt(((_now_n[n]    - _gt_n[n]) ** 2).mean()))
        print(f"  {c:<10} {rm:>12.4g} {rp:>14.4g}")

    # ---- 3. PHYSICAL-UNIT FRAMES ----
    inv, _c, S = ds.inverse_transform_data, np.ascontiguousarray, args.stride
    ins_phys  = [inv(_c(item["ts"][:, t, ::S, ::S]))       for t in range(n_inputs)]
    tgts_phys = [inv(_c(item["forecast"][:, l, ::S, ::S])) for l in range(n_lead)]
    prds_phys = [inv(_c(pred_norm[:, ::S, ::S]))]
    frames = {c: ([a[n] for a in ins_phys],
                  [a[n] for a in tgts_phys],
                  [a[n] for a in prds_phys]) for c, n in ch_idx.items()}

    # ---- 4. PLOTTING ----
    n_out    = 2 * n_lead
    n_cols   = n_inputs + 1 + n_out
    w_ratios = [1] * n_inputs + [0.10] + [1] * n_out
    saved = []

    for tag, factor in scale_versions:
        n_rows   = len(channels_to_plot) * 2
        panel    = 3.4
        title_in = 0.42
        fig_w = panel * (n_inputs + n_out) + panel * w_ratios[n_inputs]
        fig_h = panel * n_rows + title_in
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=110)
        gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                               width_ratios=w_ratios, wspace=0.02, hspace=0.03,
                               left=0.004, right=0.996,
                               top=1 - title_in / fig_h, bottom=0.004)

        for r, c in enumerate(channels_to_plot):
            ins, tgts, prds = frames[c]
            for t in range(n_inputs):
                _show(fig.add_subplot(gs[r, t]), ins[t], c,
                      f"{c}\n{input_times[t]}\nINPUT {t}", factor)
            for l in range(n_lead):
                gt_lim = _lim(tgts[l], factor)
                _show(fig.add_subplot(gs[r, n_inputs + 1 + l]), tgts[l], c,
                      f"{c}\n{target_times[l]}\nGROUND TRUTH", factor)
                _show(fig.add_subplot(gs[r, n_inputs + 1 + n_lead + l]), prds[l], c,
                      f"{c}\n{target_times[l]}\nPREDICTION", factor, lim=gt_lim)

        for r, c in enumerate(channels_to_plot):
            ins, tgts, prds = frames[c]
            diffs = []
            if n_inputs >= 2:
                diffs.append((ins[-1] - ins[-2],
                              f"{c}  \u0394 inputs\n{input_times[-1]} \u2212 {input_times[-2]}", None))
            d_signal = tgts[0] - ins[-1]
            lim_sig = np.percentile(np.abs(d_signal), 99) * factor or 1.0
            diffs.append((d_signal,
                          f"{c}  \u0394 to ground truth\n{target_times[0]} \u2212 {input_times[-1]}", lim_sig))
            diffs.append((prds[0] - tgts[0],
                          f"{c}  \u0394 prediction error\nPREDICTION \u2212 GROUND TRUTH", lim_sig))

            gap_u = w_ratios[n_inputs]
            n_d   = len(diffs)
            side  = (sum(w_ratios) - (n_d + gap_u * (n_d - 1))) / 2
            ratios = [side]
            for k in range(n_d):
                if k:
                    ratios.append(gap_u)
                ratios.append(1)
            ratios.append(side)
            inner = gs[len(channels_to_plot) + r, :].subgridspec(
                1, len(ratios), width_ratios=ratios, wspace=0)
            for k, (d, title, forced) in enumerate(diffs):
                ax = fig.add_subplot(inner[0, 2 * k + 1])
                lim = forced if forced is not None else (np.percentile(np.abs(d), 99) * factor or 1.0)
                im = ax.imshow(d, cmap="coolwarm", vmin=-lim, vmax=lim)
                ax.text(0.01, 0.99, f"{title}\nRMSE {np.sqrt((d**2).mean()):.4g}",
                        transform=ax.transAxes, ha="left", va="top", fontsize=7, color="k",
                        bbox=dict(fc="w", ec="none", alpha=0.65, pad=1.5))
                ax.set_xticks([]); ax.set_yticks([])
                cax = ax.inset_axes([1.015, 0.0, 0.022, 1.0])
                fig.colorbar(im, cax=cax).ax.tick_params(labelsize=6)

        fig.suptitle(
            f"[{weights_tag}]  {split.upper()} sample {idx}  {orient}   "
            f"[{tag} scaling, 99th pct \u00d7 {factor}]   {row['GOES_class']}   "
            f"anchor='{anchor_col}' @ {anchor_event}   |   "
            f"GT frame {gt_time} (gap {gap.total_seconds()/60:.0f} min)   |   "
            f"label={float(item['ground_truth']):.4g}   |   "
            f"MSE {mse_model:.4f} vs persistence {mse_persist:.4f}",
            fontsize=10, y=0.999)

        # Filename carries weights + event + orientation, so a pretrained run and a
        # fine-tuned run of the same event sit next to each other and sort together.
        safe_time = str(row["peak_time"])[:16].replace(":", "").replace(" ", "_")
        win = "w" + "_".join(str(abs(int(m))) for m in cfg.data.time_delta_input_minutes)
        fname = (f"{weights_tag}__{split}__{row['GOES_class']}__{safe_time}"
                 f"__{orient}__{tag}__{win}__s{args.stride}.png")
        path = os.path.join(args.outdir, fname)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)
        print(f"  saved -> {path}")

    del frames, ins_phys, tgts_phys, prds_phys, pred_norm, item
    gc.collect()

    return dict(weights=weights_tag, split=split, i=idx, orientation=orient,
                GOES_class=row["GOES_class"], peak_time=str(row["peak_time"]),
                time=str(reference_timestep)[:16], mse_model=mse_model,
                mse_persistence=mse_persist, skill=1 - mse_model / mse_persist,
                figures=";".join(saved))


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    channels_to_plot = [c.strip() for c in args.channels.split(",")]
    scale_map = {"normal": 1, "enhanced": 10}
    scale_versions = [(s.strip(), scale_map[s.strip()]) for s in args.scales.split(",")]

    print("building config, data and model ...")
    input_minutes = None
    if args.input_minutes:
        input_minutes = [int(x) for x in args.input_minutes.split(",")]
    objs = surya_setup.build_everything(args.config, input_minutes=input_minutes)
    cfg, lit_model = objs["cfg"], objs["lit_model"]

    # ---- weights ----
    if args.weights == "pretrained":
        weights_tag = "pretrained"
    else:
        surya_setup.load_finetuned(lit_model, args.weights)
        weights_tag = os.path.splitext(os.path.basename(args.weights))[0]

    mx, verdict = surya_setup.weight_provenance(lit_model)
    print(f"\nmax|B| = {mx:.3e}  ->  {verdict}")
    if args.weights == "pretrained" and verdict != "STOCK PRETRAINED":
        raise SystemExit("asked for pretrained but adapters are non-zero -- aborting")
    if args.weights != "pretrained" and verdict == "STOCK PRETRAINED":
        print("WARNING: checkpoint loaded but max|B| is still 0. Either the run barely "
              "moved the adapters, or the checkpoint did not apply. Check the missing/"
              "unexpected key counts above.")

    ch_idx = {c: cfg.data.channels.index(c) for c in channels_to_plot}

    # ---- pick the events ----
    ds = objs[f"{args.split}_ds"]
    ds.flip_probability = 0.0          # orientation is rendered explicitly below

    if args.goes_class:
        classes = tuple(args.goes_class.upper())
        idxs = surya_setup.pick_by_class(ds, classes=classes)
        if args.peak_time:
            df = ds.df_valid_indices
            keep = df["peak_time"].astype(str).str.startswith(args.peak_time)
            idxs = [i for i in idxs if keep.iloc[i]]
        if not idxs:
            avail = ds.df_valid_indices["GOES_class"].str[0].value_counts().to_dict()
            raise SystemExit(
                f"no {args.goes_class}-class event in the {args.split} set.\n"
                f"available classes there: {avail}\n"
                f"Note the split is a {len(ds)}-flare random subset -- the full catalog "
                f"may hold one that was not drawn. Widen N_VAL/N_TEST in surya_setup.py "
                f"or query objs['catalog'] for the uncapped list."
            )
        print(f"\n{args.goes_class}-class events in {args.split}: positions {idxs}")
    else:
        idxs = list(range(len(ds)))

    idxs = idxs[:args.n_show]
    orientations = ["UPRIGHT", "FLIPPED"] if args.both else ["UPRIGHT"]

    summary = []
    for idx in idxs:
        for o in orientations:
            summary.append(report_and_plot(ds, idx, args.split, lit_model, cfg, o,
                                           args, ch_idx, channels_to_plot,
                                           scale_versions, weights_tag))
            gc.collect()
            torch.cuda.empty_cache()

    df = pd.DataFrame(summary)
    csv_path = os.path.join(args.outdir, f"summary_{weights_tag}_{args.split}.csv")
    df.to_csv(csv_path, index=False)
    print("\n" + df.drop(columns=["figures"]).to_string(index=False))
    print(f"\nsummary -> {csv_path}")

    if args.both and len(df) >= 2:
        piv = df.pivot_table(index=["i"], columns="orientation", values="mse_model")
        if {"UPRIGHT", "FLIPPED"}.issubset(piv.columns):
            delta = piv["FLIPPED"] - piv["UPRIGHT"]
            print("\nflipped minus upright MSE (positive = flipping hurts):")
            print(delta.to_string())


if __name__ == "__main__":
    main()
