#!/usr/bin/env python
"""
Run the SAME events through stock pretrained Surya and your fine-tuned checkpoint,
then write a side-by-side comparison.

This is the endgoal of the flip experiment: one command that produces, for each
event, four figures (pretrained/finetuned x upright/flipped) plus a table of the
MSE deltas, all in one folder you can open and scroll visually.

Examples
--------
python compare_runs.py --ckpt finetuned_32steps.ckpt --split test --goes-class MX
python compare_runs.py --ckpt ft512.ckpt --split val --goes-class X --n-show 2 --stride 1

Output
------
figures/
    pretrained__test__M2.1__2023-...__UPRIGHT__normal__s4.png
    pretrained__test__M2.1__2023-...__FLIPPED__normal__s4.png
    ft512__test__M2.1__2023-...__UPRIGHT__normal__s4.png
    ft512__test__M2.1__2023-...__FLIPPED__normal__s4.png
    comparison.csv
    comparison.html          <- all four side by side in a browser
"""
import argparse
import os
import subprocess
import sys

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="path to your fine-tuned .ckpt")
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--goes-class", default="MX")
    p.add_argument("--peak-time", default=None)
    p.add_argument("--n-show", type=int, default=1)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--outdir", default="figures")
    p.add_argument("--input-minutes", default=None,
                   help="override the input window, e.g. '-120,-60'")
    p.add_argument("--no-flip", action="store_true",
                   help="upright only (default renders both orientations)")
    return p.parse_args()


def run_one(weights, args):
    cmd = [sys.executable, "run_plots.py",
           "--weights", weights,
           "--split", args.split,
           "--n-show", str(args.n_show),
           "--stride", str(args.stride),
           "--outdir", args.outdir]
    if args.goes_class:
        cmd += ["--goes-class", args.goes_class]
    if args.peak_time:
        cmd += ["--peak-time", args.peak_time]
    if args.input_minutes:
        cmd += ["--input-minutes=" + args.input_minutes]
    if not args.no_flip:
        cmd += ["--both"]
    print("\n" + "=" * 70)
    print(" ".join(cmd))
    print("=" * 70)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(f"run_plots.py failed for weights={weights}")


def build_html(df, outdir):
    """A plain page showing each event's four panels together, so the comparison
    is one scroll in a browser instead of four files opened by hand."""
    rows = []
    for (cls, peak), grp in df.groupby(["GOES_class", "peak_time"], sort=True):
        cells = []
        for _, r in grp.sort_values(["weights", "orientation"]).iterrows():
            for fig in str(r["figures"]).split(";"):
                if not fig:
                    continue
                rel = os.path.basename(fig)
                cells.append(
                    f'<figure><figcaption>{r["weights"]} &middot; '
                    f'{r["orientation"]} &middot; MSE {r["mse_model"]:.4f} &middot; '
                    f'skill {r["skill"]:+.1%}</figcaption>'
                    f'<a href="{rel}"><img src="{rel}"></a></figure>'
                )
        rows.append(f"<section><h2>{cls} &mdash; {peak}</h2>"
                    f'<div class="grid">{"".join(cells)}</div></section>')

    html = f"""<!doctype html>
<meta charset="utf-8">
<title>Surya flip experiment &mdash; comparison</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.5 system-ui, sans-serif; margin: 2rem; max-width: 100%; }}
  h1 {{ font-size: 1.4rem; }}
  h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #8884; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: 1.5rem; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 13px; opacity: .8; margin-bottom: .3rem; }}
  img {{ width: 100%; height: auto; border: 1px solid #8884; }}
  table {{ border-collapse: collapse; font-size: 13px; }}
  th, td {{ border: 1px solid #8884; padding: .25rem .5rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
</style>
<h1>Surya flip experiment</h1>
<p>Each event rendered with stock pretrained weights and with the fine-tuned
checkpoint, upright and flipped. Click a figure to open it full size.</p>
<h2>Summary</h2>
{df.drop(columns=["figures"]).to_html(index=False, float_format=lambda v: f"{v:.4f}")}
{"".join(rows)}
"""
    path = os.path.join(outdir, "comparison.html")
    with open(path, "w") as fh:
        fh.write(html)
    return path


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    run_one("pretrained", args)
    run_one(args.ckpt, args)

    ft_tag = os.path.splitext(os.path.basename(args.ckpt))[0]
    parts = []
    for tag in ("pretrained", ft_tag):
        csv = os.path.join(args.outdir, f"summary_{tag}_{args.split}.csv")
        if os.path.exists(csv):
            parts.append(pd.read_csv(csv))
    if not parts:
        raise SystemExit("no summary CSVs were written -- check the logs above")

    df = pd.concat(parts, ignore_index=True)
    out_csv = os.path.join(args.outdir, "comparison.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(df.drop(columns=["figures"]).to_string(index=False))

    # does fine-tuning help, and does flipping hurt?
    piv = df.pivot_table(index=["GOES_class", "peak_time", "orientation"],
                         columns="weights", values="mse_model")
    if {"pretrained", ft_tag}.issubset(piv.columns):
        piv["finetuned_minus_pretrained"] = piv[ft_tag] - piv["pretrained"]
        print("\nMSE by weights (negative delta = fine-tuning helped):")
        print(piv.to_string())

    piv2 = df.pivot_table(index=["weights", "GOES_class", "peak_time"],
                          columns="orientation", values="mse_model")
    if {"UPRIGHT", "FLIPPED"}.issubset(piv2.columns):
        piv2["flipped_minus_upright"] = piv2["FLIPPED"] - piv2["UPRIGHT"]
        print("\nMSE by orientation (positive delta = flipping hurts):")
        print(piv2.to_string())

    html = build_html(df, args.outdir)
    print(f"\ncsv  -> {out_csv}")
    print(f"html -> {html}")
    print(f"pngs -> {args.outdir}/")


if __name__ == "__main__":
    main()
