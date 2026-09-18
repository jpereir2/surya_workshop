# -*- coding: utf-8 -*-
import glob, os
import pandas as pd

files = glob.glob("figures/*/summary_events.csv") + glob.glob("figures/*/comparison.csv")
rows = []
for f in sorted(files):
    d = pd.read_csv(f)
    d["run"] = os.path.basename(os.path.dirname(f))
    rows.append(d)
a = pd.concat(rows, ignore_index=True)
a["flare"] = a["GOES_class"].astype(str) + " " + a["peak_time"].astype(str).str[:10]
is120 = a["figures"].astype(str).str.contains("w120") | a["run"].str.contains("120")
a["window"] = is120.map({True: "-120/-60", False: "-60/-36"})
a = a.drop_duplicates(subset=["weights", "flare", "window", "orientation"])
keep = ["flare", "window", "weights", "orientation", "mse_model", "mse_persistence", "skill"]
a = a[keep].sort_values(["flare", "window", "weights", "orientation"])
a.to_csv("ALL_EVENTS_TABLE.csv", index=False)
pd.set_option("display.width", 200)
print(a.to_string(index=False, float_format="{:.4f}".format))

p = a.pivot_table(index=["flare", "window", "weights"], columns="orientation", values="mse_model")
p["flip_minus_upright"] = p["FLIPPED"] - p["UPRIGHT"]
print("\n=== FLIP EFFECT (positive = flipping hurts) ===")
print(p.to_string(float_format="{:.4f}".format))
p.to_csv("TABLE_flip_effect.csv")

pre = a[a["weights"] == "pretrained"].set_index(["flare", "window", "orientation"])["mse_model"]
ft = a[a["weights"] != "pretrained"].copy()
ft["pretrained_mse"] = [pre.get((r.flare, r.window, r.orientation)) for r in ft.itertuples()]
ft["finetuned_minus_pretrained"] = ft["mse_model"] - ft["pretrained_mse"]
ft = ft[["flare", "window", "weights", "orientation", "pretrained_mse", "mse_model", "finetuned_minus_pretrained"]]
print("\n=== FINE-TUNING EFFECT (negative = fine-tuning helped) ===")
print(ft.to_string(index=False, float_format="{:.4f}".format))
ft.to_csv("TABLE_finetune_effect.csv", index=False)
