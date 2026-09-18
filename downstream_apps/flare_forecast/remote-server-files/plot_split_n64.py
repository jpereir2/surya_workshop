# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rcParams.update({"font.size": 14})
import surya_setup

surya_setup.N_TRAIN, surya_setup.N_VAL, surya_setup.N_TEST = 64, 32, 32
surya_setup.SUBSET_SEED = 42
objs = surya_setup.build_everything(input_minutes=[-60, -36], verbose=False)

COLOR = {"C": "#9fb8d4", "M": "#e08a3c", "X": "#a5243d"}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), gridspec_kw={"width_ratios": [4, 1, 1]})
for ax, name in zip(axes, ("train", "val", "test")):
    df = objs[name + "_ds"].df_valid_indices.copy()
    df["year"] = pd.to_datetime(df["peak_time"].astype(str)).dt.year
    df["cls"] = df["GOES_class"].astype(str).str[0]
    t = df.pivot_table(index="year", columns="cls", aggfunc="size", fill_value=0)
    t = t.reindex(columns=["C", "M", "X"], fill_value=0)
    bottom = np.zeros(len(t))
    for c in ["C", "M", "X"]:
        ax.bar(t.index.astype(str), t[c].values, bottom=bottom, color=COLOR[c],
               label="{} (n={})".format(c, int(t[c].sum())))
        bottom += t[c].values
    ax.set_title("{}: {} flares".format(name, len(df)))
    ax.set_xlabel("year")
    ax.legend(fontsize=8)
axes[0].set_ylabel("flares")
fig.tight_layout()
fig.savefig("figures/data_split_by_year_n64.png", dpi=150)
print("saved -> figures/data_split_by_year_n64.png")
