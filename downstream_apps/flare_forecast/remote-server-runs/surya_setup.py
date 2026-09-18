"""
Shared setup for the flip experiment: config, scalers, datasets, loaders, model.

Everything here is lifted from notebook cells 2-11 with no logic changes, so a
script that imports this gets the exact same objects the notebook builds.

Run from:  <repo_root>/downstream_apps/flare_forecast/
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # must precede `import torch`

import sys
sys.path.append("../../")

import copy
from functools import partial

import numpy as np
import pandas as pd
import torch
import lightning as L
from torch.utils.data import DataLoader

from workshop_infrastructure.utils import build_scalers, apply_peft_lora, load_pretrained_weights
from workshop_infrastructure.assets import ensure_assets
from workshop_infrastructure.models.finetune_models import HelioSpectformer2D
from workshop_infrastructure.datasets.builders import build_helio_datasets, _seed_worker
from downstream_apps.flare_forecast.configs import load_flare_config
from downstream_apps.flare_forecast.datasets.flare_dataset import flare_Dataset
from downstream_apps.flare_forecast.metrics.flare_metrics import FlareMetrics
from downstream_apps.flare_forecast.lightning_modules.pl_flare_baseline import FlareLightningModule

torch.set_float32_matmul_precision("medium")


# ============================== knobs ========================================
FLIP_PROBABILITY = 0.5        # training mix: 0.0 upright | 0.5 mixed | 1.0 flipped
N_TRAIN, N_VAL, N_TEST = 32, 16, 16
VAL_YEARS  = (2014, 2015)     # cycle 24 maximum
TEST_YEARS = (2023,)          # cycle 25
SUBSET_SEED = 42

# On the 248 GB box these can go far higher than the 8/1 the workshop machine
# needed. Host RAM per active loader ~ workers x PREFETCH x 2.6 GB, so 16 x 2 is
# about 83 GB -- comfortable here, fatal on a 60 GB machine.
LOADER_WORKERS = int(os.environ.get("SURYA_WORKERS", 16))
PREFETCH       = int(os.environ.get("SURYA_PREFETCH", 2))
# =============================================================================


def pick_stratified(ds, n, years=None, seed=0):
    """Random subset, spread as evenly as possible across the available years."""
    times = pd.DatetimeIndex(ds.valid_indices)
    pool = np.arange(len(times))
    if years is not None:
        pool = pool[np.isin(times.year, list(years))]
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} samples available for years={years}, need {n}")
    rng = np.random.default_rng(seed)
    buckets = {}
    for p in pool:
        buckets.setdefault(times[p].year, []).append(p)
    for y in buckets:
        rng.shuffle(buckets[y])
    picked = []
    while len(picked) < n:
        for y in sorted(buckets):
            if buckets[y] and len(picked) < n:
                picked.append(buckets[y].pop())
    return sorted(picked)


def take(ds, positions):
    """Shrink a dataset to `positions`. These three attributes ARE the sample list.

    NOTE: this mutates in place. If you want to keep the uncapped catalog around,
    pass copy.copy(ds) instead of ds.
    """
    ds.valid_indices    = [ds.valid_indices[i] for i in positions]
    ds.df_valid_indices = ds.df_valid_indices.iloc[positions]
    ds.adjusted_length  = len(positions)
    return ds


def pick_by_class(ds, classes=("M", "X"), years=None):
    """Positions of every flare in `ds` whose GOES class starts with one of `classes`.

    This is the 'give me an X-class event' selector. Returns positions into the
    dataset as it currently stands, so call it on the UNCAPPED set if you want
    the full pool rather than whatever pick_stratified happened to draw.
    """
    df = ds.df_valid_indices
    mask = df["GOES_class"].str.startswith(tuple(classes))
    if years is not None:
        mask &= np.isin(pd.DatetimeIndex(df["peak_time"]).year, list(years))
    return np.where(mask.values)[0].tolist()


def build_everything(config_path="./configs/config_script.yaml", verbose=True,
                     input_minutes=None):
    """Cells 2-11, in order. Returns a dict of every object the notebook defines.

    input_minutes overrides cfg.data.time_delta_input_minutes, e.g. [-120, -60]
    instead of the config's [-60, -36]. The list length must stay the same as
    cfg.model.time_embedding.time_dim (2), or the backbone shape will not match.
    """
    cfg = load_flare_config(config_path)

    if input_minutes is not None:
        old = list(cfg.data.time_delta_input_minutes)
        if len(input_minutes) != len(old):
            raise ValueError(
                f"input_minutes must have the same length as the config's "
                f"{old} (time_dim is fixed by the pretrained backbone); "
                f"got {input_minutes}"
            )
        cfg.data.time_delta_input_minutes = list(input_minutes)
        if verbose:
            print(f"input window overridden: {old} -> {list(input_minutes)}")

    if verbose:
        print(f"job={cfg.job_id}  batch_size={cfg.batch_size}  dtype={cfg.dtype}  "
              f"inputs={cfg.data.time_delta_input_minutes}  anchor={cfg.data.ds_time_column}")

    ensure_assets(cfg, which=["scalers", "weights"])
    scalers = build_scalers(info=cfg.data.scalers_path)
    if verbose:
        print(f"scalers for {len(scalers)} channels")

    # ---- datasets -----------------------------------------------------------
    train_full, val_full = build_helio_datasets(
        cfg, flare_Dataset, scalers=scalers,
        return_surya_stack=True,
        max_number_of_samples=None,
        flip_probability=0.0,
        ds_flare_index_path=cfg.data.flare_index_path,
        ds_time_column=cfg.data.ds_time_column,
        ds_time_tolerance=cfg.data.ds_time_tolerance,
        ds_match_direction=cfg.data.ds_match_direction,
    )
    test_full = copy.copy(val_full)

    # Keep an UNCAPPED copy of the catalog before take() shrinks anything. This is
    # what you query for "is there an X-class flare in year Y" -- the capped sets
    # only hold whatever pick_stratified drew.
    catalog = val_full.df_valid_indices.copy()

    train_ds = take(train_full, pick_stratified(train_full, N_TRAIN, seed=SUBSET_SEED))
    val_ds   = take(val_full,   pick_stratified(val_full,  N_VAL,  years=VAL_YEARS,  seed=SUBSET_SEED))
    test_ds  = take(test_full,  pick_stratified(test_full, N_TEST, years=TEST_YEARS, seed=SUBSET_SEED + 1))

    train_ds.flip_probability = FLIP_PROBABILITY
    val_ds.flip_probability = test_ds.flip_probability = 0.0

    def make_loader(ds, shuffle):
        kw = dict(batch_size=1, num_workers=LOADER_WORKERS, pin_memory=True, drop_last=False)
        if LOADER_WORKERS > 0:
            kw.update(
                multiprocessing_context="spawn",   # the boto3 handle does not survive fork
                persistent_workers=False,          # workers die at pass end, so RAM is released
                prefetch_factor=PREFETCH,
                worker_init_fn=partial(_seed_worker, base_seed=SUBSET_SEED),
            )
        g = torch.Generator(); g.manual_seed(SUBSET_SEED)
        return DataLoader(ds, shuffle=shuffle, generator=(g if shuffle else None), **kw)

    train_data_loader = make_loader(train_ds, shuffle=True)
    val_data_loader   = make_loader(val_ds,   shuffle=False)
    test_data_loader  = make_loader(test_ds,  shuffle=False)

    if verbose:
        for _n, _d in (("train", train_ds), ("val", val_ds), ("test", test_ds)):
            _t = pd.DatetimeIndex(_d.valid_indices)
            _c = _d.df_valid_indices["GOES_class"].str[0].value_counts().to_dict()
            print(f"{_n:5s} {len(_d):3d} flares | years {dict(sorted(_t.year.value_counts().items()))} "
                  f"| class {dict(sorted(_c.items()))} | flip_p {_d.flip_probability}")
        print(f"\n{LOADER_WORKERS} workers x {PREFETCH} prefetch "
              f"~ {LOADER_WORKERS * PREFETCH * 2.6:.0f} GB host RAM per active loader")

    # ---- model --------------------------------------------------------------
    model = HelioSpectformer2D.from_config(
        cfg.model,
        ft_out_chans=len(cfg.data.channels),
        ft_unembedding_type="linear",
        dtype=cfg.dtype,
        use_latitude_in_learned_flow=cfg.use_latitude_in_learned_flow,
    )
    load_pretrained_weights(model, cfg.model.pretrained_path)

    # load_pretrained_weights only tries "k" and "backbone.k", so the decoder
    # (checkpoint "unembed.*" -> model "head_unembed.*") is dropped silently.
    # Without this the output layer stays at random init and MSE parks near 1.0.
    _ckpt = torch.load(cfg.model.pretrained_path, weights_only=True, map_location="cpu")
    _msd = model.state_dict()
    _copied = []
    for _k, _v in _ckpt.items():
        _t = f"head_{_k}"
        if _t in _msd:
            assert _v.shape == _msd[_t].shape, f"shape mismatch {_t}"
            _msd[_t] = _v
            _copied.append(_k)
    assert _copied, f"no head_* match; unembed keys = {[k for k in _ckpt if k.startswith('unembed')]}"
    model.load_state_dict(_msd, strict=True)
    if verbose:
        print(f"decoder: {len(_copied)} tensors copied")
    del _ckpt, _msd

    if cfg.model.freeze_backbone:
        for name, param in model.named_parameters():
            if name.startswith("backbone."):
                param.requires_grad = False
    if cfg.model.use_lora:
        model = apply_peft_lora(model, cfg.model.lora_config)

    if verbose:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ---- lightning module ---------------------------------------------------
    class Flare2DLightningModule(FlareLightningModule):
        """FlareLightningModule with a 2D image target instead of a scalar label."""

        @staticmethod
        def _target(batch):
            fc = batch["forecast"]
            assert fc.shape[2] == 1, (
                f"expected one target frame, got {fc.shape[2]}. Set training.rollout_steps: 0."
            )
            return fc[:, :, 0].float()

        def _step(self, batch, stage):
            target = self._target(batch)
            if self.preprocess_fn is not None:
                batch = self.preprocess_fn(batch)
            output = self(batch)
            loss_fn = self.training_loss if stage == "train" else self.validation_loss
            eval_fn = self.training_evaluation if stage == "train" else self.validation_evaluation
            losses, weights = loss_fn(output, target)
            loss = self._combine_losses(losses, weights)
            self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=self.batch_size, sync_dist=True)
            for k, v in losses.items():
                self.log(f"{stage}_loss_{k}", v, batch_size=self.batch_size, sync_dist=True)
            mets, mw = eval_fn(output, target)
            if len(mw) > 0:
                for k, v in mets.items():
                    self.log(f"{stage}_metric_{k}", v, batch_size=self.batch_size, sync_dist=True)
            return loss

        def training_step(self, batch, batch_idx):   return self._step(batch, "train")
        def validation_step(self, batch, batch_idx): self._step(batch, "val")

    L.seed_everything(42, workers=True)
    metrics = {
        "train_loss":    FlareMetrics("train_loss"),
        "val_loss":      FlareMetrics("val_loss"),
        "train_metrics": FlareMetrics("train_metrics"),
        "val_metrics":   FlareMetrics("val_metrics"),
    }
    lit_model = Flare2DLightningModule(model, metrics, lr=cfg.learning_rate, batch_size=1)

    return dict(
        cfg=cfg, scalers=scalers, model=model, lit_model=lit_model,
        train_ds=train_ds, val_ds=val_ds, test_ds=test_ds,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        test_data_loader=test_data_loader,
        catalog=catalog,
        LitModuleClass=Flare2DLightningModule,
    )


def load_finetuned(lit_model, ckpt_path):
    """Load your fine-tuned checkpoint into an already-built lit_model, in place."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing, unexpected = lit_model.load_state_dict(sd, strict=False)
    print(f"loaded {ckpt_path}: {len(missing)} missing, {len(unexpected)} unexpected keys")
    del ckpt, sd
    return lit_model


def weight_provenance(lit_model):
    """max|B| over LoRA B matrices. Exactly 0 means the adapters are still at init,
    i.e. this IS stock pretrained Surya. Anything else means fine-tuned weights."""
    bs = [p for n, p in lit_model.named_parameters() if "lora_B" in n]
    if not bs:
        return None, "NO LORA ADAPTERS FOUND"
    mx = max(float(p.abs().max()) for p in bs)
    return mx, ("STOCK PRETRAINED" if mx == 0 else "FINE-TUNED")
