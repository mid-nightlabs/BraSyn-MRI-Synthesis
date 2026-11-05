"""
Hyperparameter optimization for BraSyn MRI synthesis using Optuna.

BUG FIX vs original:
- n_train_batches_per_epoch was accepted as a parameter but NEVER USED (dead code).
  Now implemented via itertools.islice to actually limit batches per epoch,
  which can 3-5x speedup HPO trials on large datasets.

NEW: context_slices added to search space when --context_slices > 0.
"""

import argparse
import itertools
from pathlib import Path
from typing import Optional

import optuna
import torch
import yaml
from optuna.trial import Trial
from tqdm import tqdm

from dataset import MODALITIES, get_dataloaders
from model import build_model
from metrics import compute_batch_metrics
from train import _ssim_loss


def _train_epoch_limited(model, loader, optimizer, device, loss_weights,
                          max_batches: Optional[int] = None):
    """Train one epoch, optionally capped at max_batches for fast HPO."""
    import torch.nn as nn
    model.train()
    total_loss = 0.0
    n_batches = 0
    iterator = itertools.islice(loader, max_batches) if max_batches else loader
    for x, y in tqdm(iterator, desc="HPO-Train", leave=False,
                     total=max_batches or len(loader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        out = out.clamp(0.0, 1.0)
        l1 = nn.functional.l1_loss(out, y)
        ssim_l = _ssim_loss(out, y)
        loss = loss_weights["l1"] * l1 + loss_weights["ssim"] * ssim_l
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _validate_epoch(model, loader, device):
    model.eval()
    total_ssim = 0.0
    n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        out = out.clamp(0.0, 1.0)
        m = compute_batch_metrics(out, y)
        total_ssim += m["ssim"]
        n += 1
    return total_ssim / max(n, 1)


def objective(
    trial: Trial,
    data_root: str,
    target_modality: str,
    epochs_per_trial: int,
    slice_axis: int,
    patch_size: tuple,
    train_split: float,
    device: str,
    n_train_batches_per_epoch: Optional[int],  # FIX: now actually used
    search_context: bool = False,
) -> float:
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 24, 32])
    base_channels = trial.suggest_categorical("base_channels", [24, 32, 48, 64])
    depth = trial.suggest_int("depth", 3, 5)
    dropout = trial.suggest_float("dropout", 0.0, 0.3)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    loss_ssim_weight = trial.suggest_float("loss_ssim_weight", 0.2, 1.0)
    context_slices = trial.suggest_int("context_slices", 0, 2) if search_context else 0

    train_loaders, val_loaders, _, _ = get_dataloaders(
        data_root=data_root,
        slice_axis=slice_axis,
        patch_size=patch_size,
        batch_size=batch_size,
        train_split=train_split,
        context_slices=context_slices,
    )
    train_loader = train_loaders[target_modality]
    val_loader = val_loaders[target_modality]

    model = build_model(
        base_channels=base_channels,
        depth=depth,
        dropout=dropout,
        use_attention=True,
        context_slices=context_slices,
        device=device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_weights = {"l1": 1.0, "ssim": loss_ssim_weight}

    best_ssim = -1.0
    for epoch in range(1, epochs_per_trial + 1):
        # FIX: actually limit batches via n_train_batches_per_epoch
        _train_epoch_limited(model, train_loader, optimizer, device,
                              loss_weights, max_batches=n_train_batches_per_epoch)
        ssim = _validate_epoch(model, val_loader, device)
        trial.report(ssim, epoch)
        if ssim > best_ssim:
            best_ssim = ssim
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_ssim


def main():
    parser = argparse.ArgumentParser(description="BraSyn HPO with Optuna")
    parser.add_argument("--data_root", type=str, default="data/BraSyn")
    parser.add_argument("--target", type=str, default="t2f", choices=MODALITIES)
    parser.add_argument("--epochs_per_trial", type=int, default=15)
    parser.add_argument("--n_trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--study_name", type=str, default="brasyn_synthesis")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--save_best_config", type=str, default="config_best.yaml")
    # FIX: actually use this parameter now
    parser.add_argument("--max_batches_per_epoch", type=int, default=None,
                        help="Limit batches per trial epoch for faster HPO. "
                             "E.g. --max_batches_per_epoch 50 uses only 50 batches.")
    parser.add_argument("--search_context", action="store_true", default=False,
                        help="Include context_slices (2.5D) in HPO search space")
    args = parser.parse_args()

    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        data_cfg = cfg.get("data", {})
        hpo_cfg = cfg.get("hpo", {})
        args.data_root = data_cfg.get("data_root", args.data_root)
        args.n_trials = hpo_cfg.get("n_trials", args.n_trials)
        args.timeout = hpo_cfg.get("timeout", args.timeout)
        args.max_batches_per_epoch = hpo_cfg.get("max_batches_per_epoch",
                                                   args.max_batches_per_epoch)

    patch_size = (160, 160)
    slice_axis = 2
    train_split = 0.85
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(
        lambda t: objective(
            t,
            data_root=args.data_root,
            target_modality=args.target,
            epochs_per_trial=args.epochs_per_trial,
            slice_axis=slice_axis,
            patch_size=patch_size,
            train_split=train_split,
            device=device,
            n_train_batches_per_epoch=args.max_batches_per_epoch,
            search_context=args.search_context,
        ),
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    print("Best trial:", study.best_trial.number)
    print("Best SSIM:", study.best_value)
    print("Best params:", study.best_params)

    best = study.best_params
    best_config = {
        "data": {"data_root": args.data_root, "patch_size": list(patch_size),
                 "slice_axis": slice_axis,
                 "context_slices": best.get("context_slices", 0)},
        "model": {
            "base_channels": best["base_channels"],
            "depth": best["depth"],
            "dropout": best["dropout"],
            "use_attention": True,
        },
        "training": {
            "batch_size": best["batch_size"],
            "learning_rate": best["lr"],
            "weight_decay": best["weight_decay"],
            "loss_weights": {"l1": 1.0, "ssim": best["loss_ssim_weight"]},
        },
    }
    with open(args.save_best_config, "w") as f:
        yaml.dump(best_config, f, default_flow_style=False)
    print(f"Best config saved to {args.save_best_config}")


if __name__ == "__main__":
    main()
