"""Train JigNet.

    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --set train.epochs=2 data.root=/content/drive/MyDrive/.../jigsaw_dataset

Everything lands in runs/<name>/ : config.yaml, environment.json, metrics.csv, history.json,
checkpoints/checkpoint_epoch_XXX.pth (resume), last.pth (= `jignet_weights.pth` of the notebook),
best_val_loss.pth, curves.png.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import JigsawPairDataset
from models.checkpoint import get_latest_checkpoint, save_checkpoint
from models.inference import normalize_transform
from models.jignet import JigNet
from utils.config import load_config, parse_args
from utils.runs import make_run_dir
from utils.seed import make_generator, seed_worker, set_seed


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, correct, total = 0.0, 0, 0
    preds, targets = [], []
    with torch.set_grad_enabled(train):
        for a, b, y in loader:
            a, b, y = a.to(device), b.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(a, b)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * y.size(0)
            p = logits.argmax(1)
            correct += (p == y).sum().item()
            total += y.size(0)
            preds.append(p.cpu().numpy())
            targets.append(y.cpu().numpy())
    f1 = f1_score(np.concatenate(targets), np.concatenate(preds), average="macro", zero_division=0)
    return loss_sum / total, correct / total, f1


def plot_curves(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    ax[0].plot(df.epoch, df.train_loss, label="train"); ax[0].plot(df.epoch, df.val_loss, label="val")
    ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(df.epoch, df.train_acc, label="train"); ax[1].plot(df.epoch, df.val_acc, label="val")
    ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].grid(True)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close(fig)


def main():
    args = parse_args("Train JigNet")
    cfg = load_config(args.config, args.overrides)
    set_seed(cfg.seed, cfg.get("deterministic", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = make_run_dir(cfg.output.runs_dir, cfg.name, cfg)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    print(f"device: {device} | run dir: {run_dir}")

    # ---- data
    tf = normalize_transform(cfg.data.mean, cfg.data.std)      # no augmentation (as in the notebook)
    pairs_csv = str(Path(cfg.data.root) / cfg.data.pairs_csv)
    legacy = cfg.data.get("legacy_prefix")
    ds = {s: JigsawPairDataset(pairs_csv, s, cfg.data.root, tf, legacy) for s in ("train", "val")}
    common = dict(batch_size=cfg.train.batch_size, num_workers=cfg.data.num_workers, pin_memory=device.type == "cuda",
                  worker_init_fn=seed_worker)
    train_loader = DataLoader(ds["train"], shuffle=True, generator=make_generator(cfg.seed), **common)
    val_loader = DataLoader(ds["val"], shuffle=False, **common)
    print(f"train pairs: {len(ds['train'])} | val pairs: {len(ds['val'])}")

    # ---- class weights
    cw = cfg.train.class_weights
    if cw == "auto":
        counts = ds["train"].data["label"].value_counts().sort_index()
        cw = (len(ds["train"]) / (cfg.model.num_classes * counts)).tolist()
    class_weights = torch.tensor(cw, dtype=torch.float32).to(device)
    print("class weights:", [round(float(w), 4) for w in class_weights])

    # ---- model / optimisation
    model = JigNet(num_classes=cfg.model.num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.eta_min)

    hist = {k: [] for k in ("epoch", "train_loss", "train_acc", "train_f1", "val_loss", "val_acc", "val_f1", "lr")}
    start_epoch = 0
    latest = get_latest_checkpoint(ckpt_dir) if cfg.train.resume else None
    if latest:
        ck = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state_dict"])
        optimizer.load_state_dict(ck["optimizer_state_dict"])
        scheduler.load_state_dict(ck["scheduler_state_dict"])
        hist, start_epoch = ck["history"], ck["epoch"]
        print(f"resumed from {latest} (epoch {start_epoch})")

    best_val = min(hist["val_loss"]) if hist["val_loss"] else float("inf")
    for epoch in range(start_epoch, cfg.train.epochs):
        t0 = time.time()
        lr = optimizer.param_groups[0]["lr"]
        tr = run_epoch(model, train_loader, criterion, device, optimizer)
        va = run_epoch(model, val_loader, criterion, device)
        scheduler.step()
        hist["epoch"].append(epoch + 1)
        hist["train_loss"].append(tr[0]); hist["train_acc"].append(tr[1]); hist["train_f1"].append(tr[2])
        hist["val_loss"].append(va[0]); hist["val_acc"].append(va[1]); hist["val_f1"].append(va[2])
        hist["lr"].append(lr)
        print(f"Epoch {epoch + 1:03d}/{cfg.train.epochs} | train loss {tr[0]:.4f} acc {tr[1]:.2%} | "
              f"val loss {va[0]:.4f} acc {va[1]:.2%} macroF1 {va[2]:.4f} | {time.time() - t0:.0f}s")

        if va[0] < best_val:
            best_val = va[0]
            torch.save(model.state_dict(), run_dir / "best_val_loss.pth")
        if (epoch + 1) % cfg.train.checkpoint_every == 0 or epoch + 1 == cfg.train.epochs:
            save_checkpoint(ckpt_dir / f"checkpoint_epoch_{epoch + 1:03d}.pth", model, optimizer, scheduler,
                            epoch + 1, hist, dict(cfg))
        df = pd.DataFrame(hist)
        df.to_csv(run_dir / "metrics.csv", index=False)

    torch.save(model.state_dict(), run_dir / "last.pth")          # bare weights, loadable by app.py / eval.py
    (run_dir / "history.json").write_text(json.dumps(hist, indent=2))
    if len(hist["epoch"]):
        plot_curves(pd.DataFrame(hist), run_dir / "curves.png")
        i = int(np.argmax(hist["val_acc"]))
        print(f"best val acc {hist['val_acc'][i]:.2%} @ epoch {hist['epoch'][i]} | "
              f"best val loss {min(hist['val_loss']):.4f} | final weights -> {run_dir / 'last.pth'}")


if __name__ == "__main__":
    main()
