"""
Entraînement principal (à implémenter par l'étudiant·e).

Doit exposer un main() exécutable via :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences minimales :
- lire la config YAML
- respecter les chemins 'runs/' et 'artifacts/' définis dans la config
- journaliser les scalars 'train/loss' et 'val/loss' (et au moins une métrique de classification si applicable)
- supporter le flag --overfit_small (si True, sur-apprendre sur un très petit échantillon)
"""
import os
import time
import argparse
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == yb).sum().item()
        total += xb.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", default=None, type=int)
    parser.add_argument("--max_steps", default=None, type=int)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Seed
    seed = int(args.seed) if args.seed is not None else int(config["train"].get("seed", 42))
    set_seed(seed)

    # Paths
    runs_dir = config["paths"]["runs_dir"]
    artifacts_dir = config["paths"]["artifacts_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # Device
    device = get_device()
    print("device:", device)

    # Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    # Overfit mode: keep only first N examples of train
    if args.overfit_small:
        n = int(config["train"].get("overfit_n", 32))
        base_ds = train_loader.dataset
        idx = list(range(min(n, len(base_ds))))
        subset = Subset(base_ds, idx)

        bs = int(config["train"]["batch_size"])
        nw = int(config["train"].get("num_workers", 0))
        pin = torch.cuda.is_available()

        train_loader = DataLoader(subset, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=pin)
        # for overfit proof, validate on the same subset (so val also goes down)
        val_loader = DataLoader(subset, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)

    # Model
    model = build_model(config, num_classes=meta["num_classes"]).to(device)
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)


    # Loss/optim
    criterion = nn.CrossEntropyLoss()
    lr = float(config["train"].get("lr", 1e-3))
    wd = float(config["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Epochs/steps
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(config["train"].get("epochs", 20))
    max_steps = int(args.max_steps) if args.max_steps is not None else None

    # TensorBoard run name
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"esc50_cnn_k{config['model']['kernel_size']}_ch{config['model']['channels']}_lr{lr}_wd{wd}"
    if args.overfit_small:
        run_name = "OVERFIT_" + run_name
    run_dir = os.path.join(runs_dir, f"{ts}_{run_name}")
    writer = SummaryWriter(log_dir=run_dir)

    # Save config snapshot (useful for reproducibility)
    snap_path = os.path.join(artifacts_dir, f"config_snapshot_{ts}.yaml")
    with open(snap_path, "w") as f:
        yaml.safe_dump(config, f)

    best_val_acc = -1.0
    global_step = 0

    print("num_classes:", meta["num_classes"], "input_shape:", meta["input_shape"])
    print("train/val/test:", meta["splits"])
    print("run_dir:", run_dir)

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            # stats
            bs = xb.size(0)
            epoch_loss += loss.item() * bs
            preds = logits.argmax(dim=1)
            epoch_correct += (preds == yb).sum().item()
            epoch_total += bs

            # TB (per step)
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = epoch_loss / max(1, epoch_total)
        train_acc = epoch_correct / max(1, epoch_total)

        val_loss, val_acc = evaluate(model, val_loader, device, criterion)

        # TB (per epoch)
        writer.add_scalar("train/acc", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)

        print(f"epoch {epoch:03d} | train_loss {train_loss:.4f} acc {train_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f}")

        # Save best checkpoint (by val accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch,
                "best_val_acc": best_val_acc,
            }
            torch.save(ckpt, os.path.join(artifacts_dir, "best.ckpt"))

        if max_steps is not None and global_step >= max_steps:
            print("Reached max_steps, stopping.")
            break

    writer.close()
    print("Training done. Best val acc:", best_val_acc)
    print("Best checkpoint ->", os.path.join(artifacts_dir, "best.ckpt"))


if __name__ == "__main__":
    main()
