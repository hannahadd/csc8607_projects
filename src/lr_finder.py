"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""
import os
import time
import argparse
import yaml
import math

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--seed", default=None, type=int)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = int(args.seed) if args.seed is not None else int(config["train"].get("seed", 42))
    set_seed(seed)

    device = get_device()
    print("device:", device)

    # Data: use train loader only
    train_loader, _, _, meta = get_dataloaders(config)
    print("num_classes:", meta["num_classes"], "input_shape:", meta["input_shape"])

    # Model
    model = build_model(config).to(device)
    model.train()

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Paths / TB
    runs_dir = config["paths"]["runs_dir"]
    os.makedirs(runs_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"lr_finder_k{config['model']['kernel_size']}_ch{config['model']['channels']}"
    run_dir = os.path.join(runs_dir, f"{ts}_{run_name}")
    writer = SummaryWriter(log_dir=run_dir)
    print("run_dir:", run_dir)

    # LR sweep settings
    lr_start = float(config.get("lr_finder", {}).get("lr_start", 1e-6))
    lr_end = float(config.get("lr_finder", {}).get("lr_end", 1.0))
    num_iters = int(config.get("lr_finder", {}).get("num_iters", 200))

    # Optimizer (start LR, we will update each iter)
    wd = float(config["train"].get("weight_decay", 1e-4))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start, weight_decay=wd)

    # Log-space multiplier per iter
    mult = (lr_end / lr_start) ** (1 / max(1, num_iters - 1))

    best_loss = float("inf")
    global_step = 0

    data_iter = iter(train_loader)

    for i in range(num_iters):
        try:
            xb, yb = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            xb, yb = next(data_iter)

        xb = xb.to(device)
        yb = yb.to(device)

        # current LR
        lr = lr_start * (mult ** i)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)

        # stop if loss is NaN/inf
        if not torch.isfinite(loss):
            print(f"Stopped: non-finite loss at iter {i}, lr={lr}")
            break

        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu())
        best_loss = min(best_loss, loss_val)

        # TB logs (required tags)
        writer.add_scalar("lr_finder/lr", lr, global_step)
        writer.add_scalar("lr_finder/loss", loss_val, global_step)

        if (i + 1) % 20 == 0:
            print(f"iter {i+1:03d}/{num_iters} | lr={lr:.2e} | loss={loss_val:.4f}")

        # basic early stop: if loss explodes a lot
        if loss_val > 4.0 * best_loss and i > 10:
            print(f"Early stop: loss diverging (loss={loss_val:.4f}, best={best_loss:.4f}) at lr={lr:.2e}")
            break

        global_step += 1

    writer.close()
    print("LR finder done. TensorBoard logs in:", run_dir)


if __name__ == "__main__":
    main()
