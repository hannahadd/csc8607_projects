import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model
from src.utils import count_parameters


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    device = get_device()
    train_loader, _, _, meta = get_dataloaders(config)
    xb, yb = next(iter(train_loader))

    model = build_model(config).to(device)
    xb = xb.to(device)
    yb = yb.to(device)

    print("device:", device)
    print("params:", count_parameters(model))

    criterion = nn.CrossEntropyLoss()

    model.train()
    logits = model(xb)
    loss = criterion(logits, yb)
    print("initial loss:", float(loss))

    model.zero_grad(set_to_none=True)
    loss.backward()

    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item()
    print("grad_norm_sum_l2:", total_norm)
    print("log(50) =", float(torch.log(torch.tensor(50.0))))

if __name__ == "__main__":
    main()
