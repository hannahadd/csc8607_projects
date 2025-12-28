"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""
# src/evaluate.py
import argparse
import yaml
import torch
import torch.nn as nn

from src.data_loading import get_dataloaders
from src.model import build_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _extract_state_dict(ckpt_obj):
    """
    Supports checkpoints saved as:
    - plain state_dict
    - dict with keys: model_state_dict / state_dict / model / model_state
    Also strips 'model.' prefix if present.
    """
    state = ckpt_obj
    if isinstance(ckpt_obj, dict):
        for k in ["model_state_dict", "state_dict", "model", "model_state"]:
            if k in ckpt_obj:
                state = ckpt_obj[k]
                break

    # If it's still a dict but not a state_dict, try best effort
    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not recognized: expected a state_dict or a dict containing one.")

    # Strip "model." prefix if needed
    if any(key.startswith("model.") for key in state.keys()):
        state = {key.replace("model.", "", 1): val for key, val in state.items()}

    return state


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        total_loss += float(loss) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum())
        total += int(xb.size(0))

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc, total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = get_device()
    train_loader, val_loader, test_loader, meta = get_dataloaders(config)

    print(f"device: {device}")
    print(f"num_classes: {meta['num_classes']} input_shape: {meta['input_shape']}")
    print(f"TEST size: {len(test_loader.dataset)}")
    print(f"checkpoint: {args.checkpoint}")

    model = build_model(config, num_classes=meta["num_classes"]).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = _extract_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, n = evaluate(model, test_loader, device, criterion)
    print(f"test_loss: {test_loss:.4f}")
    print(f"test_accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
