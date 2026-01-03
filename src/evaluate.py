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


def _confusion_matrix_from_preds(targets: torch.Tensor, preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    # targets, preds: (N,) int64 on CPU
    idx = targets * num_classes + preds
    cm = torch.bincount(idx, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def macro_f1_from_cm(cm: torch.Tensor, eps: float = 1e-12) -> float:
    # cm: (C, C) where rows=true, cols=pred
    tp = torch.diag(cm).float()
    fp = cm.sum(dim=0).float() - tp
    fn = cm.sum(dim=1).float() - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    return float(f1.mean().item())


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

    if not isinstance(state, dict):
        raise ValueError("Checkpoint format not recognized: expected a state_dict or a dict containing one.")

    # Strip "model." prefix if needed
    if any(key.startswith("model.") for key in state.keys()):
        state = {key.replace("model.", "", 1): val for key, val in state.items()}

    return state


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int):
    model.eval()
    total_loss = 0.0
    total = 0

    # Pour macro-F1 (via matrice de confusion)
    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.int64)  # CPU

    # Pour top-5 accuracy
    top5_correct = 0

    # Pour top-1 accuracy
    correct = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)  # (B, C)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        total += bs

        # top-1 preds
        preds = logits.argmax(dim=1)  # (B,)
        correct += int((preds == yb).sum().item())

        # confusion matrix (CPU)
        cm_total += _confusion_matrix_from_preds(
            targets=yb.detach().cpu().long(),
            preds=preds.detach().cpu().long(),
            num_classes=num_classes,
        )

        # top-5 accuracy
        top5 = logits.topk(5, dim=1).indices  # (B, 5)
        top5_correct += int((top5 == yb.unsqueeze(1)).any(dim=1).sum().item())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    f1_macro = macro_f1_from_cm(cm_total)
    top5_acc = top5_correct / max(total, 1)

    return avg_loss, acc, f1_macro, top5_acc, total


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

    test_loss, test_acc, test_f1_macro, test_top5_acc, n = evaluate(
        model, test_loader, device, criterion, num_classes=meta["num_classes"]
    )

    print(f"test_loss: {test_loss:.4f}")
    print(f"test_accuracy: {test_acc:.4f}")
    print(f"test_f1_macro: {test_f1_macro:.4f}")
    print(f"test_top5_accuracy: {test_top5_acc:.4f}")


if __name__ == "__main__":
    main()
