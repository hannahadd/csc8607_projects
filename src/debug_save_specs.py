import os
import yaml
import matplotlib.pyplot as plt
from src.data_loading import get_dataloaders

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    os.makedirs("artifacts", exist_ok=True)
    train_loader, _, _, _ = get_dataloaders(config)
    xb, yb = next(iter(train_loader))

    # 6 exemples
    n = 6
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    axes = axes.flatten()

    for i in range(n):
        spec = xb[i, 0].numpy()  # (64, T)
        axes[i].imshow(spec, aspect="auto", origin="lower")
        axes[i].set_title(f"y={int(yb[i])}")
        axes[i].axis("off")

    plt.tight_layout()
    out_path = "artifacts/spec_examples_train.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved ->", out_path)

if __name__ == "__main__":
    main()
