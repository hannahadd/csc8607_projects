import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["dataset"]["annotation_csv"]
    split = config["dataset"]["split"]
    df = pd.read_csv(csv_path)

    def counts(folds):
        return df[df["fold"].isin(folds)]["category"].value_counts().sort_index()

    train_counts = counts(split["train_folds"])
    val_counts   = counts(split["val_folds"])
    test_counts  = counts(split["test_folds"])

    os.makedirs("artifacts", exist_ok=True)

    dist = pd.DataFrame({"train": train_counts, "val": val_counts, "test": test_counts}).fillna(0).astype(int)
    dist.to_csv("artifacts/class_distribution.csv")

    plt.figure(figsize=(12, 4))
    train_counts.plot(kind="bar")
    plt.tight_layout()
    plt.savefig("artifacts/class_distribution_train.png", dpi=200)
    plt.close()

    print("Saved -> artifacts/class_distribution.csv")
    print("Saved -> artifacts/class_distribution_train.png")
    print("Split sizes:", len(df[df["fold"].isin(split["train_folds"])]),
          len(df[df["fold"].isin(split["val_folds"])]),
          len(df[df["fold"].isin(split["test_folds"])]))

if __name__ == "__main__":
    main()
