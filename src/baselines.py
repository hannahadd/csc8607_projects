import yaml
import pandas as pd
import numpy as np

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["dataset"]["annotation_csv"]
    split = config["dataset"]["split"]

    df = pd.read_csv(csv_path)

    # test split
    df_test = df[df["fold"].isin(split["test_folds"])].copy()

    # majority baseline
    counts = df_test["category"].value_counts()
    majority_class = counts.index[0]
    majority_acc = counts.iloc[0] / len(df_test)

    # random uniform baseline: expected accuracy = 1/num_classes
    num_classes = df["category"].nunique()
    random_acc = 1.0 / num_classes

    print("TEST size:", len(df_test))
    print("Num classes:", num_classes)
    print("Majority class:", majority_class, "count:", int(counts.iloc[0]))
    print("Majority baseline accuracy:", majority_acc)
    print("Random-uniform baseline accuracy:", random_acc)

if __name__ == "__main__":
    main()
