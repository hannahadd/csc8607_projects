"""
Mini grid search — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.grid_search --config configs/config.yaml

Exigences minimales :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser les hparams et résultats de chaque run (ex: TensorBoard HParams ou équivalent)
"""
import os
import time
import argparse
import yaml
import copy
import subprocess
import sys
import re
import csv
from itertools import product


def _format_float(x: float) -> str:
    # name-friendly string, e.g. 0.001 -> 1p00e-03
    s = f"{x:.2e}"
    return s.replace(".", "p")


def _parse_best_val_acc(stdout: str):
    m = re.search(r"Best val acc:\s*([0-9.]+)", stdout)
    return float(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        base_cfg = yaml.safe_load(f)

    gs = base_cfg.get("grid_search", {})
    epochs = int(gs.get("epochs", 3))
    seed = int(gs.get("seed", base_cfg.get("train", {}).get("seed", 42)))

    hp = base_cfg.get("hparams", {})
    lrs = hp.get("lr", [5.0e-3, 1.0e-2, 2.0e-2])
    wds = hp.get("weight_decay", [1.0e-5, 1.0e-4])
    channels_list = hp.get("channels", [[32, 64, 128], [48, 96, 192]])
    kernels = hp.get("kernel_size", [3, 5])

    ts = time.strftime("%Y%m%d-%H%M%S")
    grid_root = os.path.join(base_cfg["paths"]["artifacts_dir"], "grid_search", ts)
    os.makedirs(grid_root, exist_ok=True)

    summary_csv = os.path.join(grid_root, "summary.csv")
    summary_md = os.path.join(grid_root, "summary.md")

    combos = list(product(lrs, wds, channels_list, kernels))
    print(f"Grid search: {len(combos)} runs | epochs={epochs} | seed={seed}")
    print("Artifacts:", grid_root)

    results = []

    for idx, (lr, wd, ch, k) in enumerate(combos, start=1):
        run_id = (
            f"run{idx:02d}_k{k}_ch{ch[0]}-{ch[1]}-{ch[2]}_"
            f"lr{_format_float(float(lr))}_wd{_format_float(float(wd))}"
        )
        run_art_dir = os.path.join(grid_root, run_id)
        os.makedirs(run_art_dir, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        cfg["train"]["lr"] = float(lr)
        cfg["train"]["weight_decay"] = float(wd)
        cfg["train"]["epochs"] = int(epochs)
        cfg["train"]["seed"] = int(seed)

        cfg["model"]["channels"] = [int(x) for x in ch]
        cfg["model"]["kernel_size"] = int(k)

        # each run gets its own artifacts dir to avoid overwriting best.ckpt
        cfg["paths"]["artifacts_dir"] = run_art_dir

        cfg_path = os.path.join(run_art_dir, "config.yaml")
        with open(cfg_path, "w") as f:
            yaml.safe_dump(cfg, f)

        print(f"\n[{idx:02d}/{len(combos)}] {run_id}")
        cmd = [
            sys.executable, "-m", "src.train",
            "--config", cfg_path,
            "--max_epochs", str(epochs),
            "--seed", str(seed),
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout = proc.stdout
        stderr = proc.stderr

        # keep logs for debugging
        with open(os.path.join(run_art_dir, "train_stdout.txt"), "w") as f:
            f.write(stdout)
        with open(os.path.join(run_art_dir, "train_stderr.txt"), "w") as f:
            f.write(stderr)

        best_acc = _parse_best_val_acc(stdout)
        if best_acc is None:
            print("WARNING: could not parse best val acc. Check train_stdout.txt")
            best_acc = float("nan")

        print("best_val_acc:", best_acc)

        results.append({
            "run_id": run_id,
            "lr": float(lr),
            "weight_decay": float(wd),
            "channels": str(ch),
            "kernel_size": int(k),
            "epochs": epochs,
            "seed": seed,
            "best_val_acc": best_acc,
            "artifacts_dir": run_art_dir,
        })

    # sort by best val acc
    results_sorted = sorted(results, key=lambda r: (r["best_val_acc"] if r["best_val_acc"] == r["best_val_acc"] else -1), reverse=True)

    # write CSV
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results_sorted[0].keys()))
        w.writeheader()
        for r in results_sorted:
            w.writerow(r)

    # write Markdown table
    with open(summary_md, "w") as f:
        f.write(f"# Grid search summary ({ts})\n\n")
        f.write("| rank | run_id | lr | wd | channels | k | best_val_acc |\n")
        f.write("|---:|---|---:|---:|---|---:|---:|\n")
        for i, r in enumerate(results_sorted, start=1):
            f.write(f"| {i} | {r['run_id']} | {r['lr']:.2e} | {r['weight_decay']:.2e} | {r['channels']} | {r['kernel_size']} | {r['best_val_acc']:.4f} |\n")

    best = results_sorted[0]
    print("\n=== BEST CONFIG (by val acc) ===")
    print(best)
    print("\nSummary files:")
    print(" -", summary_csv)
    print(" -", summary_md)


if __name__ == "__main__":
    main()
