import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

from .preprocessing import get_preprocess_transforms
from .augmentation import get_augmentation_transforms

import numpy as np
import soundfile as sf


@dataclass
class ESC50Item:
    filepath: str
    label: int

def load_wav(path: str):
    """
    Robust wav loader (works even if torchaudio backend requires torchcodec).
    Returns: waveform (C, N) float32 torch tensor, sample_rate int
    """
    data, sr = sf.read(path, dtype="float32", always_2d=True)  # (N, C)
    waveform = torch.from_numpy(data.T)  # (C, N)
    return waveform, sr


class ESC50Dataset(Dataset):
    def __init__(self, items: List[ESC50Item], preprocess_fn, augment_fn=None):
        self.items = items
        self.preprocess_fn = preprocess_fn
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        wav, sr = load_wav(it.filepath)
        x = self.preprocess_fn(wav, sr)         # (1, 64, T)
        if self.augment_fn is not None:
            x = self.augment_fn(x)
        y = torch.tensor(it.label, dtype=torch.long)
        return x, y


def _build_items(df: pd.DataFrame, audio_dir: str, label_map: Dict[str, int]) -> List[ESC50Item]:
    items = []
    for _, row in df.iterrows():
        path = os.path.join(audio_dir, row["filename"])
        items.append(ESC50Item(filepath=path, label=label_map[row["category"]]))
    return items


def get_dataloaders(config) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    dcfg = config["dataset"]
    csv_path = dcfg["annotation_csv"]
    audio_dir = dcfg["audio_dir"]

    split = dcfg["split"]
    train_folds = set(split["train_folds"])
    val_folds = set(split["val_folds"])
    test_folds = set(split["test_folds"])

    df = pd.read_csv(csv_path)

    # stable label mapping
    categories = sorted(df["category"].unique().tolist())
    label_map = {c: i for i, c in enumerate(categories)}

    df_train = df[df["fold"].isin(train_folds)].reset_index(drop=True)
    df_val = df[df["fold"].isin(val_folds)].reset_index(drop=True)
    df_test = df[df["fold"].isin(test_folds)].reset_index(drop=True)

    preprocess_fn = get_preprocess_transforms(config)
    augment_fn = get_augmentation_transforms(config)

    train_ds = ESC50Dataset(_build_items(df_train, audio_dir, label_map), preprocess_fn, augment_fn=augment_fn)
    val_ds = ESC50Dataset(_build_items(df_val, audio_dir, label_map), preprocess_fn, augment_fn=None)
    test_ds = ESC50Dataset(_build_items(df_test, audio_dir, label_map), preprocess_fn, augment_fn=None)

    tcfg = config["train"]
    bs = int(tcfg["batch_size"])
    nw = int(tcfg.get("num_workers", 0))

    pin = torch.cuda.is_available()  # True uniquement si GPU CUDA (cluster)

    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=True)

    # meta: input_shape computed from dummy 5 sec signal
    sr = int(config["preprocess"]["sample_rate"])
    dummy = torch.zeros(1, sr * 5)
    x0 = preprocess_fn(dummy, sr)  # (1, 64, T)

    meta = {
        "num_classes": len(categories),
        "input_shape": tuple(x0.shape),
        "label_map": label_map,
        "splits": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }
    return train_loader, val_loader, test_loader, meta
