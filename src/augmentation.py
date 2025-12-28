import random
import torch
import torchaudio


def get_augmentation_transforms(config):
    a = config.get("augment", {})
    enabled = bool(a.get("enabled", False))

    freq_cfg = a.get("freq_mask", {})
    time_cfg = a.get("time_mask", {})
    shift_cfg = a.get("time_shift", {})

    freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=int(freq_cfg.get("max_width", 8)))
    time_mask = torchaudio.transforms.TimeMasking(time_mask_param=int(time_cfg.get("max_width", 20)))

    def augment(spec: torch.Tensor) -> torch.Tensor:
        # spec: (1, n_mels, T)
        if not enabled:
            return spec

        x = spec

        # time shift (roll)
        if bool(shift_cfg.get("enabled", True)) and random.random() < float(shift_cfg.get("p", 0.3)):
            max_pct = float(shift_cfg.get("max_shift_pct", 0.05))
            T = x.size(-1)
            max_shift = int(T * max_pct)
            if max_shift > 0:
                k = random.randint(-max_shift, max_shift)
                x = torch.roll(x, shifts=k, dims=-1)

        # freq masking
        if bool(freq_cfg.get("enabled", True)) and random.random() < float(freq_cfg.get("p", 0.5)):
            for _ in range(int(freq_cfg.get("num_masks", 1))):
                x = freq_mask(x)

        # time masking
        if bool(time_cfg.get("enabled", True)) and random.random() < float(time_cfg.get("p", 0.5)):
            for _ in range(int(time_cfg.get("num_masks", 1))):
                x = time_mask(x)

        return x

    return augment
