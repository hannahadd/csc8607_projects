import torch
import torchaudio


def get_preprocess_transforms(config):
    p = config["preprocess"]
    target_sr = int(p["sample_rate"])
    n_mels = int(p["n_mels"])
    n_fft = int(p["n_fft"])
    win_length = int(p["win_length"])
    hop_length = int(p["hop_length"])
    to_db = bool(p.get("to_db", True))
    normalize = p.get("normalize", "per_feature")

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power") if to_db else None

    def _normalize(x: torch.Tensor) -> torch.Tensor:
        # x: (1, n_mels, T)
        if normalize in (None, "none"):
            return x
        if normalize == "per_sample":
            mean = x.mean()
            std = x.std().clamp_min(1e-6)
            return (x - mean) / std
        # per_feature: normalize each mel band over time
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
        return (x - mean) / std

    def preprocess(waveform: torch.Tensor, sr: int) -> torch.Tensor:
        # waveform: (C, N) or (N,)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        # mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # resample
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=target_sr)

        spec = mel(waveform)  # (1, n_mels, T)
        if amp_to_db is not None:
            spec = amp_to_db(spec)

        spec = _normalize(spec)
        return spec

    return preprocess
