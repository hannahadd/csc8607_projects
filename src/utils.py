"""
Utils génériques.

Fonctions attendues (signatures imposées) :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

def set_seed(seed: int) -> None:
    """Initialise les seeds (numpy/torch/python). À implémenter."""
    raise NotImplementedError("set_seed doit être implémentée par l'étudiant·e.")


def get_device(prefer: str | None = "auto") -> str:
    """Retourne 'cpu' ou 'cuda' (ou choix basé sur 'auto'). À implémenter."""
    raise NotImplementedError("get_device doit être implémentée par l'étudiant·e.")


def count_parameters(model) -> int:
    """Retourne le nombre de paramètres entraînables du modèle. À implémenter."""
    raise NotImplementedError("count_parameters doit être implémentée par l'étudiant·e.")


def save_config_snapshot(config: dict, out_dir: str) -> None:
    """Sauvegarde une copie de la config (ex: YAML) dans out_dir. À implémenter."""
    raise NotImplementedError("save_config_snapshot doit être implémentée par l'étudiant·e.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
