import os
import yaml
from pathlib import Path


def deep_merge(d1, d2):
    for k, v in d2.items():
        if (
            k in d1 and isinstance(d1[k], dict) and isinstance(v, dict)
        ):
            deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1


def resolve_placeholders(d):
    if isinstance(d, dict):
        return {k: resolve_placeholders(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [resolve_placeholders(i) for i in d]
    elif isinstance(d, str):
        for key, val in os.environ.items():
            d = d.replace(f"${{{key}}}", val)
        return d
    return d


def load_config(
    default_path="/content/inaturalist/config/default.yaml",
    local_path="/content/drive/MyDrive/iNaturalist/local.yaml"
):
    default_file = Path(default_path)
    if not default_file.exists():
        raise FileNotFoundError(f"❌ default.yaml nicht gefunden unter: {default_path}")

    with open(default_file, "r") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ValueError("❌ Fehler beim Laden von default.yaml – Inhalt ist kein Dictionary")

    local_file = Path(local_path)
    if local_file.exists():
        with open(local_file, "r") as f:
            local = yaml.safe_load(f)
        if isinstance(local, dict):
            config = deep_merge(config, local)

    config = resolve_placeholders(config)
    return config


# automatisch laden
cfg = load_config()