# config.py – Lädt default.yaml + local.yaml und macht Pfade zugänglich

import yaml
import os
from pathlib import Path
import re


# 🔁 Rekursives Dictionary-Merge
def deep_merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and k in a:
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


# 🔗 Platzhalter wie ${paths.base_data_dir} ersetzen
_placeholder_pattern = re.compile(r"\$\{([^}]+)\}")

def resolve_placeholders(config):
    def _resolve(value):
        if isinstance(value, str):
            for match in _placeholder_pattern.findall(value):
                parts = match.split('.')
                ref = config
                for part in parts:
                    ref = ref.get(part, f"$UNRESOLVED:{match}")
                value = value.replace(f"${{{match}}}", str(ref))
        elif isinstance(value, dict):
            return {k: _resolve(v) for k, v in value.items()}
        return value
    return _resolve(config)


# 📦 Konfiguration laden

def load_config(
    default_path="/content/inatrualist/config/default.yaml",
    local_path="/content/drive/MyDrive/iNaturalist/local.yaml"
):
    with open(default_path, "r") as f:
        config = yaml.safe_load(f)

    local_file = Path(local_path)
    if local_file.exists():
        with open(local_file, "r") as f:
            local = yaml.safe_load(f)
        config = deep_merge(config, local)

    config = resolve_placeholders(config)
    return config


# ✅ Direkt beim Import laden
CONFIG = load_config()

# Beispielzugriff:
# base_dir = CONFIG["paths"]["base_data_dir"]