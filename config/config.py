import os
import yaml
from pathlib import Path

# === 🧩 Hilfsfunktionen ===

def deep_merge(d1, d2):
    """Rekursives Zusammenführen von zwei Dictionaries."""
    for k, v in d2.items():
        if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
            deep_merge(d1[k], v)
        else:
            d1[k] = v
    return d1


def resolve_placeholders(d, root=None):
    """Ersetzt rekursiv ${...}-Platzhalter innerhalb der YAML-Struktur."""
    if root is None:
        root = d

    if isinstance(d, dict):
        return {k: resolve_placeholders(v, root) for k, v in d.items()}
    elif isinstance(d, list):
        return [resolve_placeholders(i, root) for i in d]
    elif isinstance(d, str) and "${" in d:
        # Suche rekursiv nach Werten im Root-Dict
        for key_path in _find_all_keys(root):
            placeholder = "${" + key_path + "}"
            if placeholder in d:
                val = _get_value_by_path(root, key_path)
                if val is not None:
                    d = d.replace(placeholder, str(val))
        return d
    else:
        return d


def _find_all_keys(d, prefix=""):
    """Hilfsfunktion: findet alle möglichen Key-Pfade (für ${...})."""
    keys = []
    if isinstance(d, dict):
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            keys.append(full_key)
            keys.extend(_find_all_keys(v, full_key))
    return keys


def _get_value_by_path(d, path):
    """Hilfsfunktion: Zugriff auf verschachtelten Key durch Pfad (z. B. paths.base_data_dir)."""
    parts = path.split(".")
    cur = d
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        else:
            return None
    return cur


# === 🔧 Hauptfunktion ===

def load_config(
    default_path="/content/inaturalist/config/default.yaml",
    local_path="/content/drive/MyDrive/iNaturalist/local.yaml"
):
    """Lädt und kombiniert default.yaml + local.yaml mit rekursivem Merge + Platzhalterauflösung."""
    default_file = Path(default_path)
    if not default_file.exists():
        raise FileNotFoundError(f"❌ default.yaml nicht gefunden unter: {default_path}")

    with open(default_file, "r") as f:
        config = yaml.safe_load(f) or {}

    local_file = Path(local_path)
    if local_file.exists():
        with open(local_file, "r") as f:
            local = yaml.safe_load(f)
        if isinstance(local, dict):
            config = deep_merge(config, local)

    # Platzhalter (z. B. ${paths.base_data_dir}) rekursiv auflösen
    config = resolve_placeholders(config)
    return config


# === 🚀 Automatisches Laden ===
cfg = load_config()

if __name__ == "__main__":
    print("✅ config.py erfolgreich geladen.")
    print(f"📁 Basisdaten: {cfg['paths']['base_data_dir']}")
    print(f"📦 NDVI: {cfg['paths']['ndvi_dir']}")
    print(f"📦 Output: {cfg['paths']['output_dir']}")