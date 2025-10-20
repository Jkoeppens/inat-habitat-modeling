# setup_colab.py

import os
import sys

from google.colab import drive
from config.config import load_config

def setup_colab():
    # 📂 Google Drive einbinden
    drive.mount('/content/drive')

    # 🛠️ Projektpfad setzen (wenn Repo geklont ist)
    repo_path = "/content/inaturalist"
    if repo_path not in sys.path:
        sys.path.append(repo_path)

    # 📜 Konfiguration laden
    cfg = load_config("/content/drive/MyDrive/iNaturalist/local.yaml")

    # 🔍 Übersicht ausgeben
    print("\n✅ Konfiguration geladen.")
    print(f"\n📁 Projektname: {cfg['project']['name']}")
    print(f"📂 Basisverzeichnis: {cfg['data']['base_dir']}")
    print(f"📦 NDVI-Verzeichnis: {cfg['data']['raster_dirs']['NDVI']}")

    return cfg
    