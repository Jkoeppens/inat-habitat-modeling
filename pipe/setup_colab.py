# /pipe/setup_colab.py

import sys
import os

def setup_repo(base_path="/content/inaturalist"):
    if base_path not in sys.path:
        sys.path.append(base_path)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"📁 Repo-Verzeichnis nicht gefunden: {base_path}")

    print(f"✅ Pfad gesetzt: {base_path}")