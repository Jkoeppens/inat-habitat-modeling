# setup_colab.py – einmal pro Session ausführen
import sys
import os
import ee

def setup_colab(repo_url="https://github.com/Jkoeppens/inat-habitat-modeling.git", subdir="pipe"):
    # Installiere notwendige Pakete
    !pip install -q rasterio geopandas shapely

    # Earth Engine Auth
    try:
        ee.Initialize()
        print("✅ Earth Engine ist bereits initialisiert")
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize()
        print("🔑 Authentifizierung abgeschlossen")

    # Git sparse clone für /pipe
    os.system("rm -rf pipe temp_repo")
    os.makedirs("pipe", exist_ok=True)
    os.system(f"git clone --depth 1 --filter=blob:none --sparse {repo_url} temp_repo")
    os.chdir("temp_repo")
    os.system("git sparse-checkout init --cone")
    os.system(f"git sparse-checkout set {subdir}")
    os.system(f"cp -r {subdir} ../{subdir}")
    os.chdir("..")
    os.system("rm -rf temp_repo")

    # Füge Colab-Pfad hinzu
    if '/content' not in sys.path:
        sys.path.append('/content')

    # Google Drive mounten
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)

    print("✅ Setup abgeschlossen: pipe/ geladen, Earth Engine aktiv, Drive gemountet")
