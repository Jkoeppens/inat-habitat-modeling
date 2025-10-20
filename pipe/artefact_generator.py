# artefact_generator_refactored.py – robuste NDVI/NDWI-Artefakt-Berechnung
# Version: 2025-10 – Colab-kompatibel mit Geary-Fallback & Stichprobenmodus

import os
import time
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local, Geary_Local
from tqdm import tqdm
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg


# === Hilfsfunktionen ===

def local_std(arr, size=11):
    """Lokale Standardabweichung mit NaN-handling."""
    return generic_filter(arr, np.nanstd, size=size, mode='nearest')


def save_raster(out_path, profile, data):
    """Speichert eine GeoTIFF-Datei im float32-Format."""
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)


def safe_geary(values, w):
    """
    Robuster Wrapper für Geary_Local – nutzt Fallback, wenn moderne API fehlt.
    """
    try:
        # Versuch: moderne sklearn-API
        model = Geary_Local()
        if hasattr(model, "fit_transform"):
            res = model.fit_transform(values.reshape(-1, 1))
            return np.array(res).ravel()
    except Exception:
        pass

    try:
        # Versuch: klassische API
        g = Geary_Local(values, w)
        for attr in ["Cs", "geary_local", "localG", "G_local", "stat", "statistic"]:
            if hasattr(g, attr):
                print(f"     ↳ Geary_Local ({attr}) verwendet.")
                return np.array(getattr(g, attr)).ravel()
    except Exception:
        pass

    # Fallback: manuelle Berechnung
    print("     ⚙️ Geary-Fallback aktiv (manuelle Berechnung).")
    y = values.copy()
    mean_y = np.mean(y)
    numerator = np.zeros_like(y)
    for i, neighbors in enumerate(w.neighbors.values()):
        numerator[i] = np.sum((y[i] - y[neighbors])**2)
    denominator = 2 * len(y) * np.sum((y - mean_y)**2)
    return (len(y) - 1) * numerator / denominator


# === Hauptfunktion ===

def generate_environmental_artefacts(sample=False, sample_size=500, downsample=5):
    """
    Berechnet lokale Umweltartefakte (STD, Moran, Geary)
    für alle Raster im NDVI/NDWI-Verzeichnis.

    Args:
        sample (bool): Wenn True, nur mittleren Ausschnitt verarbeiten.
        sample_size (int): Größe der Stichprobe in Pixeln.
        downsample (int): Reduktionsfaktor für Moran/Geary-Berechnung.
    """
    dirs = cfg["data"]["raster_dirs"]

    for index, index_dir in dirs.items():
        full_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(full_path, exist_ok=True)

        raster_files = sorted([
            os.path.join(full_path, f) for f in os.listdir(full_path)
            if f.endswith(".tif") and index in f and "_STD_" not in f and "_MORAN_" not in f and "_GEARY_" not in f
        ])

        print(f"\n📂 {index}: {len(raster_files)} Raster gefunden")
        for f in raster_files[:3]:
            print(f"   - {os.path.basename(f)}")

        for path in tqdm(raster_files, desc=f"{index}-Analyse"):
            t0 = time.time()
            base = os.path.basename(path)
            month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
            print(f"\n🧮 Verarbeite {index} → {month}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            if sample:
                mid_y, mid_x = arr.shape[0] // 2, arr.shape[1] // 2
                arr = arr[mid_y - sample_size//2 : mid_y + sample_size//2,
                          mid_x - sample_size//2 : mid_x + sample_size//2]
                print(f"     🔎 Stichprobe: {arr.shape}")

            # --- STD ---
            print("  • Berechne lokale STD …")
            std_map = local_std(arr, size=11)
            out_std = os.path.join(full_path, f"{index}_STD_{month}{'_sample' if sample else ''}.tif")
            save_raster(out_std, prof, std_map)
            print(f"     ✅ STD gespeichert: {os.path.basename(out_std)}")

            # --- Moran & Geary ---
            print("  • Berechne Moran & Geary …")
            sub_arr = arr[::downsample, ::downsample].astype("float64")
            sub_arr[np.isnan(sub_arr)] = 0

            w = lat2W(*sub_arr.shape)
            w.transform = "r"

            moran = Moran_Local(sub_arr.ravel(), w)
            geary_vals = safe_geary(sub_arr.ravel(), w)

            moran_map = np.repeat(np.repeat(moran.Is.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)
            geary_map = np.repeat(np.repeat(geary_vals.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)

            moran_map = moran_map[:arr.shape[0], :arr.shape[1]]
            geary_map = geary_map[:arr.shape[0], :arr.shape[1]]

            out_moran = os.path.join(full_path, f"{index}_MORAN_{month}{'_sample' if sample else ''}.tif")
            out_geary = os.path.join(full_path, f"{index}_GEARY_{month}{'_sample' if sample else ''}.tif")

            save_raster(out_moran, prof, moran_map)
            save_raster(out_geary, prof, geary_map)

            print(f"     ✅ MORAN & GEARY gespeichert.")
            print(f"     ⏱️ Dauer: {time.time()-t0:.1f}s")

    print("\n🏁 Fertig! Alle Artefakte berechnet.")
