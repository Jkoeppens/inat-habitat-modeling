# artefact_generator_fast.py – stabile Version (2025-10)
# Robust gegen Drive-I/O, speichert zuerst lokal, dann Kopie zu Drive.
# Unterstützt Resume, KeepAlive & progress logging.

import os
import shutil
import time
import datetime
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local
from tqdm import tqdm
import psutil, traceback, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


# === 🧠 Hilfsfunktionen ===

def local_std(arr, size=11):
    """Blockweise lokale Standardabweichung."""
    return generic_filter(arr, np.nanstd, size=size, mode="nearest")

def safe_save_to_drive(out_path, prof, data):
    """Speichert Raster lokal in /content/tmp, dann Kopie auf Drive."""
    os.makedirs("/content/tmp", exist_ok=True)
    tmp_path = f"/content/tmp/{os.path.basename(out_path)}"

    meta = prof.copy()
    meta.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(tmp_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)

    try:
        shutil.copy(tmp_path, out_path)
        print(f"  💾 Gespeichert → {out_path}")
    except Exception as e:
        print(f"⚠️ Fehler beim Kopieren auf Drive: {e}")
        print("  Datei bleibt lokal in /content/tmp erhalten.")


def compute_moran(arr, w):
    """Berechnet lokalen Moran."""
    try:
        moran = Moran_Local(arr.ravel(), w)
        return moran.Is
    except Exception as e:
        print(f"⚠️ Moran fehlgeschlagen: {e}")
        return np.zeros_like(arr.ravel())


# === 🚀 Hauptfunktion ===

def generate_environmental_artefacts_fast(
    std_size=11, downsample=5, resume=True, keepalive=True
):
    """
    Berechnet robuste STD- und Moran-Artefakte (Geary optional)
    und speichert sicher auf Google Drive.

    Args:
        std_size (int): Fenstergröße für STD
        downsample (int): Downsampling für Moran
        resume (bool): Überspringe existierende Artefakte
        keepalive (bool): Ausgabe alle 5 Min
    """

    dirs = cfg["data"]["raster_dirs"]

    print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] 📊 Starte Artefaktlauf")
    os.makedirs("/content/tmp", exist_ok=True)

    for index, index_dir in dirs.items():
        full_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(full_path, exist_ok=True)

        files = sorted([
            os.path.join(full_path, f) for f in os.listdir(full_path)
            if f.endswith(".tif") and index in f and "_STD_" not in f and "_MORAN_" not in f
        ])
        print(f"\n📂 {index}: {len(files)} Raster ohne Artefakte")

        for i, path in enumerate(files, 1):
            base = os.path.basename(path)
            month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
            print(f"\n🧮 [{i}/{len(files)}] {base} @ {datetime.datetime.now():%H:%M:%S}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            out_std = os.path.join(full_path, f"{index}_STD_{month}.tif")
            out_moran = os.path.join(full_path, f"{index}_MORAN_{month}.tif")

            # === Resume: überspringen falls bereits vorhanden
            if resume and os.path.exists(out_std) and os.path.exists(out_moran):
                print("  ⏭️ Artefakte existieren, überspringe.")
                continue

            # === Lokale STD
            t0 = time.time()
            print("  ▶️ Berechne lokale STD ...")
            std_map = local_std(arr, size=std_size)
            safe_save_to_drive(out_std, prof, std_map)
            print(f"  ✅ STD fertig in {time.time()-t0:.1f}s")

            # === Moran
            print("  ▶️ Berechne lokalen Moran ...")
            sub = arr[::downsample, ::downsample].astype("float64")
            sub[np.isnan(sub)] = 0
            w = lat2W(*sub.shape)
            w.transform = "r"
            moran_vals = compute_moran(sub, w)
            moran_map = np.repeat(
                np.repeat(moran_vals.reshape(sub.shape), downsample, axis=0),
                downsample, axis=1
            )[:arr.shape[0], :arr.shape[1]]
            safe_save_to_drive(out_moran, prof, moran_map)
            print(f"  ✅ MORAN fertig in {time.time()-t0:.1f}s")

            # === KeepAlive
            if keepalive and (i % 1 == 0):
                ram = psutil.virtual_memory().percent
                print(f"  💾 RAM={ram}% | Zeit={datetime.datetime.now():%H:%M:%S}")

    print(f"\n🏁 Fertig @ {datetime.datetime.now():%H:%M:%S}")