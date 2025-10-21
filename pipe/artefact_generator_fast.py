# artefact_generator_fast.py – High-Performance Umweltartefakt-Generator
# Version 2025-10 – kombiniert schnelle STD (Numba) + Moran/Geary + Fallbacks

import os
import time
import datetime
import numpy as np
import rasterio
from tqdm import tqdm
from libpysal.weights import lat2W
from esda import Moran_Local, Geary_Local
from scipy.ndimage import generic_filter
from numba import njit, prange
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg


# === 🧠 Performance STD mit Numba (blockweise) ===
@njit(parallel=True, fastmath=True)
def _local_std_core(arr, size=11):
    pad = size // 2
    h, w = arr.shape
    out = np.empty((h, w), dtype=np.float32)

    for i in prange(h):
        for j in range(w):
            i_min = max(0, i - pad)
            i_max = min(h, i + pad + 1)
            j_min = max(0, j - pad)
            j_max = min(w, j + pad + 1)

            mean = 0.0
            count = 0.0
            for x in range(i_min, i_max):
                for y in range(j_min, j_max):
                    val = arr[x, y]
                    if not np.isnan(val):
                        mean += val
                        count += 1
            if count == 0:
                out[i, j] = np.nan
                continue

            mean /= count
            var = 0.0
            for x in range(i_min, i_max):
                for y in range(j_min, j_max):
                    val = arr[x, y]
                    if not np.isnan(val):
                        diff = val - mean
                        var += diff * diff
            out[i, j] = np.sqrt(var / count)
    return out


def local_std_numba_blockwise(arr, size=11, block_size=1024):
    """Blockweise lokale STD mit Numba-Kern."""
    h, w = arr.shape
    out = np.full((h, w), np.nan, dtype=np.float32)
    pad = size // 2

    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(h, y + block_size)
            x_end = min(w, x + block_size)
            y0 = max(0, y - pad)
            y1 = min(h, y_end + pad)
            x0 = max(0, x - pad)
            x1 = min(w, x_end + pad)
            sub = arr[y0:y1, x0:x1]
            sub_std = _local_std_core(sub, size=size)

            sy0 = pad if y > 0 else 0
            sy1 = sub_std.shape[0] - pad if y_end < h else sub_std.shape[0]
            sx0 = pad if x > 0 else 0
            sx1 = sub_std.shape[1] - pad if x_end < w else sub_std.shape[1]
            out[y:y_end, x:x_end] = sub_std[sy0:sy1, sx0:sx1]
    return out


# === 🧩 Hilfsfunktionen ===
def save_raster(out_path, profile, data):
    """Speichert GeoTIFF robust (float32, BigTIFF bei Bedarf)."""
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw", BIGTIFF="IF_NEEDED")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)


def safe_geary(values, w):
    """Robuster Geary_Local mit automatischem Fallback."""
    try:
        g = Geary_Local(values, w)
        for attr in ["Cs", "geary_local", "G_local", "statistic_"]:
            if hasattr(g, attr):
                return np.array(getattr(g, attr)).ravel()
    except Exception:
        pass

    # Fallback manuell
    y = values.copy()
    mean_y = np.mean(y)
    numerator = np.zeros_like(y)
    for i, neigh in enumerate(w.neighbors.values()):
        numerator[i] = np.sum((y[i] - y[neigh]) ** 2)
    denominator = 2 * len(y) * np.sum((y - mean_y) ** 2)
    return (len(y) - 1) * numerator / denominator


# === 🧮 Hauptfunktion ===
def generate_environmental_artefacts_fast(block_size=1024, std_size=11, downsample=5):
    """
    Berechnet STD, Moran und Geary (schnell & robust) für alle Rasterdateien.
    """
    dirs = cfg["data"]["raster_dirs"]
    start_time = datetime.datetime.now()

    for index, index_dir in dirs.items():
        base_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(base_path, exist_ok=True)

        raster_files = sorted([
            os.path.join(base_path, f) for f in os.listdir(base_path)
            if f.endswith(".tif") and index in f and "_STD_" not in f
        ])
        print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] 📂 {index}: {len(raster_files)} Raster ohne vollständige Artefakte")

        for path in tqdm(raster_files, desc=f"{index}-Analyse"):
            base = os.path.basename(path)
            month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
            print(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] 🧮 Verarbeite {index} → {month}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            # === STD ===
            std_path = os.path.join(base_path, f"{index}_STD_{month}.tif")
            if not os.path.exists(std_path):
                print(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] ▶️ Starte STD")
                std_map = local_std_numba_blockwise(arr, size=std_size, block_size=block_size)
                save_raster(std_path, prof, std_map)
                print(f"     ✅ STD gespeichert: {os.path.basename(std_path)}")
            else:
                print(f"     ⏩ STD übersprungen (existiert).")

            # === Moran & Geary ===
            moran_path = os.path.join(base_path, f"{index}_MORAN_{month}.tif")
            geary_path = os.path.join(base_path, f"{index}_GEARY_{month}.tif")

            if os.path.exists(moran_path) and os.path.exists(geary_path):
                print("     ⏩ Moran/Geary übersprungen (existieren).")
                continue

            print("  • Berechne Moran & Geary …")
            sub_arr = arr[::downsample, ::downsample].astype("float64")
            sub_arr[np.isnan(sub_arr)] = 0

            w = lat2W(*sub_arr.shape)
            w.transform = "r"

            try:
                moran = Moran_Local(sub_arr.ravel(), w)
                moran_vals = moran.Is
                print("     ✅ Moran_Local berechnet.")
            except Exception as e:
                print(f"     ⚠️ Moran_Local Fehler: {e}")
                continue

            geary_vals = safe_geary(sub_arr.ravel(), w)
            print("     ✅ Geary_Local berechnet.")

            # Upscale zurück auf Originalgröße
            moran_map = np.repeat(np.repeat(moran_vals.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)
            geary_map = np.repeat(np.repeat(geary_vals.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)
            moran_map = moran_map[:arr.shape[0], :arr.shape[1]]
            geary_map = geary_map[:arr.shape[0], :arr.shape[1]]

            save_raster(moran_path, prof, moran_map)
            save_raster(geary_path, prof, geary_map)
            print("     ✅ Moran & Geary gespeichert.")

    print(f"\n🏁 Fertig! Laufzeit: {datetime.datetime.now() - start_time}")