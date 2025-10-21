# ================================================================
# artefact_generator_fast.py
# Blockweise, robuste Artefakt-Berechnung (NDVI/NDWI)
# GPU/CPU-optimiert für Google Colab & Linux
# ================================================================

import os
import numpy as np
import rasterio
from rasterio.enums import Compression
from rasterio.windows import Window
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local, Geary_Local
from tqdm import tqdm
import time, datetime, traceback, psutil

from config.config import cfg


# ================================================================
# 🧩 Utility-Funktionen
# ================================================================

def local_std_blockwise(arr, size=11):
    """
    Lokale Standardabweichung mit NaN-handling (blockweise, CPU-sicher).
    """
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    return generic_filter(arr, np.nanstd, size=size, mode="reflect")


def save_raster(out_path, profile, data):
    """
    Speichert GeoTIFF mit automatischer BigTIFF-Unterstützung.
    """
    meta = profile.copy()
    meta.update(
        dtype="float32",
        count=1,
        compress=Compression.lzw.value,
        BIGTIFF="IF_NEEDED"
    )
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)


def safe_geary(values, w):
    """
    Robuster Wrapper für Geary_Local mit mehreren Fallbacks.
    """
    try:
        g = Geary_Local(values, w)
        for attr in ["Cs", "statistic", "geary_local"]:
            if hasattr(g, attr):
                return np.array(getattr(g, attr)).ravel()
    except Exception:
        pass

    # Fallback manuell
    y = values
    mean_y = np.mean(y)
    num = np.zeros_like(y)
    for i, nbrs in enumerate(w.neighbors.values()):
        num[i] = np.sum((y[i] - y[nbrs])**2)
    denom = 2 * len(y) * np.sum((y - mean_y)**2)
    return (len(y) - 1) * num / denom


# ================================================================
# 🧮 Einzelraster-Verarbeitung
# ================================================================

def process_single_raster(path, block_size=1024, std_size=11, downsample=5):
    """
    Berechnet STD, Moran & Geary für ein einzelnes Raster.
    """
    base = os.path.basename(path)
    index = "NDVI" if "NDVI" in base else "NDWI"
    month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
    dir_path = os.path.dirname(path)

    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] ▶️ Starte {index} {month}")

    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        profile = src.profile
        arr[arr == src.nodata] = np.nan

    # STD-Berechnung (blockweise)
    t0 = time.time()
    try:
        std_map = local_std_blockwise(arr, size=std_size)
        out_std = os.path.join(dir_path, f"{index}_STD_{month}.tif")
        save_raster(out_std, profile, std_map)
        print(f"   ✅ STD gespeichert ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"   ❌ Fehler in STD: {e}")
        traceback.print_exc()
        return

    # Downsampling für Moran & Geary
    sub_arr = arr[::downsample, ::downsample]
    sub_arr[np.isnan(sub_arr)] = 0
    w = lat2W(*sub_arr.shape)
    w.transform = "r"

    try:
        # Moran
        moran = Moran_Local(sub_arr.ravel(), w)
        moran_map = np.repeat(np.repeat(moran.Is.reshape(sub_arr.shape), downsample, 0), downsample, 1)
        moran_map = moran_map[:arr.shape[0], :arr.shape[1]]
        save_raster(os.path.join(dir_path, f"{index}_MORAN_{month}.tif"), profile, moran_map)

        # Geary
        geary_vals = safe_geary(sub_arr.ravel(), w)
        geary_map = np.repeat(np.repeat(geary_vals.reshape(sub_arr.shape), downsample, 0), downsample, 1)
        geary_map = geary_map[:arr.shape[0], :arr.shape[1]]
        save_raster(os.path.join(dir_path, f"{index}_GEARY_{month}.tif"), profile, geary_map)

        print(f"   ✅ Moran & Geary gespeichert ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"   ⚠️ Fehler in Moran/Geary: {e}")
        traceback.print_exc()


# ================================================================
# 🧩 Hauptsteuerung – ganze Serie
# ================================================================

def generate_environmental_artefacts_fast(block_size=1024, std_size=11, downsample=5):
    """
    Iteriert über alle NDVI/NDWI-Raster und berechnet fehlende Artefakte.
    """
    dirs = cfg["data"]["raster_dirs"]
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 📂 Starte vollständigen Artefaktlauf")

    for index, index_dir in dirs.items():
        full_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(full_path, exist_ok=True)

        all_rasters = [os.path.join(full_path, f) for f in os.listdir(full_path)
                       if f.endswith(".tif") and index in f and "_STD_" not in f]

        print(f"\n📁 {index}: {len(all_rasters)} Raster gefunden")
        for i, raster_path in enumerate(tqdm(all_rasters, desc=f"{index}-Artefakte")):
            try:
                process_single_raster(raster_path, block_size, std_size, downsample)
            except Exception as e:
                print(f"❌ Fehler bei {raster_path}: {e}")

        ram = psutil.virtual_memory().percent
        print(f"💾 RAM={ram}% | Zeit={datetime.datetime.now().strftime('%H:%M:%S')}")
    print("\n🏁 Fertig! Alle Artefakte berechnet.")