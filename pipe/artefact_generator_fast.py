# artefact_generator_fast.py – optimierte, BigTIFF-fähige & resumable Artefakterzeugung

import os
import numpy as np
import rasterio
from libpysal.weights import lat2W
from esda import Moran_Local
from tqdm import tqdm
import datetime
from numba import njit, prange
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


# === 🪵 Logging ===
def log(msg):
    log_dir = os.path.join(cfg["data"]["base_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "artefact_run.log")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(log_path, "a") as f:
        f.write(line + "\n")


# === ⚡ Numba-beschleunigte lokale STD ===
@njit(parallel=True)
def local_std_numba(arr, size):
    h = size // 2
    out = np.full_like(arr, np.nan, dtype=np.float32)
    for i in prange(h, arr.shape[0] - h):
        for j in range(h, arr.shape[1] - h):
            window = arr[i - h:i + h + 1, j - h:j + h + 1].ravel()
            valid = window[~np.isnan(window)]
            if valid.size > 1:
                mean = valid.mean()
                out[i, j] = np.sqrt(((valid - mean) ** 2).sum() / (valid.size - 1))
    return out


# === 🔍 Welche Artefakte fehlen? ===
def artefacts_missing(index):
    base_path = os.path.join(cfg["data"]["base_dir"], cfg["data"]["raster_dirs"][index])
    all_rasters = [f for f in os.listdir(base_path) if f.endswith(".tif") and f"{index}_BerlinBB" in f]
    missing = []
    for raster in all_rasters:
        month_tag = "_".join(raster.split("_")[-2:]).replace(".tif", "")
        needed = [
            f"{index}_STD_{month_tag}.tif",
            f"{index}_MORAN_{month_tag}.tif",
            f"{index}_GEARY_{month_tag}.tif"
        ]
        if not all(os.path.exists(os.path.join(base_path, n)) for n in needed):
            missing.append(os.path.join(base_path, raster))
    return missing


# === 💾 Speicherintelligentes Raster-Writer ===
def save_raster(out_path, profile, data):
    meta = profile.copy()

    # Schätze Dateigröße (float32 = 4 Byte pro Pixel)
    est_gb = data.size * 4 / 1e9
    bigtiff = "YES" if est_gb > 3 else "IF_NEEDED"

    # Wertebereich auf [-1, 1] clippen, um float16 zu ermöglichen
    clipped = np.clip(data, -1, 1)
    dtype = "float16" if np.nanmax(np.abs(clipped)) <= 1 else "float32"
    compressed = clipped.astype(dtype)

    meta.update(
        dtype=dtype,
        count=1,
        compress="lzw",
        BIGTIFF=bigtiff
    )

    log(f"  💾 Speichere Raster: {os.path.basename(out_path)} "
        f"({dtype}, ~{est_gb:.2f} GB, BigTIFF={bigtiff})")

    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(compressed, 1)


# === 🧮 Hauptfunktion ===
def generate_environmental_artefacts_fast(block_size=1024, std_size=11, downsample=5):
    indices = ["NDVI", "NDWI"]
    for index in indices:
        dir_path = os.path.join(cfg["data"]["base_dir"], cfg["data"]["raster_dirs"][index])
        os.makedirs(dir_path, exist_ok=True)

        missing = artefacts_missing(index)
        log(f"📂 {index}: {len(missing)} Raster ohne vollständige Artefakte")

        for raster_path in tqdm(missing, desc=f"{index}-Analyse"):
            base = os.path.basename(raster_path)
            month_tag = "_".join(base.split("_")[-2:]).replace(".tif", "")
            log(f"\n🧮 Verarbeite {index} → {month_tag}")

            # === Raster laden ===
            with rasterio.open(raster_path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            # === Lokale STD berechnen (blockweise) ===
            log("  • Berechne lokale STD (Numba + Blockweise)...")
            h, w = arr.shape
            std_map = np.full_like(arr, np.nan, dtype=np.float32)

            for i in range(0, h, block_size):
                for j in range(0, w, block_size):
                    block = arr[i:i + block_size, j:j + block_size]
                    std_map[i:i + block.shape[0], j:j + block.shape[1]] = local_std_numba(block, size=std_size)

            save_raster(os.path.join(dir_path, f"{index}_STD_{month_tag}.tif"), prof, std_map)
            log(f"  ✅ STD gespeichert: {month_tag}")

            # === Moran & Geary (downsampled) ===
            log("  • Berechne Moran & Geary (Downsample)...")
            sub_arr = arr[::downsample, ::downsample]
            sub_arr[np.isnan(sub_arr)] = 0
            w_obj = lat2W(*sub_arr.shape)
            w_obj.transform = "r"

            # Moran Local
            moran = Moran_Local(sub_arr.ravel(), w_obj)
            moran_map = np.repeat(np.repeat(moran.Is.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)
            moran_map = moran_map[:arr.shape[0], :arr.shape[1]]

            # Geary Fallback
            try:
                from esda import Geary_Local
                geary = Geary_Local(sub_arr.ravel(), w_obj)
                geary_values = getattr(geary, "geary_local", getattr(geary, "Cs", None))
                if geary_values is None:
                    raise AttributeError
                geary_map = np.repeat(np.repeat(geary_values.reshape(sub_arr.shape), downsample, axis=0), downsample, axis=1)
                log("  ✅ Geary_Local (standard) verwendet.")
            except Exception:
                from scipy.ndimage import uniform_filter
                log("  ⚠️ Fallback: manuelle Geary-Berechnung ...")
                mean = uniform_filter(sub_arr, size=3)
                diff = (sub_arr - mean) ** 2
                geary_map = diff / np.nanmean(diff)
                geary_map = np.repeat(np.repeat(geary_map, downsample, axis=0), downsample, axis=1)

            save_raster(os.path.join(dir_path, f"{index}_MORAN_{month_tag}.tif"), prof, moran_map)
            save_raster(os.path.join(dir_path, f"{index}_GEARY_{month_tag}.tif"), prof, geary_map)
            log(f"  ✅ MORAN & GEARY gespeichert: {month_tag}")

    log("\n🏁 Fertig! Alle Artefakte berechnet.")