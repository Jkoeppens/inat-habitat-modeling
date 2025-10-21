# === artefact_generator_live.py ===
# NDVI/NDWI Artefaktberechnung mit Live-Status während STD
# Kompatibel mit Colab (2025-10)

import os, sys, time, datetime, shutil, psutil
import numpy as np
import rasterio
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


def local_std_blockwise(arr, size=11, block_size=512, progress_cb=None):
    """Berechnet lokale STD blockweise und ruft progress_cb nach jedem Block."""
    result = np.zeros_like(arr, dtype=np.float32)
    ny, nx = arr.shape
    total_blocks = (ny // block_size + 1) * (nx // block_size + 1)
    done_blocks = 0

    for y in range(0, ny, block_size):
        for x in range(0, nx, block_size):
            block = arr[y:y+block_size, x:x+block_size]
            result[y:y+block_size, x:x+block_size] = generic_filter(
                block, np.nanstd, size=size, mode="nearest"
            )
            done_blocks += 1
            if progress_cb:
                progress_cb(done_blocks, total_blocks)
    return result


def save_raster(out_path, profile, data):
    """Schreibt TIFF robust (lokal + Copy auf Drive)."""
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw", BIGTIFF="IF_NEEDED")
    tmp = "/content/tmp_" + os.path.basename(out_path)
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)
    shutil.move(tmp, out_path)


def generate_environmental_artefacts_live(block_size=512, std_size=11, downsample=5):
    """Berechnet STD + Moran (Geary optional) mit Live-Status."""
    dirs = cfg["data"]["raster_dirs"]

    for index, index_dir in dirs.items():
        full_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(full_path, exist_ok=True)

        raster_files = sorted([
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.endswith(".tif") and index in f and "_STD_" not in f
        ])
        print(f"\n📂 {index}: {len(raster_files)} Raster gefunden")

        for i, path in enumerate(raster_files, 1):
            base = os.path.basename(path)
            month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
            print(f"\n🧮 [{i}/{len(raster_files)}] {index}_{month} @ {datetime.datetime.now().strftime('%H:%M:%S')}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            # --- STD mit Live-Monitor ---
            t0 = time.time()
            print("  ▶️ Berechne lokale STD (mit Live-Status) ...")
            def progress(done, total):
                if done % 5 == 0:
                    ram = psutil.virtual_memory().percent
                    pct = 100 * done / total
                    print(f"     🧮 Block {done}/{total} ({pct:.1f}%) – RAM {ram:.1f}%")
                    sys.stdout.flush()

            std_map = local_std_blockwise(arr, size=std_size, block_size=block_size, progress_cb=progress)
            print(f"  ✅ STD fertig in {time.time()-t0:.1f}s")

            out_std = os.path.join(full_path, f"{index}_STD_{month}.tif")
            save_raster(out_std, prof, std_map)
            print(f"  💾 Gespeichert: {os.path.basename(out_std)}")

            # --- Moran (reduziert) ---
            print("  ▶️ Berechne lokalen Moran ...")
            sub = arr[::downsample, ::downsample]
            sub[np.isnan(sub)] = 0
            w = lat2W(*sub.shape)
            w.transform = "r"
            moran = Moran_Local(sub.ravel(), w)
            moran_map = np.repeat(np.repeat(moran.Is.reshape(sub.shape), downsample, 0), downsample, 1)
            moran_map = moran_map[:arr.shape[0], :arr.shape[1]]

            out_moran = os.path.join(full_path, f"{index}_MORAN_{month}.tif")
            save_raster(out_moran, prof, moran_map)
            print(f"  ✅ MORAN gespeichert ({time.time()-t0:.1f}s gesamt)")

    print("\n🏁 Lauf abgeschlossen – alle Artefakte erzeugt.")