# artefact_generator.py – Erstellung von NDVI/NDWI-Ableitungen

import os
import numpy as np
import rasterio
from rasterio import windows
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local, Geary_Local
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from config.config import cfg


def local_std(arr, size=11):
    return generic_filter(arr, np.nanstd, size=size, mode='nearest')

def save_raster(out_path, profile, data):
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)

def generate_environmental_artefacts():
    dirs = cfg["data"]["raster_dirs"]

    for index, index_dir in dirs.items():
        full_path = os.path.join(cfg["data"]["base_dir"], index_dir)
        os.makedirs(full_path, exist_ok=True)

        raster_files = sorted([
            os.path.join(full_path, f) for f in os.listdir(full_path)
            if f.endswith(".tif") and index in f
        ])

        print(f"📂 {index}: {len(raster_files)} Raster gefunden")
        for f in raster_files[:3]: print("  -", os.path.basename(f))

        for path in tqdm(raster_files, desc=f"{index}-Analyse"):
            base = os.path.basename(path)
            month = base.split("_")[-2] + "_" + base.split("_")[-1].split(".")[0]
            print(f"\n🧮 Verarbeite {index} → {month}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            # Lokale STD
            print("  • STD-Berechnung …")
            std_map = local_std(arr, size=11)
            out_std = os.path.join(full_path, f"{index}_STD_{month}.tif")
            save_raster(out_std, prof, std_map)
            print(f"  ✅ STD gespeichert: {os.path.basename(out_std)}")

            # Moran & Geary (downsampled)
            print("  • Lokaler Moran & Geary …")
            sub_arr = arr[::5, ::5]
            sub_arr[np.isnan(sub_arr)] = 0
            w = lat2W(*sub_arr.shape)
            w.transform = "r"

            moran = Moran_Local(sub_arr.ravel(), w)
            geary = Geary_Local(sub_arr.ravel(), w)

            moran_map = np.repeat(np.repeat(moran.Is.reshape(sub_arr.shape), 5, axis=0), 5, axis=1)
            geary_map = np.repeat(np.repeat(geary.Cs.reshape(sub_arr.shape), 5, axis=0), 5, axis=1)

            moran_map = moran_map[:arr.shape[0], :arr.shape[1]]
            geary_map = geary_map[:arr.shape[0], :arr.shape[1]]

            out_moran = os.path.join(full_path, f"{index}_MORAN_{month}.tif")
            out_geary = os.path.join(full_path, f"{index}_GEARY_{month}.tif")

            save_raster(out_moran, prof, moran_map)
            save_raster(out_geary, prof, geary_map)

            print(f"  ✅ MORAN & GEARY gespeichert: {os.path.basename(out_moran)}, {os.path.basename(out_geary)}")

    print("\n🏁 Fertig! Alle Artefakte berechnet.")
