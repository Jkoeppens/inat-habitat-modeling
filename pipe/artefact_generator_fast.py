# ============================================================
# ⚡ artefact_generator_fast.py
# Version: 2025-10 | Unterstützt Einzelfallberechnung & Blockmodus
# ============================================================

import os, time, numpy as np, rasterio
from tqdm import tqdm
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local, Geary_Local

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def local_std(arr, size=11):
    """Lokale Standardabweichung mit NaN-handling (blockweise)."""
    arr = np.nan_to_num(arr, nan=np.nanmean(arr))
    return generic_filter(arr, np.nanstd, size=size, mode='nearest')


def save_raster(out_path, profile, data):
    """Speichert ein GeoTIFF im float32-Format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw", BIGTIFF="IF_NEEDED")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)


def compute_moran_geary(arr, downsample=5):
    """Berechnet lokale Moran- & Geary-Werte auf einem Downsample."""
    sub = arr[::downsample, ::downsample].astype("float64")
    sub[np.isnan(sub)] = 0
    w = lat2W(*sub.shape)
    w.transform = "r"

    moran = Moran_Local(sub.ravel(), w)
    geary = Geary_Local(sub.ravel(), w)

    moran_map = np.repeat(np.repeat(moran.Is.reshape(sub.shape), downsample, axis=0), downsample, axis=1)
    geary_map = np.repeat(np.repeat(geary.Cs.reshape(sub.shape), downsample, axis=0), downsample, axis=1)

    return moran_map[:arr.shape[0], :arr.shape[1]], geary_map[:arr.shape[0], :arr.shape[1]]


# ------------------------------------------------------------
# Hauptfunktion
# ------------------------------------------------------------

def generate_environmental_artefacts_fast(
    base_dir=None,
    single_file=None,
    compute_std=True,
    compute_moran=True,
    compute_geary=True,
    std_size=11,
    downsample=5,
):
    """
    Berechnet Umwelt-Artefakte (lokale STD, Moran, Geary)
    - entweder für alle Raster im Ordner (base_dir)
    - oder gezielt für eine einzelne Datei (single_file)
    """

    if single_file:
        files = [single_file]
    elif base_dir:
        files = [os.path.join(base_dir, f) for f in os.listdir(base_dir)
                 if f.endswith(".tif") and "_STD_" not in f and "_MORAN_" not in f and "_GEARY_" not in f]
    else:
        raise ValueError("Bitte base_dir oder single_file angeben!")

    print(f"\n📊 Starte Artefaktlauf ({len(files)} Raster)")
    for i, path in enumerate(files, 1):
        t0 = time.time()
        base = os.path.basename(path)
        prefix = base.split("_")[0]
        month = "_".join(base.replace(".tif", "").split("_")[-2:])

        print(f"\n🧮 [{i}/{len(files)}] {base}")
        with rasterio.open(path) as src:
            arr = src.read(1).astype("float32")
            prof = src.profile
            arr[arr == src.nodata] = np.nan

        if compute_std:
            print("  ▶️ Berechne STD ...")
            std_map = local_std(arr, size=std_size)
            out_std = os.path.join(base_dir or os.path.dirname(path), f"{prefix}_STD_{month}.tif")
            save_raster(out_std, prof, std_map)
            print(f"     ✅ STD gespeichert: {os.path.basename(out_std)}")

        if compute_moran or compute_geary:
            print("  ▶️ Berechne Moran & Geary ...")
            try:
                moran_map, geary_map = compute_moran_geary(arr, downsample=downsample)
                if compute_moran:
                    out_moran = os.path.join(base_dir or os.path.dirname(path), f"{prefix}_MORAN_{month}.tif")
                    save_raster(out_moran, prof, moran_map)
                    print(f"     ✅ MORAN gespeichert: {os.path.basename(out_moran)}")
                if compute_geary:
                    out_geary = os.path.join(base_dir or os.path.dirname(path), f"{prefix}_GEARY_{month}.tif")
                    save_raster(out_geary, prof, geary_map)
                    print(f"     ✅ GEARY gespeichert: {os.path.basename(out_geary)}")
            except Exception as e:
                print(f"     ⚠️ Fehler bei Moran/Geary: {e}")

        print(f"     ⏱️ Dauer: {time.time() - t0:.1f}s")

    print("\n🏁 Lauf abgeschlossen.")