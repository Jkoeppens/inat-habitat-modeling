# ============================================================
# 📘 env_feature_extractor.py
# Version: 2025-10 | Nur Lesemodus – ergänzt NDVI/NDWI + Artefaktwerte
# ============================================================

import os
import rasterio
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from glob import glob

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def find_raster(base_dir, prefix, year, month):
    """
    Findet Rasterdateien wie NDVI_BerlinBB_2021_05.tif oder NDVI_STD_2021_05.tif
    """
    pattern = f"{prefix}_*{year}_{str(month).zfill(2)}*.tif"
    files = sorted(glob(os.path.join(base_dir, pattern)))
    return files[0] if files else None


def read_raster_value(raster_path, lon, lat):
    """
    Liest den Pixelwert eines Raster-TIFs an einer gegebenen Koordinate.
    """
    if raster_path is None or not os.path.exists(raster_path):
        return np.nan
    try:
        with rasterio.open(raster_path) as src:
            row, col = src.index(lon, lat)
            val = src.read(1)[row, col]
            if val == src.nodata:
                return np.nan
            return float(val)
    except Exception:
        return np.nan


def extract_features(cfg):
    """
    Ergänzt Beobachtungsdaten (Pilze, Meisen) um NDVI/NDWI + Artefaktwerte.
    Erwartet: inaturalist_combined.csv im output_dir.
    """

    base_dir = cfg["paths"]["base_data_dir"]
    out_dir = cfg["paths"]["output_dir"]

    ndvi_dir = cfg["paths"]["ndvi_dir"]
    ndwi_dir = cfg["paths"]["ndwi_dir"]

    infile = os.path.join(out_dir, "inaturalist_combined.csv")
    if not os.path.exists(infile):
        raise FileNotFoundError(f"❌ {infile} fehlt – bitte zuerst inat_loader ausführen!")

    df = pd.read_csv(infile)
    df["date"] = pd.to_datetime(df["date"])

    results = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc="🔍 Extrahiere Umweltwerte"):
        lat, lon = row["latitude"], row["longitude"]
        year, month = row["date"].year, row["date"].month

        ndvi_file = find_raster(ndvi_dir, "NDVI", year, month)
        ndwi_file = find_raster(ndwi_dir, "NDWI", year, month)

        # Artefakt-Raster prüfen
        ndvi_std = find_raster(ndvi_dir, "NDVI_STD", year, month)
        ndvi_moran = find_raster(ndvi_dir, "NDVI_MORAN", year, month)
        ndvi_geary = find_raster(ndvi_dir, "NDVI_GEARY", year, month)

        ndwi_std = find_raster(ndwi_dir, "NDWI_STD", year, month)
        ndwi_moran = find_raster(ndwi_dir, "NDWI_MORAN", year, month)
        ndwi_geary = find_raster(ndwi_dir, "NDWI_GEARY", year, month)

        vals = {
            "latitude": lat,
            "longitude": lon,
            "date": row["date"].strftime("%Y-%m-%d"),
            "species": row["species"],
            "NDVI": read_raster_value(ndvi_file, lon, lat),
            "NDWI": read_raster_value(ndwi_file, lon, lat),
            "NDVI_STD": read_raster_value(ndvi_std, lon, lat),
            "NDVI_MORAN": read_raster_value(ndvi_moran, lon, lat),
            "NDVI_GEARY": read_raster_value(ndvi_geary, lon, lat),
            "NDWI_STD": read_raster_value(ndwi_std, lon, lat),
            "NDWI_MORAN": read_raster_value(ndwi_moran, lon, lat),
            "NDWI_GEARY": read_raster_value(ndwi_geary, lon, lat),
        }
        results.append(vals)

    df_out = pd.DataFrame(results)

    outfile = os.path.join(out_dir, "inaturalist_features.csv")
    df_out.to_csv(outfile, index=False)
    print(f"\n✅ Features gespeichert: {outfile}")
    print(df_out.describe(include='all'))

    return df_out