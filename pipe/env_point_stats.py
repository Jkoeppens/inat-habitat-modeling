# ============================================================
# 📘 env_point_stats.py
# Zweck: Berechnet STD, Moran & Geary nur an Fundpunkten
# ============================================================

import os
import numpy as np
import pandas as pd
import rasterio
from esda import Moran, Geary
from libpysal.weights import lat2W
from tqdm import tqdm

def extract_pointwise_stats(cfg, window=11):
    """
    Berechnet lokale STD, Moran & Geary direkt an Fundpunkten.
    - Nutzt vorhandene NDVI/NDWI Raster
    - Kein globales Artefakt nötig
    """

    base_dir = cfg["paths"]["base_data_dir"]
    output_dir = cfg["paths"]["output_dir"]
    combined_path = os.path.join(output_dir, "inaturalist_combined.csv")

    if not os.path.exists(combined_path):
        raise FileNotFoundError("❌ combined.csv nicht gefunden.")

    df = pd.read_csv(combined_path)
    print(f"📄 {len(df)} Punkte geladen.")

    ndvi_dir = cfg["paths"]["ndvi_dir"]
    ndwi_dir = cfg["paths"]["ndwi_dir"]

    raster_paths = {
        "NDVI": sorted([os.path.join(ndvi_dir, f) for f in os.listdir(ndvi_dir) if f.endswith(".tif")]),
        "NDWI": sorted([os.path.join(ndwi_dir, f) for f in os.listdir(ndwi_dir) if f.endswith(".tif")]),
    }

    results = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="🧩 Punktstatistiken"):
        lat, lon = row["latitude"], row["longitude"]

        row_result = {"latitude": lat, "longitude": lon, "species": row.get("species", "")}

        for key, rasters in raster_paths.items():
            for path in rasters:
                month = "_".join(os.path.basename(path).split("_")[-2:]).replace(".tif", "")
                try:
                    with rasterio.open(path) as src:
                        # Pixelposition finden
                        px, py = src.index(lon, lat)
                        pad = window // 2
                        # Lokales Fenster lesen
                        win = rasterio.windows.Window(px - pad, py - pad, window, window)
                        arr = src.read(1, window=win).astype("float32")
                        arr[arr == src.nodata] = np.nan
                        vals = arr.flatten()
                        vals = vals[~np.isnan(vals)]
                        if len(vals) < 5:
                            continue

                        # STD
                        row_result[f"{key}_STD_{month}"] = float(np.nanstd(vals))

                        # Moran & Geary (lokal)
                        if vals.shape[0] > 8:
                            n = int(np.sqrt(len(vals)))
                            if n*n == len(vals):  # quadratisches Fenster
                                w = lat2W(n, n)
                                w.transform = "r"
                                mor = Moran(vals, w)
                                gea = Geary(vals, w)
                                row_result[f"{key}_MORAN_{month}"] = float(mor.I)
                                row_result[f"{key}_GEARY_{month}"] = float(gea.C)
                except Exception as e:
                    pass

        results.append(row_result)

    df_out = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "inat_points_localstats.csv")
    df_out.to_csv(out_path, index=False)

    print(f"\n✅ Fertig! Lokale Punktstatistiken gespeichert unter: {out_path}")
    return df_out