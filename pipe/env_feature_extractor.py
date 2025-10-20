# env_feature_extractor.py – Punktweise Extraktion von Umwelt-Artefakten

import os
import pandas as pd
import geopandas as gpd
from shapely import wkt
from rasterstats import point_query
from tqdm import tqdm
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def extract_environmental_features(csv_in, ndvi_dir, ndwi_dir, out_csv):
    df = pd.read_csv(csv_in)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    for col in [
        "NDVI_std_100m", "NDWI_std_100m",
        "Moran_I_local", "Geary_C_local",
        "NDWI_Moran_I_local", "NDWI_Geary_C_local"
    ]:
        gdf[col] = np.nan

    months = sorted(gdf["source_month"].dropna().unique())
    print(f"📅 Verarbeite Monate: {months}")

    for month in tqdm(months):
        subset = gdf[gdf["source_month"] == month].copy()
        if subset.empty:
            continue

        def try_extract(path, column):
            if os.path.exists(path):
                vals = point_query(subset, path)
                gdf.loc[subset.index, column] = vals

        # NDVI
        try_extract(os.path.join(ndvi_dir, f"NDVI_STD_{month}.tif"), "NDVI_std_100m")
        try_extract(os.path.join(ndvi_dir, f"NDVI_MORAN_{month}.tif"), "Moran_I_local")
        try_extract(os.path.join(ndvi_dir, f"NDVI_GEARY_{month}.tif"), "Geary_C_local")

        # NDWI
        try_extract(os.path.join(ndwi_dir, f"NDWI_STD_{month}.tif"), "NDWI_std_100m")
        try_extract(os.path.join(ndwi_dir, f"NDWI_MORAN_{month}.tif"), "NDWI_Moran_I_local")
        try_extract(os.path.join(ndwi_dir, f"NDWI_GEARY_{month}.tif"), "NDWI_Geary_C_local")

    gdf.to_csv(out_csv, index=False)
    print(f"\n✅ Ergebnisse gespeichert unter: {out_csv}")
