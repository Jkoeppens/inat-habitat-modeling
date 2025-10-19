# pipe/feature_extractor.py

import os
import rasterio
import numpy as np
import pandas as pd
from rasterio.windows import from_bounds
from shapely.geometry import Point


def get_matching_raster_path(observed_on, base_path, var='NDVI', lag_months=0):
    date = pd.to_datetime(observed_on) - pd.DateOffset(months=lag_months)
    filename = f"{var}_BerlinBB_{date.year}_{date.month:02d}.tif"
    return os.path.join(base_path, filename)


def extract_features_from_raster(gdf, base_path, var='NDVI', lag_months=0, buffer_m=100):
    features = []

    for _, row in gdf.iterrows():
        obs_date = pd.to_datetime(row['observed_on'])
        raster_path = get_matching_raster_path(obs_date, base_path, var, lag_months)

        if not os.path.exists(raster_path):
            print(f"❌ Raster fehlt: {raster_path}")
            continue

        try:
            with rasterio.open(raster_path) as src:
                coords = (row.geometry.x, row.geometry.y)
                point_row, point_col = src.index(*coords)
                ndvi_at_point = src.read(1)[point_row, point_col]

                # Buffer-Statistik
                buffer_radius = buffer_m / src.res[0]  # Pixelanzahl bei gegebener Auflösung
                window_size = int(np.ceil(buffer_radius)) * 2 + 1

                row_off = max(0, point_row - window_size // 2)
                col_off = max(0, point_col - window_size // 2)
                window = src.read(1, window=rasterio.windows.Window(col_off, row_off, window_size, window_size))
                window = window.astype(float)
                window[window < -1] = np.nan
                window[window > 1] = np.nan
                ndvi_std = np.nanstd(window)

                features.append({
                    'obs_id': row['id'],
                    f'{var}_at_point': ndvi_at_point,
                    f'{var}_std_{buffer_m}m': ndvi_std,
                    'observed_on': row['observed_on'],
                    'lon': row.geometry.x,
                    'lat': row.geometry.y
                })
        except Exception as e:
            print(f"⚠️ Fehler bei {row['id']}: {e}")
            continue

    return pd.DataFrame(features)
