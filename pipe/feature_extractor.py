# pipe/feature_extractor.py – refaktoriert mit zentraler config

import os
import re
import rasterio
import numpy as np
import pandas as pd
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shapely.geometry import Point
from rasterio.windows import from_bounds

from config.config import cfg  # zentrale Konfiguration


def slugify(text):
    return re.sub(r'\W+', '_', text.strip().lower())


def make_filename(prefix, taxon_name, region_name, index_name, ext="csv", date=None):
    slug_taxon = slugify(taxon_name)
    slug_region = slugify(region_name)
    slug_index = slugify(index_name)
    date_str = f"_{date}" if date else ""
    return f"{prefix}_{slug_taxon}_{slug_region}_{slug_index}{date_str}.{ext}"


def get_matching_raster_path(observed_on, var='NDVI', lag_months=None):
    base_dir = cfg["data"]["base_dir"]
    raster_subdir = cfg["data"]["raster_dirs"].get(var, "")
    lag = lag_months if lag_months is not None else cfg["feature_extraction"].get("lag_months", 1)

    date = pd.to_datetime(observed_on) - pd.DateOffset(months=lag)
    filename = f"{var}_BerlinBB_{date.year}_{date.month:02d}.tif"
    return os.path.join(base_dir, raster_subdir, filename)


def extract_features_from_raster(gdf, var='NDVI', lag_months=None, buffer_m=None):
    """
    Extrahiert Punkt- und Puffer-Features aus Rasterdateien basierend auf Beobachtungsdaten.

    Args:
        gdf (GeoDataFrame): Beobachtungsdaten mit 'observed_on' und Geometrie.
        var (str): Index ('NDVI' oder 'NDWI')
        lag_months (int): Zeitversatz zur Beobachtung (z. B. 1 Monat zurück)
        buffer_m (int): Radius für Bufferstatistik (in Metern)

    Returns:
        DataFrame mit extrahierten Features
    """
    features = []
    buffer = buffer_m if buffer_m is not None else cfg["feature_extraction"].get("buffer_m", 100)
    lag = lag_months if lag_months is not None else cfg["feature_extraction"].get("lag_months", 1)

    for _, row in gdf.iterrows():
        try:
            obs_date = pd.to_datetime(row['observed_on'])
            raster_path = get_matching_raster_path(obs_date, var, lag)

            if not os.path.exists(raster_path):
                print(f"❌ Raster fehlt: {raster_path}")
                continue

            with rasterio.open(raster_path) as src:
                coords = (row.geometry.x, row.geometry.y)

                # Punktwert
                point_row, point_col = src.index(*coords)
                val_at_point = src.read(1)[point_row, point_col]

                # Buffer via from_bounds
                window = from_bounds(*Point(*coords).buffer(buffer).bounds, transform=src.transform)
                buf = src.read(1, window=window, boundless=True, fill_value=np.nan)
                buf = buf.astype(float)
                buf[buf < -1] = np.nan
                buf[buf > 1] = np.nan
                std_val = np.nanstd(buf)

                features.append({
                    'obs_id': row.get('obs_id', 'unknown'),
                    f'{var}_at_point': val_at_point,
                    f'{var}_std_{buffer}m': std_val,
                    'observed_on': row['observed_on'],
                    'lon': row.geometry.x,
                    'lat': row.geometry.y
                })
        except Exception as e:
            print(f"⚠️ Fehler bei Beobachtung {row.get('obs_id', 'unknown')}: {e}")
            continue

    return pd.DataFrame(features)
