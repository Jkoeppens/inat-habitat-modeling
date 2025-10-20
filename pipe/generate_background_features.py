from pipe.feature_extractor import extract_features_from_raster, make_filename
from config.config import cfg
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os

# 📥 Fundpunkte laden
taxon_name = "Clitocybe nebularis"
region_name = "Berlin"
index = "NDVI"

fund_file = make_filename("features", taxon_name, region_name, index)
fund_path = os.path.join(cfg["paths"]["output_path"], fund_file)
fund_df = pd.read_csv(fund_path)
fund_df["label"] = 1

# 🌍 BBox aus Punkten
gdf_fund = gpd.GeoDataFrame(fund_df, geometry=gpd.points_from_xy(fund_df.lon, fund_df.lat), crs="EPSG:4326")
bbox = gdf_fund.total_bounds

# 🎲 Zufallspunkte generieren
n_bg = len(fund_df) * 3
np.random.seed(42)
random_lons = np.random.uniform(bbox[0], bbox[2], n_bg)
random_lats = np.random.uniform(bbox[1], bbox[3], n_bg)
random_dates = np.random.choice(fund_df["observed_on"].values, size=n_bg)

bg_df = pd.DataFrame({
    "lon": random_lons,
    "lat": random_lats,
    "observed_on": random_dates,
    "obs_id": [f"bg_{i}" for i in range(n_bg)]
})
bg_gdf = gpd.GeoDataFrame(bg_df, geometry=gpd.points_from_xy(bg_df.lon, bg_df.lat), crs="EPSG:4326")

# 🧪 Features extrahieren
bg_feat = extract_features_from_raster(bg_gdf, var=index)
bg_feat["label"] = 0

# 🔀 Kombinieren & speichern
full_df = pd.concat([fund_df, bg_feat], ignore_index=True)
full_df = full_df.sample(frac=1, random_state=42).reset_index(drop=True)

output_file = make_filename("samples", taxon_name, region_name, index)
output_path = os.path.join(cfg["paths"]["output_path"], output_file)
full_df.to_csv(output_path, index=False)

print(f"✅ Kombinierte Punktmenge gespeichert unter: {output_path}")
