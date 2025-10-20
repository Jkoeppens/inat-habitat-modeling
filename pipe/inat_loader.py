import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time
import re
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg  # zentrale Konfiguration

def slugify(text):
    return re.sub(r'\W+', '_', text.strip().lower())

def make_filename(prefix, taxon_name, region_name, ext="csv", date=None):
    slug_taxon = slugify(taxon_name)
    slug_region = slugify(region_name)
    date_str = f"_{date}" if date else ""
    return f"{prefix}_{slug_taxon}_{slug_region}{date_str}.{ext}"

def load_inat_observations(taxon_id=None, bbox=None, max_accuracy=None, max_pages=None, quality_grade=None):
    """
    Lädt Beobachtungen von iNaturalist über die API.
    Falls keine Parameter übergeben werden, werden Defaults aus der config verwendet.

    Returns:
        GeoDataFrame mit Beobachtungen (EPSG:4326)
    """
    taxon_id = taxon_id or cfg["inat"].get("default_taxon_id")
    bbox = bbox or cfg["inat"].get("bbox_default")
    max_accuracy = max_accuracy or cfg["inat"].get("max_accuracy", 30)
    max_pages = max_pages or cfg["inat"].get("max_pages", 50)
    quality_grade = quality_grade or cfg["inat"].get("quality_grade", "research")

    if not taxon_id or not bbox:
        raise ValueError("Taxon-ID und BBox müssen definiert sein (via Parameter oder config).")

    print(f"🔍 iNaturalist API-Abfrage gestartet für Taxon {taxon_id}...")

    params = {
        "taxon_id": taxon_id,
        "quality_grade": quality_grade,
        "geo": True,
        "nelat": bbox[3], "nelng": bbox[2],
        "swlat": bbox[1], "swlng": bbox[0],
        "per_page": 200,
        "page": 1
    }

    all_obs = []
    for page in range(1, max_pages + 1):
        params["page"] = page
        try:
            r = requests.get("https://api.inaturalist.org/v1/observations", params=params, timeout=10)
            r.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"⚠️ API-Fehler bei Seite {page}: {e}")
            break

        res = r.json().get("results", [])
        if not res:
            break
        all_obs.extend(res)

        if len(res) < 200:
            break
        time.sleep(1)

    if not all_obs:
        print("ℹ️ Keine Beobachtungen im angegebenen Gebiet gefunden.")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    df = pd.json_normalize(all_obs)
    df = df[df["geojson.coordinates"].notnull()]
    df = df[df["positional_accuracy"] <= max_accuracy]

    df["lat"] = df["geojson.coordinates"].apply(lambda x: x[1])
    df["lon"] = df["geojson.coordinates"].apply(lambda x: x[0])
    df["observed_on"] = pd.to_datetime(df["observed_on"])
    df["geometry"] = df.apply(lambda row: Point(row["lon"], row["lat"]), axis=1)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf[["geometry", "observed_on", "id", "lat", "lon"]].rename(columns={"id": "obs_id"})
