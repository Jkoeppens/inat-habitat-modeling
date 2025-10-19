# scripts/inat_loader.py

import requests
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import time

def load_inat_observations(taxon_id, bbox, max_accuracy=30, max_pages=50):
    """
    Lädt Beobachtungen von iNaturalist per API.

    Args:
        taxon_id (int): iNat Taxon-ID
        bbox (list): [minlon, minlat, maxlon, maxlat]
        max_accuracy (int): maximale Lageungenauigkeit in Metern
        max_pages (int): maximale Seitenzahl für API-Abruf

    Returns:
        GeoDataFrame mit Beobachtungen (EPSG:4326)
    """
    params = {
        "taxon_id": taxon_id,
        "quality_grade": "research",
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

    if "geojson.coordinates" not in df.columns:
        print("⚠️ Beobachtungen ohne Koordinaten – keine gültigen Punkte vorhanden.")
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    df = df[df["geojson.coordinates"].notnull()]
    df = df[df["positional_accuracy"] <= max_accuracy]

    df["geometry"] = df["geojson.coordinates"].apply(lambda coords: Point(coords[0], coords[1]))

    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
