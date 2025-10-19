# config.py – zentrale Parameter für das Projekt

# ▶️ Earth Engine Projekt-ID (falls verwendet)
PROJECT_ID = "inaturalist-474012"

# ▶️ Untersuchungsgebiet (Berlin + Umgebung)
GEOMETRY = {
    "bbox": [12.8, 52.2, 13.8, 52.8],
    "ee_geometry": None  # wird später als ee.Geometry.Rectangle(...) gesetzt
}

# ▶️ Standard-Parameter für Sentinel-2 Filterung
S2_START_DATE = "2022-01-01"
S2_END_DATE = "2022-12-31"
CLOUD_FILTER = 20

# ▶️ Ziel-Taxon (z. B. Laetiporus sulphureus)
TAXON_ID = 53713  # iNaturalist Taxon-ID
TAXON_NAME = "Laetiporus sulphureus"

# ▶️ Ort für iNat-Suche (optional)
PLACE_ID = 97394  # Deutschland

# ▶️ Feature-Extraktion
NDVI_BANDS = ["B8", "B4"]
NDWI_BANDS = ["B3", "B8"]
DAYS_BACK = 30
BUFFER_RADIUS = 100  # in Metern

# ▶️ Ausgabeordner
OUTPUT_PATH = "data/processed"
RAW_PATH = "data/raw"

# Wird im Main-Skript ergänzt:
# GEOMETRY['ee_geometry'] = ee.Geometry.Rectangle(GEOMETRY['bbox'])
