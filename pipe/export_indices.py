import ee
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config.config import cfg

def init_gee(project_id=None):
    """
    Initialisiert Google Earth Engine mit optionalem Projekt.
    """
    project_id = project_id or cfg["gee"]["default_project"]
    try:
        ee.Initialize(project=project_id)
        print(f"✅ Earth Engine initialisiert mit Projekt: {project_id}")
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"🔑 Authentifizierung erfolgreich mit Projekt: {project_id}")

def mask_scl(img):
    scl = img.select('SCL')
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))  # Vegetation, Boden, Wasser
    return img.updateMask(mask)

def compute_index(img, index):
    if index == 'NDVI':
        return img.normalizedDifference(['B8', 'B4']).rename('NDVI')
    elif index == 'NDWI':
        return img.normalizedDifference(['B3', 'B8']).rename('NDWI')
    else:
        raise ValueError("Index muss 'NDVI' oder 'NDWI' sein")

def get_monthly_index(year, month, region, index):
    start = ee.Date.fromYMD(year, month, 1)
    end = start.advance(1, 'month')

    col = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')\
        .filterBounds(region)\
        .filterDate(start, end)\
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))\
        .map(mask_scl)\
        .map(lambda img: img.addBands(compute_index(img, index)))

    return col.select(index).median().clip(region)

def export_monthly_index(year=None, months=None, region=None, index='NDVI', folder=None):
    """
    Exportiert monatliche Mittelwerte von NDVI/NDWI für eine Region aus GEE nach Google Drive.

    Parameter:
        - year: Jahr (z. B. 2023)
        - months: Liste von Monaten
        - region: ee.Geometry (oder None → Standard aus cfg)
        - index: 'NDVI' oder 'NDWI'
        - folder: Zielordner auf Drive (optional, sonst aus cfg)
    """
    year = year or cfg["export"]["start_year"]
    months = months or cfg["export"]["months"]
    folder = folder or cfg["data"]["raster_dirs"][index]

    # Region aus cfg laden falls nicht übergeben
    if region is None:
        bbox = cfg["inat"]["bbox_default"]
        region = ee.Geometry.Rectangle(bbox)

    for m in months:
        image = get_monthly_index(year, m, region, index)
        description = f"{index}_BerlinBB_{year}_{m:02d}"
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            region=region.coordinates().getInfo(),
            scale=cfg["export"]["scale"],
            crs='EPSG:4326',
            maxPixels=int(cfg["export"]["max_pixels"])
        )
        task.start()
        print(f"🚀 Export gestartet: {description}")
