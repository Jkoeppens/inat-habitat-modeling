# export_ndvi_ndwi.py

import ee

def init_gee(project_id='inaturalist-474012'):
    try:
        ee.Initialize(project=project_id)
        print(f"✅ Earth Engine initialisiert mit Projekt: {project_id}")
    except ee.EEException:
        ee.Authenticate()
        ee.Initialize(project=project_id)
        print(f"🔑 Authentifizierung erfolgreich mit Projekt: {project_id}")


def mask_scl(img):
    scl = img.select('SCL')
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
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


def export_monthly_index(year, months, region, index='NDVI', folder=None):
    folder = folder or f"{index}_Exports"
    for m in months:
        image = get_monthly_index(year, m, region, index)
        description = f"{index}_BerlinBB_{year}_{m:02d}"
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=folder,
            fileNamePrefix=description,
            region=region.coordinates().getInfo(),
            scale=10,
            crs='EPSG:4326',
            maxPixels=1e13
        )
        task.start()
        print(f"🚀 Export gestartet: {description}")


# Beispiel zur Nutzung
if __name__ == "__main__":
    init_gee()

    # Region: Berlin + Brandenburg
    geometry = ee.Geometry.Rectangle([12.8, 52.2, 13.8, 52.8])

    year = 2023
    months = [6, 7, 8, 9]  # Beispielmonate

    export_monthly_index(year, months, geometry, index='NDVI')
    export_monthly_index(year, months, geometry, index='NDWI')
