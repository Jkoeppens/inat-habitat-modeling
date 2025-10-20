# eda.py – Explorative Analyse von Umweltstruktur

import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def run_eda(path):
    df = pd.read_csv(path)
    df["geometry"] = df["geometry"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    print("🧩 Datensatz:", len(gdf), "Funde")

    cols = [
        "NDVI_at_point", "NDWI_at_point",
        "NDVI_std_100m", "NDWI_std_100m",
        "Moran_I_local", "Geary_C_local",
        "NDWI_Moran_I_local", "NDWI_Geary_C_local"
    ]
    num = gdf[cols].select_dtypes(np.number)
    print(num.describe())

    corr = num.corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="RdYlBu_r", center=0, fmt=".2f")
    plt.title("Korrelationsmatrix der Umweltkennwerte")
    plt.show()

    sns.jointplot(x="NDVI_at_point", y="NDWI_at_point", data=gdf, kind="hex")
    plt.suptitle("NDVI vs. NDWI – Vegetationsvitalität und Feuchte", y=1.02)
    plt.show()

    sns.pairplot(
        gdf,
        vars=["NDVI_at_point", "NDVI_std_100m", "Moran_I_local", "Geary_C_local"],
        diag_kind="kde",
    )
    plt.suptitle("Vegetationsstruktur und Autokorrelation", y=1.02)
    plt.show()

    sns.pairplot(
        gdf,
        vars=["NDWI_at_point", "NDWI_std_100m", "NDWI_Moran_I_local", "NDWI_Geary_C_local"],
        diag_kind="kde",
    )
    plt.suptitle("Feuchtigkeitsstruktur und Autokorrelation", y=1.02)
    plt.show()

    df["observed_on"] = pd.to_datetime(df["observed_on"], errors="coerce")
    df["month"] = df["observed_on"].dt.month
    plt.figure(figsize=(10,4))
    sns.boxplot(x="month", y="NDVI_at_point", data=df)
    plt.title("NDVI zum Fundzeitpunkt (monatlich)")
    plt.show()

# Beispiel:
# run_eda("/content/drive/MyDrive/iNaturalist/inat_c_nebularis_env_full.csv")
