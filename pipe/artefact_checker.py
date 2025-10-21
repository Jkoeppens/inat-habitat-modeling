# ============================================================
# 📘 artefact_checker.py
# Version: 2025-10 | Prüft und ergänzt fehlende Umwelt-Artefakte
# ============================================================

import os
from glob import glob
from datetime import datetime
from tqdm import tqdm
from pipe import artefact_generator_fast  # nutzt deine schnelle Version

# ------------------------------------------------------------
# Hilfsfunktionen
# ------------------------------------------------------------

def list_rasters(base_dir, prefix):
    """Listet alle TIFs mit Prefix (z.B. NDVI oder NDWI)."""
    return sorted(glob(os.path.join(base_dir, f"{prefix}_*.tif")))


def extract_date_from_filename(filename):
    """Extrahiert Jahr und Monat aus Rasternamen."""
    try:
        parts = os.path.basename(filename).split("_")
        year, month = int(parts[-2]), int(parts[-1].split(".")[0])
        return year, month
    except Exception:
        return None, None


def check_missing_artefacts(base_dir, prefix="NDVI"):
    """Prüft, welche Artefakte (STD, MORAN, GEARY) pro Monat fehlen."""
    base_files = list_rasters(base_dir, prefix)
    status = []

    for path in base_files:
        year, month = extract_date_from_filename(path)
        if not year:
            continue
        stem = f"{prefix}_{year}_{str(month).zfill(2)}"
        expected = [
            os.path.join(base_dir, f"{prefix}_STD_{year}_{str(month).zfill(2)}.tif"),
            os.path.join(base_dir, f"{prefix}_MORAN_{year}_{str(month).zfill(2)}.tif"),
            os.path.join(base_dir, f"{prefix}_GEARY_{year}_{str(month).zfill(2)}.tif"),
        ]
        existing = [os.path.exists(e) for e in expected]
        if not all(existing):
            missing = [n for n, ok in zip(["STD", "MORAN", "GEARY"], existing) if not ok]
            status.append((stem, missing))
    return status


def generate_missing_artefacts(cfg, mode="NDVI"):
    """Erzeugt gezielt fehlende Artefakte."""
    base_dir = cfg["paths"]["ndvi_dir"] if mode == "NDVI" else cfg["paths"]["ndwi_dir"]
    missing = check_missing_artefacts(base_dir, prefix=mode)

    if not missing:
        print(f"✅ Alle Artefakte für {mode} vollständig.")
        return

    print(f"\n⚙️ Fehlende Artefakte für {mode}: {len(missing)} Datensätze\n")
    for stem, types in missing:
        print(f"   - {stem}: {', '.join(types)} fehlen")

    # Nutzerhinweis: Blockweise Verarbeitung
    print("\n🚀 Starte gezielte Berechnung fehlender Artefakte ...")

    for stem, types in tqdm(missing):
        base_path = os.path.join(base_dir, f"{stem}.tif")
        if not os.path.exists(base_path):
            print(f"⚠️ Basisraster fehlt: {base_path}")
            continue

        try:
            artefact_generator_fast.generate_environmental_artefacts_fast(
                single_file=base_path,
                compute_std="STD" in types,
                compute_moran="MORAN" in types,
                compute_geary="GEARY" in types,
            )
        except Exception as e:
            print(f"❌ Fehler bei {stem}: {e}")

    print("\n🏁 Artefakt-Check abgeschlossen.")