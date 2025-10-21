# === 🌍 iNaturalist Fetcher für Colab ===
# Lädt Beobachtungen für Ziel- und Vergleichsart gemäß local.yaml
# Speichert CSVs in /outputs/

import requests, pandas as pd, os, time
from datetime import datetime
from tqdm import tqdm

def fetch_inat_observations(taxon_id, bbox, start_date, end_date, max_pages=50, sleep=1.0):
    """Lädt iNaturalist-Beobachtungen per API."""
    all_results = []
    base_url = "https://api.inaturalist.org/v1/observations"

    bbox_str = ",".join(map(str, bbox))
    print(f"🔍 Lade Beobachtungen für Taxon {taxon_id} (BBox={bbox_str}) ...")

    for page in tqdm(range(1, max_pages + 1), desc=f"Seiten für Taxon {taxon_id}"):
        params = {
            "taxon_id": taxon_id,
            "nelat": bbox[3], "nelng": bbox[2],
            "swlat": bbox[1], "swlng": bbox[0],
            "d1": start_date, "d2": end_date,
            "per_page": 200,
            "page": page,
            "order_by": "observed_on",
            "order": "desc"
        }
        try:
            resp = requests.get(base_url, params=params, timeout=30)
            if resp.status_code != 200:
                print(f"⚠️ Fehler (Status {resp.status_code}), breche ab.")
                break

            results = resp.json().get("results", [])
            if not results:
                break

            all_results.extend(results)
            time.sleep(sleep)
        except Exception as e:
            print("❌ API-Fehler:", e)
            break

    print(f"✅ {len(all_results)} Beobachtungen geladen.")
    return all_results


def parse_results(results, species_name):
    """Extrahiert relevante Felder aus iNaturalist JSON."""
    records = []
    for obs in results:
        records.append({
            "species": species_name,
            "taxon_id": obs.get("taxon", {}).get("id"),
            "latitude": obs.get("geojson", {}).get("coordinates", [None, None])[1],
            "longitude": obs.get("geojson", {}).get("coordinates", [None, None])[0],
            "observed_on": obs.get("observed_on"),
            "quality_grade": obs.get("quality_grade"),
            "user_login": obs.get("user", {}).get("login"),
            "place_guess": obs.get("place_guess"),
        })
    return pd.DataFrame(records)


def run_inat_fetch(cfg_local):
    """Gesamtpipeline: lädt Ziel- und Vergleichsart & speichert CSV."""
    base_dir = cfg_local["paths"]["output_dir"]
    os.makedirs(base_dir, exist_ok=True)

    species_target = cfg_local["inat"]["species"]["target"]
    species_contrast = cfg_local["inat"]["species"]["contrast"]
    period = cfg_local["inat"]["species"]["period"]
    bbox = cfg_local["inat"]["region_bbox"]
    max_pages = cfg_local["inat"]["max_pages"]

    print(f"📅 Zeitraum: {period['start']} → {period['end']}")
    print(f"🗺️ BBox: {bbox}")

    # --- Zielart ---
    res_target = fetch_inat_observations(
        taxon_id=species_target["id"],
        bbox=bbox,
        start_date=period["start"],
        end_date=period["end"],
        max_pages=max_pages
    )
    df_target = parse_results(res_target, species_target["name"])
    out_target = os.path.join(base_dir, f"inaturalist_{species_target['name'].replace(' ', '_')}.csv")
    df_target.to_csv(out_target, index=False)
    print(f"💾 Gespeichert: {out_target}")

    # --- Vergleichsart ---
    res_contrast = fetch_inat_observations(
        taxon_id=species_contrast["id"],
        bbox=bbox,
        start_date=period["start"],
        end_date=period["end"],
        max_pages=max_pages
    )
    df_contrast = parse_results(res_contrast, species_contrast["name"])
    out_contrast = os.path.join(base_dir, f"inaturalist_{species_contrast['name'].replace(' ', '_')}.csv")
    df_contrast.to_csv(out_contrast, index=False)
    print(f"💾 Gespeichert: {out_contrast}")

    # --- Kombinierte Datei ---
    df_all = pd.concat([df_target, df_contrast], ignore_index=True)
    out_combined = os.path.join(base_dir, "inaturalist_combined.csv")
    df_all.to_csv(out_combined, index=False)
    print(f"💾 Kombiniert gespeichert: {out_combined} ({len(df_all)} Zeilen)")

    return df_all


# === 🧩 Aufruf für Colab ===
if "cfg_local" in globals():
    df_combined = run_inat_fetch(cfg_local)
    print("\n✅ Fetch abgeschlossen. Vorschau:")
    display(df_combined.head())
else:
    print("⚠️ Keine cfg_local geladen. Bitte zuerst local.yaml laden!")