# 🛰️ inat-habitat-modeling

**Modulare Analysepipeline zur Habitatmodellierung auf Basis von iNaturalist-Punktdaten und Satellitenbildern aus Google Earth Engine (Sentinel-2).**

Ziel ist es, Umweltmerkmale an Fundorten von Arten automatisiert zu extrahieren, mit Hintergrundpunkten zu vergleichen und als Eingabe für Habitat- oder Verbreitungsmodelle aufzubereiten.

---

## 📦 Hauptfunktionen

* 🔍 **iNaturalist-Daten laden** (API oder lokal gespeicherte CSV)
* 🛰️ **Umweltmerkmale extrahieren** mit Google Earth Engine:

  * NDVI, NDWI, monatliche Zeitreihen
  * statistische Kennwerte, räumliche Heterogenität
* 🟢 **Beobachtungspunkte + Hintergrundpunkte** als Trainingsdaten kombinieren
* 📤 **Export als CSV** für Modellierung (z. B. scikit-learn, XGBoost)
* 🧪 **QA-Tools**: Heatmaps, Zeitreihenplots, Punktdiagnosen

---

## 📂 Projektstruktur

```bash
inat-habitat-modeling/
├── notebooks/                # Explorative Phasen
│   ├── 01_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_background_sampling.ipynb
│   ├── 04_model_input.ipynb
│   └── 05_qa_visuals.ipynb
├── scripts/                 # Wiederverwendbare Funktionen
│   ├── gee_features.py
│   ├── inat_loader.py
│   ├── sampling.py
│   └── model_utils.py
├── data/
│   ├── raw/                 # z. B. iNat-CSV, lokale Punkte
│   └── processed/           # Modellinput als CSV
├── config.py                # Zentrale Parameter (Region, Taxon, Filter)
├── requirements.txt         # Python-Abhängigkeiten
└── README.md
```

---

## 🚧 Technische Herausforderungen (aktuell)

* Sentinel-2-Stapel mit vielen Masken → führen oft zu leeren Extraktionen
* Punktbasierte Reduktion (`sampleRegions`) funktioniert nur bei ausreichend gültiger NDVI-Abdeckung
* `.getInfo()`-Loops nicht für große Punktmengen geeignet → Batch-Export nötig

---

## ✅ Status

* [x] Punktdaten-Abfrage aus iNaturalist
* [x] NDVI/NDWI-Feature-Extraktion mit Zeitfenster & Puffer
* [x] Hintergrundpunkte & Merge
* [x] CSV-Export für Modelltraining
* [ ] Batch-Verarbeitung großer Mengen (Export aus GEE)
* [ ] Modelltraining & Evaluation

---

## 📘 Nutzung

```bash
# Klonen
git clone https://github.com/deinname/inat-habitat-modeling.git
cd inat-habitat-modeling

# Notebook starten oder Funktionen aus scripts/ verwenden
```

---

## 📄 Lizenz

MIT License – offen für Forschung & Projekte.

---

## 🤝 Mitwirken

Pull Requests willkommen – Ziel ist ein flexibles, allgemein nutzbares Toolkit für Geo-Biodiversitätsanalysen auf Punktbasis.
