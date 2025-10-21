# ============================================================
# artefact_generator_fast.py (robuste Version mit Diagnose & Fortsetzung)
# ============================================================

import os, sys, datetime, traceback, psutil, numpy as np, rasterio
from rasterio.enums import BigTiff
from scipy.ndimage import generic_filter
from libpysal.weights import lat2W
from esda import Moran_Local
from tqdm import tqdm
from numba import njit
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import cfg


# === 🧠 Helferfunktionen ===
def log(msg):
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    print(f"{ts} {msg}")
    log_dir = os.path.join(cfg["data"]["base_dir"], "logs")
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "artefact_run.log"), "a") as f:
        f.write(f"{ts} {msg}\n")


def safe_step(func, step_name):
    """Führt Schritt sicher aus, protokolliert Fehler + Systemzustand."""
    try:
        log(f"▶️ Starte Schritt: {step_name}")
        result = func()
        log(f"✅ Erfolgreich: {step_name}")
        return result
    except Exception as e:
        tb = traceback.format_exc()
        log(f"❌ Fehler in Schritt '{step_name}': {e}\n{tb}")
        ram = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=1)
        log(f"🧠 RAM={ram}% | CPU={cpu}% | Zeitpunkt={datetime.datetime.now()}")
        return None


def save_raster(out_path, profile, data):
    """Speichert Raster sicher mit BigTIFF-Fallback."""
    meta = profile.copy()
    meta.update(dtype="float32", count=1, compress="lzw", BIGTIFF="IF_NEEDED")
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data.astype("float32"), 1)


# === 🧮 Beschleunigte lokale STD mit Numba ===
@njit(parallel=True)
def local_std_numba(arr, size=11):
    pad = size // 2
    padded = np.pad(arr, pad, mode="reflect")
    out = np.zeros_like(arr)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            win = padded[i:i+size, j:j+size]
            m = np.mean(win)
            out[i, j] = np.sqrt(np.mean((win - m) ** 2))
    return out


# === 🧩 Hauptfunktion ===
def generate_environmental_artefacts_fast(block_size=1024, std_size=11, downsample=5):
    base = cfg["data"]["base_dir"]
    dirs = cfg["data"]["raster_dirs"]

    for index, rel_dir in dirs.items():
        dir_path = os.path.join(base, rel_dir)
        os.makedirs(dir_path, exist_ok=True)

        all_rasters = sorted([
            os.path.join(dir_path, f) for f in os.listdir(dir_path)
            if f.endswith(".tif") and index in f and "STD" not in f
        ])

        # === Überspringe bereits vollständige Artefakte ===
        todo = []
        for path in all_rasters:
            name = os.path.basename(path)
            month_tag = "_".join(name.split("_")[-2:]).replace(".tif", "")
            std_file = os.path.join(dir_path, f"{index}_STD_{month_tag}.tif")
            mor_file = os.path.join(dir_path, f"{index}_MORAN_{month_tag}.tif")
            gea_file = os.path.join(dir_path, f"{index}_GEARY_{month_tag}.tif")
            if not (os.path.exists(std_file) and os.path.exists(mor_file) and os.path.exists(gea_file)):
                todo.append(path)

        log(f"📂 {index}: {len(todo)} Raster ohne vollständige Artefakte")

        for path in tqdm(todo, desc=f"{index}-Analyse"):
            base_name = os.path.basename(path)
            month_tag = "_".join(base_name.split("_")[-2:]).replace(".tif", "")
            log(f"\n🧮 Verarbeite {index} → {month_tag}")

            with rasterio.open(path) as src:
                arr = src.read(1).astype("float32")
                prof = src.profile
                arr[arr == src.nodata] = np.nan

            # === Lokale STD ===
            std_map = safe_step(lambda: local_std_numba(arr, size=std_size), "Lokale STD")
            if std_map is None:
                log("⚠️ STD-Berechnung übersprungen. Weiter mit nächstem Raster.")
                continue

            out_std = os.path.join(dir_path, f"{index}_STD_{month_tag}.tif")
            safe_step(lambda: save_raster(out_std, prof, std_map), "STD speichern")

            # === Moran & Geary (Downsample) ===
            def moran_geary():
                sub_arr = arr[::downsample, ::downsample]
                sub_arr[np.isnan(sub_arr)] = 0
                w = lat2W(*sub_arr.shape)
                w.transform = "r"
                moran = Moran_Local(sub_arr.ravel(), w)
                moran_map = np.repeat(np.repeat(moran.Is.reshape(sub_arr.shape), downsample, 0), downsample, 1)
                moran_map = moran_map[:arr.shape[0], :arr.shape[1]]

                # vereinfachte Geary-Schätzung
                mu = np.mean(sub_arr)
                var = np.var(sub_arr)
                diffs = (sub_arr - mu)
                N = sub_arr.size
                W = np.sum(list(w.weights.values()))
                geary_val = (N - 1) / (2 * W) * np.sum([w_i * np.mean((diffs[i] - diffs)**2)
                                                       for i, w_i in enumerate(w.weights.values())]) / var
                geary_map = np.full_like(sub_arr, geary_val)
                geary_map = np.repeat(np.repeat(geary_map, downsample, 0), downsample, 1)
                geary_map = geary_map[:arr.shape[0], :arr.shape[1]]
                return moran_map, geary_map

            result = safe_step(moran_geary, "Moran & Geary")
            if result is None:
                continue

            moran_map, geary_map = result
            out_moran = os.path.join(dir_path, f"{index}_MORAN_{month_tag}.tif")
            out_geary = os.path.join(dir_path, f"{index}_GEARY_{month_tag}.tif")

            safe_step(lambda: save_raster(out_moran, prof, moran_map), "Moran speichern")
            safe_step(lambda: save_raster(out_geary, prof, geary_map), "Geary speichern")

    log("\n🏁 Fertig! Alle Artefakte berechnet.")