# === 🧮 Beschleunigte lokale STD mit Numba (blockweise, robust) ===
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def _local_std_core(arr, size=11):
    """
    Kernfunktion für lokale Standardabweichung.
    Läuft voll in Numba ohne np.pad().
    """
    pad = size // 2
    h, w = arr.shape
    out = np.empty((h, w), dtype=np.float32)

    for i in prange(h):
        for j in range(w):
            # Grenzen des lokalen Fensters (Reflexion an Rändern)
            i_min = max(0, i - pad)
            i_max = min(h, i + pad + 1)
            j_min = max(0, j - pad)
            j_max = min(w, j + pad + 1)

            # lokale Mittelwert- und Varianzberechnung
            mean = 0.0
            count = 0.0
            for x in range(i_min, i_max):
                for y in range(j_min, j_max):
                    val = arr[x, y]
                    if not np.isnan(val):
                        mean += val
                        count += 1
            if count == 0:
                out[i, j] = np.nan
                continue
            mean /= count

            var = 0.0
            for x in range(i_min, i_max):
                for y in range(j_min, j_max):
                    val = arr[x, y]
                    if not np.isnan(val):
                        diff = val - mean
                        var += diff * diff
            out[i, j] = np.sqrt(var / count)
    return out


def local_std_numba_blockwise(arr, size=11, block_size=1024):
    """
    Führt lokale STD-Berechnung blockweise aus, um Speicher zu sparen.
    """
    h, w = arr.shape
    out = np.full((h, w), np.nan, dtype=np.float32)

    # Blockweise Verarbeitung mit Überlappung (wegen Fenstergröße)
    pad = size // 2
    for y in range(0, h, block_size):
        for x in range(0, w, block_size):
            y_end = min(h, y + block_size)
            x_end = min(w, x + block_size)

            # Block mit Puffer-Rand extrahieren
            y0 = max(0, y - pad)
            y1 = min(h, y_end + pad)
            x0 = max(0, x - pad)
            x1 = min(w, x_end + pad)

            sub = arr[y0:y1, x0:x1]
            sub_std = _local_std_core(sub, size=size)

            # Ausschnitt in Ergebnis zurückschreiben (ohne Padding)
            sy0 = pad if y > 0 else 0
            sy1 = sub_std.shape[0] - pad if y_end < h else sub_std.shape[0]
            sx0 = pad if x > 0 else 0
            sx1 = sub_std.shape[1] - pad if x_end < w else sub_std.shape[1]
            out[y:y_end, x:x_end] = sub_std[sy0:sy1, sx0:sx1]

    return out