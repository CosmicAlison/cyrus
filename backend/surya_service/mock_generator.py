"""
Generates all four Surya model outputs synthetically for local dev.
Produces the same dict structure as the real pipelines.

generate_all() is called by runner.py when SURYA_USE_MOCK=true.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Any

import numpy as np

_WAVELENGTHS = np.linspace(6.5, 33.3, 1343)
_SXR_BINS    = _WAVELENGTHS <= 15.0
_THERM_BINS  = (_WAVELENGTHS >= 17.0) & (_WAVELENGTHS <= 30.0)
_HE2_BINS    = (_WAVELENGTHS >= 29.9) & (_WAVELENGTHS <= 30.9)


def generate_all(
    solar_dt: datetime,
    tick: int = 0,
    cached_wind: dict | None = None,
) -> dict[str, Any]:
    """Generate all four model outputs for one inference tick."""
    rng = random.Random(int(solar_dt.timestamp()) // 3600) 

    flare    = _gen_flare(solar_dt, rng)
    ar_now   = _gen_ar(solar_dt, rng, seed_offset=0)
    ar_flare = None

    if flare["prediction"] == 1 and flare.get("time_target"):
        try:
            target_dt = datetime.fromisoformat(
                flare["time_target"].replace(" ", "T")
            )
            ar_rng  = random.Random(int(target_dt.timestamp()) // 3600 + 1)
            ar_flare = _gen_ar(target_dt, ar_rng, seed_offset=1,
                               base_ar=ar_now)
        except Exception:
            pass

    if cached_wind and tick % 3 != 0:
        wind = {**cached_wind, "_cached": True}
    else:
        wind = _gen_wind(solar_dt, rng)

    euv = _gen_euv(solar_dt, rng, flare["prediction"])

    return {
        "flare":    flare,
        "ar_now":   ar_now,
        "ar_flare": ar_flare,
        "euv":      euv,
        "wind":     wind,
    }


#Per-model generators

def _gen_flare(solar_dt: datetime, rng: random.Random) -> dict[str, Any]:
    prob_roll = rng.random()
    if prob_roll < 0.20:
        prob = rng.uniform(0.70, 0.95)
    elif prob_roll < 0.50:
        prob = rng.uniform(0.45, 0.70)
    else:
        prob = rng.uniform(0.05, 0.45)

    prediction  = 1 if prob >= 0.5 else 0
    time_target = (solar_dt + timedelta(hours=2)).isoformat()

    # Simulate GOES class from probability
    if prob >= 0.85:
        goes = "X" + str(round(rng.uniform(1.0, 9.9), 1))
    elif prob >= 0.65:
        goes = "M" + str(round(rng.uniform(1.0, 9.9), 1))
    elif prob >= 0.45:
        goes = "C" + str(round(rng.uniform(1.0, 9.9), 1))
    else:
        goes = "B" + str(round(rng.uniform(1.0, 9.9), 1))

    return {
        "prediction":  prediction,
        "probability": round(prob, 4),
        "time_input":  solar_dt.isoformat(),
        "time_target": time_target,
        "goes_class":  goes,
    }


def _gen_ar(
    solar_dt: datetime,
    rng: random.Random,
    seed_offset: int = 0,
    base_ar: dict | None = None,
) -> dict[str, Any]:
    """
    Generate AR mask statistics.
    If base_ar provided, evolve from it (for ar_flare — regions grow/rotate).
    """
    n_regions = rng.randint(1, 4)
    centroids = []

    for i in range(n_regions):
        if base_ar and i < len(base_ar.get("centroids", [])):
            # Evolve existing region: slight rotation + growth
            base = base_ar["centroids"][i]
            x    = base["x"] + rng.gauss(0, 0.03)   # solar rotation drift
            y    = base["y"] + rng.gauss(0, 0.01)
            area = min(0.25, base["area_frac"] * rng.uniform(1.0, 1.3))
        else:
            x    = rng.uniform(-0.8, 0.8)
            y    = rng.uniform(-0.8, 0.8)
            area = rng.uniform(0.01, 0.15)

        x    = max(-1.0, min(1.0, x))
        y    = max(-1.0, min(1.0, y))
        prox = min(1.0, math.sqrt(x**2 + y**2))

        centroids.append({
            "x":              round(x, 4),
            "y":              round(y, 4),
            "area_frac":      round(area, 6),
            "disk_proximity": round(prox, 4),
        })

    centroids.sort(key=lambda c: c["area_frac"], reverse=True)
    total_area = sum(c["area_frac"] for c in centroids)

    return {
        "timestamp_input":     solar_dt.isoformat(),
        "timestamp_target":    solar_dt.isoformat(),
        "active_region_count": len(centroids),
        "centroids":           centroids,
        "total_area_frac":     round(total_area, 6),
        "dominant_region":     centroids[0] if centroids else None,
    }


def _gen_euv(
    solar_dt: datetime,
    rng: random.Random,
    flare_prediction: int,
) -> dict[str, Any]:
    # Base spectrum: smooth curve with coronal peaks
    base = np.zeros(1343)
    # Soft X-ray peak (6.5–15nm)
    sxr_peak = rng.uniform(0.3, 0.9) * (1.0 + flare_prediction * 0.3)
    base[_SXR_BINS] = sxr_peak * np.exp(
        -(_WAVELENGTHS[_SXR_BINS] - 10.0)**2 / 15.0
    )
    # EUV peaks (17–30nm)
    base[_THERM_BINS] = rng.uniform(0.4, 0.85) * np.exp(
        -(_WAVELENGTHS[_THERM_BINS] - 22.0)**2 / 20.0
    )
    # He II 30.4nm spike
    base[_HE2_BINS] = rng.uniform(0.5, 0.95)
    # Add noise
    base += np.random.RandomState(int(solar_dt.timestamp()) % (2**31)).normal(
        0, 0.01, 1343
    )
    spectrum = np.clip(base, 0.0, 1.0)

    indices  = np.linspace(0, 1342, 50, dtype=int)
    mini     = spectrum[indices].tolist()

    return {
        "time_input":         solar_dt.isoformat(),
        "time_target":        (solar_dt + timedelta(hours=1)).isoformat(),
        "spectrum":           spectrum.tolist(),
        "integrated_flux":    round(float(spectrum.mean()), 4),
        "soft_xray_flux":     round(float(spectrum[_SXR_BINS].mean()), 4),
        "thermospheric_flux": round(float(spectrum[_THERM_BINS].mean()), 4),
        "he2_flux":           round(float(spectrum[_HE2_BINS].mean()), 4),
        "spectrum_mini":      [round(v, 4) for v in mini],
    }


def _gen_wind(solar_dt: datetime, rng: random.Random) -> dict[str, Any]:
    # Realistic solar wind ranges
    speed = rng.gauss(450, 80)     # km/s, typical 300-700
    speed = max(250.0, min(800.0, speed))

    # Bz: random walk around 0, can go strongly negative during storms
    bz    = rng.gauss(-2.0, 6.0)   # nT, negative = southward = GIC risk
    bz    = max(-25.0, min(10.0, bz))

    return {
        "time_input":  solar_dt.isoformat(),
        "time_target": (solar_dt + timedelta(days=4)).isoformat(),
        "speed_kms":   round(speed, 1),
        "bz_gsm":      round(bz, 2),
        "bx_gse":      round(rng.gauss(0, 3), 2),
        "by_gsm":      round(rng.gauss(0, 3), 2),
        "density":     round(max(0.5, rng.gauss(6.0, 2.5)), 2),
        "_cached":     False,
    }