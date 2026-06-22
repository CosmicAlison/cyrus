"""
Reads Surya's prediction.nc output and extracts physically meaningful
threat signals that Agent 1 (HelioAnalyst) can reason over.

Output schema matches RawForecastPayload in cyrus/core/schemas.py.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

# AIA channels most sensitive to flare activity
FLARE_CHANNELS = ["131", "171", "304", "94"]

# HMI channels for magnetic complexity (CME driver)
MAGNETIC_CHANNELS = ["Bz", "Bx", "By"]

# EUV proxy channels for ionospheric / CommsOps impact
EUV_CHANNELS = ["94", "131", "171", "193", "211", "304", "335"]

# Thresholds for flare class estimation (normalised intensity)
FLARE_CLASS_THRESHOLDS = {
    "X": 0.85,
    "M": 0.65,
    "C": 0.45,
    "B": 0.25,
}


def parse_prediction_nc(
    nc_path: Path,
    job_id: str,
    start_dt: datetime,
    end_dt: datetime,
) -> dict[str, Any]:
    """
    Parse a Surya prediction NetCDF file into a structured forecast payload.

    Returns a dict matching RawForecastPayload schema:
    {
        "job_id": str,
        "forecast_start": str,       # ISO datetime
        "forecast_end": str,
        "timesteps": [               # one entry per rollout step
            {
                "timestamp": str,
                "flare_probability": float,      # 0-1
                "estimated_flare_class": str,    # X/M/C/B/A
                "peak_intensity_per_channel": dict[str, float],
                "delta_intensity_per_channel": dict[str, float],
                "active_region": {
                    "x": int, "y": int,          # pixel coords of max intensity
                    "radius_pixels": int,
                },
                "euv_integrated_flux": float,    # normalised 0-1
                "magnetic_complexity": float,    # normalised 0-1 (Bz variance)
                "solar_wind_proxy": float,       # normalised 0-1
            }
        ],
        "summary": {
            "max_flare_probability": float,
            "peak_flare_class": str,
            "peak_timestamp": str,
            "mean_euv_flux": float,
            "max_magnetic_complexity": float,
        }
    }
    """
    log.info("Parsing prediction: %s", nc_path)
    ds = xr.open_dataset(str(nc_path))

    prediction = ds["prediction"].values  # (T, C, H, W)
    channels = list(ds["channel"].values)
    timestamps = list(ds["time"].values)

    # Build channel → index map
    ch_idx = {ch: i for i, ch in enumerate(channels)}

    timestep_data = []
    for t_idx, ts in enumerate(timestamps):
        frame = prediction[t_idx]  # (C, H, W)

        # --- Flare signal (AIA channels) ---
        flare_intensities = {}
        for ch in FLARE_CHANNELS:
            if ch in ch_idx:
                flare_intensities[ch] = float(frame[ch_idx[ch]].max())

        # Composite flare probability: weighted max across sensitive channels
        weights = {"131": 0.35, "171": 0.25, "304": 0.25, "94": 0.15}
        flare_prob = sum(
            flare_intensities.get(ch, 0.0) * w for ch, w in weights.items()
        )
        flare_prob = min(flare_prob, 1.0)

        # --- Peak intensity across all channels ---
        peak_per_channel = {}
        for ch, idx in ch_idx.items():
            peak_per_channel[ch] = float(frame[idx].max())

        # --- Delta from input mean (rate of change = eruption signature) ---
        delta_per_channel = {}
        if "input" in ds:
            input_frames = ds["input"].values  # (T_in, C, H, W)
            input_mean = input_frames.mean(axis=0)  # (C, H, W)
            for ch, idx in ch_idx.items():
                delta = float(frame[idx].max()) - float(input_mean[idx].max())
                delta_per_channel[ch] = delta
        else:
            delta_per_channel = {ch: 0.0 for ch in channels}

        # --- Active region localisation (peak in AIA 171) ---
        active_region = {"x": 0, "y": 0, "radius_pixels": 0}
        if "171" in ch_idx:
            aia171 = frame[ch_idx["171"]]
            peak_idx = np.unravel_index(np.argmax(aia171), aia171.shape)
            # Estimate active region radius: pixels above 80% of max
            threshold = aia171.max() * 0.8
            region_mask = aia171 > threshold
            active_region = {
                "x": int(peak_idx[1]),
                "y": int(peak_idx[0]),
                "radius_pixels": int(np.sqrt(region_mask.sum() / np.pi)),
            }

        # --- EUV integrated flux (proxy for ionospheric impact) ---
        euv_values = [
            float(frame[ch_idx[ch]].mean())
            for ch in EUV_CHANNELS
            if ch in ch_idx
        ]
        euv_flux = float(np.mean(euv_values)) if euv_values else 0.0

        # --- Magnetic complexity (Bz spatial variance → CME likelihood) ---
        if "Bz" in ch_idx:
            bz = frame[ch_idx["Bz"]]
            mag_complexity = float(np.std(bz))  # already normalised 0-1
        else:
            mag_complexity = 0.0

        # --- Solar wind proxy (HMI Doppler + Bz magnitude) ---
        solar_wind_proxy = 0.0
        if "doppler" in ch_idx:
            solar_wind_proxy = float(frame[ch_idx["doppler"]].mean())

        # --- Flare class estimation ---
        flare_class = "A"
        for cls, threshold in FLARE_CLASS_THRESHOLDS.items():
            if flare_prob >= threshold:
                flare_class = cls
                break

        timestep_data.append({
            "timestamp": str(ts),
            "flare_probability": round(flare_prob, 4),
            "estimated_flare_class": flare_class,
            "peak_intensity_per_channel": {
                k: round(v, 4) for k, v in peak_per_channel.items()
            },
            "delta_intensity_per_channel": {
                k: round(v, 4) for k, v in delta_per_channel.items()
            },
            "active_region": active_region,
            "euv_integrated_flux": round(euv_flux, 4),
            "magnetic_complexity": round(mag_complexity, 4),
            "solar_wind_proxy": round(solar_wind_proxy, 4),
        })

    # --- Summary across all timesteps ---
    all_probs = [t["flare_probability"] for t in timestep_data]
    peak_idx_summary = int(np.argmax(all_probs))
    peak_ts = timestep_data[peak_idx_summary]["timestamp"]
    max_prob = max(all_probs)

    # Peak flare class from max probability timestep
    peak_class = "A"
    for cls, threshold in FLARE_CLASS_THRESHOLDS.items():
        if max_prob >= threshold:
            peak_class = cls
            break

    summary = {
        "max_flare_probability": round(max_prob, 4),
        "peak_flare_class": peak_class,
        "peak_timestamp": peak_ts,
        "mean_euv_flux": round(float(np.mean([t["euv_integrated_flux"] for t in timestep_data])), 4),
        "max_magnetic_complexity": round(float(max(t["magnetic_complexity"] for t in timestep_data)), 4),
    }

    log.info(
        "Parse complete — peak class: %s, max probability: %.2f",
        summary["peak_flare_class"],
        summary["max_flare_probability"],
    )

    ds.close()

    return {
        "job_id": job_id,
        "forecast_start": start_dt.isoformat(),
        "forecast_end": end_dt.isoformat(),
        "timesteps": timestep_data,
        "summary": summary,
    }