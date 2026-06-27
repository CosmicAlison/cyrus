"""
Merges outputs from all four Surya downstream models into a single
RawForecastPayload dict that gets published to cyrus.raw_forecast.

Also implements threat delta detection, only triggers the full
LangGraph agent pipeline when the threat level changes meaningfully.
Two separate queues:
  cyrus.raw_forecast  → significant change → full 5-agent pipeline
  cyrus.telemetry     → every tick → dashboard-only update (no LLM cost)
"""

import logging
from datetime import datetime, timezone
from typing import Any

log = logging.getLogger(__name__)

# Minimum flare probability change to trigger agent pipeline
THREAT_DELTA_THRESHOLD = 0.15

# Minimum Bz change (nT) to trigger agent pipeline
BZ_DELTA_THRESHOLD = 5.0

# Minimum wind speed change (km/s) to trigger agent pipeline
WIND_DELTA_THRESHOLD = 100.0


class Aggregator:
    def __init__(self) -> None:
        self._previous: dict[str, Any] | None = None

    def build(
        self,
        solar_time: datetime,
        flare:      dict[str, Any],
        ar_now:     dict[str, Any],
        ar_flare:   dict[str, Any] | None,
        euv:        dict[str, Any],
        wind:       dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge all four model outputs into one payload.
        ar_flare is None when flare.prediction == 0.
        """
        payload = {
            # Provenance 
            "job_id":      None,   # filled in by runner.py
            "solar_time":  solar_time.isoformat(),
            "real_time":   datetime.now(timezone.utc).isoformat(),

            #Model 1: Flare Forecasting
            "flare": {
                "prediction":   flare.get("prediction",   0),
                "probability":  flare.get("probability",  0.0),
                "time_input":   flare.get("time_input",   ""),
                "time_target":  flare.get("time_target",  ""),
                "goes_class":   flare.get("goes_class",   ""),
            },

            #  Model 2a: AR segmentation — current state 
            "ar_now": {
                "timestamp_input":     ar_now.get("timestamp_input",  ""),
                "timestamp_target":    ar_now.get("timestamp_target", ""),
                "active_region_count": ar_now.get("active_region_count", 0),
                "centroids":           ar_now.get("centroids",           []),
                "total_area_frac":     ar_now.get("total_area_frac",     0.0),
                "dominant_region":     ar_now.get("dominant_region",     None),
            },

            # Model 2b: AR segmentation at flare time (None if no flare)
            "ar_flare": None if ar_flare is None else {
                "timestamp_input":     ar_flare.get("timestamp_input",  ""),
                "timestamp_target":    ar_flare.get("timestamp_target", ""),
                "active_region_count": ar_flare.get("active_region_count", 0),
                "centroids":           ar_flare.get("centroids",           []),
                "total_area_frac":     ar_flare.get("total_area_frac",     0.0),
                "dominant_region":     ar_flare.get("dominant_region",     None),
            },

            # Model 3: EUV Spectra
            "euv": {
                "time_input":         euv.get("time_input",         ""),
                "time_target":        euv.get("time_target",        ""),
                "integrated_flux":    euv.get("integrated_flux",    0.0),
                "soft_xray_flux":     euv.get("soft_xray_flux",     0.0),
                "thermospheric_flux": euv.get("thermospheric_flux", 0.0),
                "he2_flux":           euv.get("he2_flux",           0.0),
                "spectrum_mini":      euv.get("spectrum_mini",      []),
                # Full 1343-point spectrum stored but not forwarded to LLM agents
                "_spectrum_full":     euv.get("spectrum",           []),
            },

            # Model 4: Solar Wind
            "wind": {
                "time_input":  wind.get("time_input",  ""),
                "time_target": wind.get("time_target", ""),
                "speed_kms":   wind.get("speed_kms",   0.0),
                "bz_gsm":      wind.get("bz_gsm",      0.0),
                "bx_gse":      wind.get("bx_gse",      0.0),
                "by_gsm":      wind.get("by_gsm",      0.0),
                "density":     wind.get("density",      0.0),
                "cached":      wind.get("_cached",      False),
            },

            # Derived threat summary
            # Pre-computed for helio_analyst
            "threat_summary": _compute_threat_summary(flare, euv, wind, ar_now),
        }

        return payload

    def threat_delta_significant(
        self, current: dict[str, Any]
    ) -> bool:
        """
        Returns True if the threat level has changed enough since the last
        tick to warrant running the full LangGraph agent pipeline.

        On the first tick always returns True.
        """
        if self._previous is None:
            self._previous = current
            return True

        prev = self._previous
        curr = current

        # Flare prediction flip (0→1 or 1→0)
        if curr["flare"]["prediction"] != prev["flare"]["prediction"]:
            log.info("[aggregator] flare prediction flipped → triggering agents")
            self._previous = curr
            return True

        # Probability spike
        prob_delta = abs(
            curr["flare"]["probability"] - prev["flare"]["probability"]
        )
        if prob_delta >= THREAT_DELTA_THRESHOLD:
            log.info("[aggregator] flare probability delta %.2f → triggering agents", prob_delta)
            self._previous = curr
            return True

        # Bz crosses southward threshold (negative Bz = GIC risk)
        bz_prev = prev["wind"]["bz_gsm"]
        bz_curr = curr["wind"]["bz_gsm"]
        if abs(bz_curr - bz_prev) >= BZ_DELTA_THRESHOLD:
            log.info("[aggregator] Bz delta %.1f nT → triggering agents", bz_curr - bz_prev)
            self._previous = curr
            return True

        # Wind speed jump
        v_prev = prev["wind"]["speed_kms"]
        v_curr = curr["wind"]["speed_kms"]
        if abs(v_curr - v_prev) >= WIND_DELTA_THRESHOLD:
            log.info("[aggregator] wind speed delta %.0f km/s → triggering agents", v_curr - v_prev)
            self._previous = curr
            return True

        return False


def _compute_threat_summary(
    flare: dict, euv: dict, wind: dict, ar_now: dict
) -> dict[str, Any]:
    """
    Pre-compute derived threat signals for helio_analyst.
    These map directly to ThreatPayload fields.
    """
    flare_prob = flare.get("probability", 0.0)
    bz         = wind.get("bz_gsm", 0.0)
    speed      = wind.get("speed_kms", 0.0)

    # GIC risk: southward Bz (negative) + high speed → transformer danger
    # Normalise Bz to 0-1 risk: -20 nT = max risk, 0 nT = no risk
    bz_risk = max(0.0, min(1.0, -bz / 20.0))

    # Atmospheric drag risk: EUV thermospheric + flare probability
    drag_risk = min(1.0, euv.get("thermospheric_flux", 0.0) * 0.6
                    + flare_prob * 0.4)

    # CommsOps: soft X-ray flux drives HF blackout
    hf_blackout_risk = min(1.0, euv.get("soft_xray_flux", 0.0) * 0.7
                           + flare_prob * 0.3)

    # Severity classification (maps to ThreatPayload.severity)
    composite = flare_prob * 0.5 + bz_risk * 0.3 + drag_risk * 0.2
    if composite >= 0.75:
        severity = "extreme"
    elif composite >= 0.55:
        severity = "high"
    elif composite >= 0.35:
        severity = "moderate"
    else:
        severity = "low"

    return {
        "severity":             severity,
        "composite_risk":       round(composite, 4),
        "flare_probability":    round(flare_prob, 4),
        "gic_risk":             round(bz_risk, 4),
        "atmospheric_drag_risk":round(drag_risk, 4),
        "hf_blackout_risk":     round(hf_blackout_risk, 4),
        "bz_southward":         bz < 0,
        "bz_nT":                round(bz, 2),
        "wind_speed_kms":       round(speed, 1),
        "active_region_count":  ar_now.get("active_region_count", 0),
        # Activation flags for LangGraph routing
        "activate_commsops":    flare_prob > 0.25 or hf_blackout_risk > 0.3,
        "activate_satops":      flare_prob > 0.45 or drag_risk > 0.4,
        "activate_gridops":     bz_risk > 0.4 and speed > 400,
    }