"""
EUV Spectra Prediction inference pipeline.

Task dir:  downstream_examples/euv_spectra_prediction/
Weights:   assets/euv_spectra_weights.pth (nasa-ibm-ai4science/euv_spectra_surya)
Model:     HelioSpectformer1D (different head from AR — 1D spectral output)

From tutorial:
  - imports: run_inference, create_spectrum_plots (no infer_single_sample exposed)
  - Dataset: NetCDF with train_time/val_time/test_time + train_spectra/val_spectra/test_spectra
  - Output: 1343-dim continuous spectrum vector [0,1] normalised
  - Model only trains 5 params: cls_token, linear.weight/bias, unembed.weight/bias
  - R²=0.985, Spectral correlation=0.994 on test set

We hook into the model directly post-load rather than using run_inference
(which only prints and saves PNG, doesn't return the array).

Output dict:
  {
    "time_input":          str,
    "time_target":         str,
    "spectrum":            list[float],   # 1343 values
    "integrated_flux":     float,         # mean across all bins
    "soft_xray_flux":      float,         # mean bins ~6.5-15nm (CommsOps driver)
    "thermospheric_flux":  float,         # mean bins ~17-30nm  (SatOps driver)
    "he2_flux":            float,         # ~30.4nm bin         (ionospheric)
    "spectrum_mini":       list[float],   # 50-point downsample for dashboard sparkline
  }
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline, _utc

log = logging.getLogger(__name__)

CHECKPOINT = "assets/euv_spectra_weights.pth"

# Wavelength axis: 1343 bins linearly spaced from 6.5 to 33.3 nm
_WAVELENGTHS = np.linspace(6.5, 33.3, 1343)

# Bin index ranges for derived signals
# soft X-ray: 6.5–15 nm → roughly bins 0–430
_SXR_BINS   = (_WAVELENGTHS <= 15.0)
# thermospheric EUV: 17–30 nm → roughly bins 490–1190
_THERM_BINS = (_WAVELENGTHS >= 17.0) & (_WAVELENGTHS <= 30.0)
# He II 30.4 nm ± 0.5 nm
_HE2_BINS   = (_WAVELENGTHS >= 29.9) & (_WAVELENGTHS <= 30.9)


class EUVSpectraPipeline(BaseInferencePipeline):
    task_dir    = "euv_spectra_prediction"
    result_type = "euv_spectra"

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")

        # HelioSpectformer1D — different from the 2D AR model
        from infer import load_model
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = load_model(
            config=self._config,
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[euv_spectra] model loaded — output dim: 1343")

    def _load_timestamp_index(self) -> list[datetime]:
        """
        EUV dataset is a NetCDF file with test_time variable.
        config["data"] should reference the NetCDF path.
        """
        data_cfg = self._config.get("data", {})
        nc_path  = (
            data_cfg.get("data_path")
            or data_cfg.get("netcdf_path")
            or data_cfg.get("test_data_path")
        )

        if not nc_path:
            candidates = list((self._task_path / "assets").rglob("*.nc"))
            nc_path    = str(candidates[0]) if candidates else None

        if not nc_path:
            log.warning("[euv_spectra] no NetCDF index found")
            return []

        full = Path(nc_path) if Path(nc_path).is_absolute() \
               else self._task_path / nc_path

        # EUV NetCDF stores test_time separately
        return self._index_from_netcdf_euv(str(full))

    def _index_from_netcdf_euv(self, nc_path: str) -> list[datetime]:
        """EUV dataset has test_time, val_time, train_time variables."""
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(nc_path)
            # Try test_time first, then val_time
            for var in ("test_time", "val_time", "time"):
                if var in ds.variables:
                    raw = nc4.num2date(ds[var][:], ds[var].units)
                    ds.close()
                    return sorted([
                        datetime(t.year, t.month, t.day, t.hour,
                                 t.minute, t.second, tzinfo=timezone.utc)
                        for t in raw
                    ])
            ds.close()
        except Exception as exc:
            log.warning("[euv_spectra] NetCDF index load failed: %s", exc)
        return []

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        import torch

        loader, scalers = self._build_fixed_dataloader(sample_idx, data_type="test")

        with torch.no_grad():
            for batch, metadata in loader:
                if self.device != "cpu":
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    dtype=self._config["dtype"],
                    enabled="cuda" in self.device,
                ):
                    output = self._model(batch)  # (B, 1343)

                spectrum = output[0].cpu().float().numpy()  # (1343,)
                # Clip to [0,1] — model trained on log-normalised values
                spectrum = np.clip(spectrum, 0.0, 1.0)

                ts_input  = _extract_ts(metadata, "timestamps_input",   0)
                ts_target = _extract_ts(metadata, "timestamps_targets",  0)

                return _build_result(spectrum, ts_input, ts_target)

        return _empty_euv_result()


# ── Signal extraction ─────────────────────────────────────────────────────────

def _build_result(spectrum: np.ndarray,
                  ts_input: str, ts_target: str) -> dict[str, Any]:
    integrated    = float(spectrum.mean())
    soft_xray     = float(spectrum[_SXR_BINS].mean())   if _SXR_BINS.any()   else 0.0
    thermospheric = float(spectrum[_THERM_BINS].mean())  if _THERM_BINS.any() else 0.0
    he2           = float(spectrum[_HE2_BINS].mean())    if _HE2_BINS.any()   else 0.0

    # Downsample to 50 points for dashboard sparkline
    indices   = np.linspace(0, len(spectrum) - 1, 50, dtype=int)
    mini      = spectrum[indices].tolist()

    return {
        "time_input":         ts_input,
        "time_target":        ts_target,
        "spectrum":           spectrum.tolist(),
        "integrated_flux":    round(integrated, 4),
        "soft_xray_flux":     round(soft_xray, 4),
        "thermospheric_flux": round(thermospheric, 4),
        "he2_flux":           round(he2, 4),
        "spectrum_mini":      [round(v, 4) for v in mini],
    }


def _empty_euv_result() -> dict[str, Any]:
    return {
        "time_input": "", "time_target": "",
        "spectrum": [], "integrated_flux": 0.0,
        "soft_xray_flux": 0.0, "thermospheric_flux": 0.0,
        "he2_flux": 0.0, "spectrum_mini": [],
    }


def _extract_ts(meta: dict, key: str, idx: int) -> str:
    try:
        arr = meta[key]
        val = arr[idx] if hasattr(arr, "__getitem__") else arr
        return str(np.datetime_as_string(np.asarray(val).ravel()[0], unit="m"))
    except Exception:
        return ""