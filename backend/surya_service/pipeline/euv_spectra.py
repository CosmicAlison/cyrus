"""
EUV Spectra Prediction inference pipeline.

Task dir:  downstream_examples/euv_spectra_prediction/
Weights:   assets/euv_spectra_weights.pth (nasa-ibm-ai4science/euv_spectra_surya)
Model:     HelioSpectformer1D — 1D spectral output, different head from AR's 2D.

"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline

log = logging.getLogger(__name__)

CHECKPOINT = "assets/euv_spectra_weights.pth"

# Wavelength axis: 1343 bins linearly spaced from 6.5 to 33.3 nm
_WAVELENGTHS = np.linspace(6.5, 33.3, 1343)
_SXR_BINS   = (_WAVELENGTHS <= 15.0)
_THERM_BINS = (_WAVELENGTHS >= 17.0) & (_WAVELENGTHS <= 30.0)
_HE2_BINS   = (_WAVELENGTHS >= 29.9) & (_WAVELENGTHS <= 30.9)


class EUVSpectraPipeline(BaseInferencePipeline):
    task_dir    = "euv_spectra_prediction"
    result_type = "euv_spectra"

    dataset_class_name = "EVEDSDataset"
    index_path_key      = "infer_data_path"
    # UNVERIFIED — confirm against this task's own config/reference infer.py
    rollout_steps = 0

    extra_dataset_kwargs = {
        "ds_eve_index_path": "infer_solar_data_path",
        # If config_infer.yaml doesn't set these explicitly, EVEDSDataset
        # falls back to its own constructor defaults (ds_time_column=None,
        # ds_match_direction="forward") — verify that's actually correct
        # before relying on it. Add them here once confirmed, e.g.:
        # "ds_time_column": "eve_time_column_key",
        # "ds_time_tolerance": "eve_time_tolerance_key",
        # "ds_match_direction": "eve_match_direction_key",
    }

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")

        infer_module = self._import_task_module()
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = infer_module.load_model(
            config=self._config,
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[euv_spectra] model loaded - output dim: 1343")

    def _load_timestamp_index(self) -> list[datetime]:
        nc_path = self._config["data"].get("infer_data_path")
        if not nc_path:
            log.warning("[euv_spectra] no infer_data_path in config")
            return []

        suffix = Path(nc_path).suffix
        if suffix == ".csv":
            return self._index_from_csv(nc_path, col="timestep")
        return self._index_from_netcdf_euv(nc_path)

    def _index_from_netcdf_euv(self, nc_path: str) -> list[datetime]:
        """EUV dataset may store test_time / val_time / time variables."""
        try:
            import netCDF4 as nc4
            ds = nc4.Dataset(nc_path)
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
                    output = self._model(batch)

                # Cast to float32 before .numpy() — same bfloat16 rule as AR.
                spectrum = output[0].to(torch.float32).cpu().numpy()
                spectrum = np.clip(spectrum, 0.0, 1.0)

                ts_input  = _extract_ts(metadata, "timestamps_input",   0)
                ts_target = _extract_ts(metadata, "timestamps_targets",  0)

                return _build_result(spectrum, ts_input, ts_target)

        return _empty_euv_result()


def _build_result(spectrum: np.ndarray, ts_input: str, ts_target: str) -> dict[str, Any]:
    integrated    = float(spectrum.mean())
    soft_xray     = float(spectrum[_SXR_BINS].mean())   if _SXR_BINS.any()   else 0.0
    thermospheric = float(spectrum[_THERM_BINS].mean())  if _THERM_BINS.any() else 0.0
    he2           = float(spectrum[_HE2_BINS].mean())    if _HE2_BINS.any()   else 0.0

    indices = np.linspace(0, len(spectrum) - 1, 50, dtype=int)
    mini    = spectrum[indices].tolist()

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