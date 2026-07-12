"""
AR Segmentation inference pipeline.

Task dir:  downstream_examples/ar_segmentation/
Weights:   assets/ar_segmentation_weights.pth (nasa-ibm-ai4science/ar_segmentation_surya)
Model:     HelioSpectformer2D or UNet (from models.py in task dir)

"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline

log = logging.getLogger(__name__)

CHECKPOINT         = "assets/ar_segmentation_weights.pth"
SIGMOID_THRESHOLD  = 0.5


class ARSegmentationPipeline(BaseInferencePipeline):
    task_dir    = "ar_segmentation"
    result_type = "ar_segmentation"

    dataset_class_name = None   # explicit: use HelioNetCDFDataset directly
    index_path_key      = "valid_data_path"
    rollout_steps        = 1    # matches NASA's own infer.py::get_dataloader

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")

        infer_module = self._import_task_module()
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = infer_module.load_model(
            config=self._config,          # dict, not a Path — was the bug
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[ar_segmentation] model loaded from %s", checkpoint_path)

    def _load_timestamp_index(self) -> list[datetime]:
        index_path = self._config["data"].get("valid_data_path", "")
        if not index_path:
            log.warning("[ar_segmentation] no valid_data_path in config")
            return []

        suffix = Path(index_path).suffix
        if suffix in (".nc", ".netcdf", ".netCDF"):
            return self._index_from_netcdf(index_path)
        elif suffix == ".csv":
            return self._index_from_csv(index_path, col="timestep")
        else:
            log.warning("[ar_segmentation] unknown index format: %s", index_path)
            return []

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        import torch

        loader, scalers = self._build_fixed_dataloader(sample_idx, data_type="valid")

        with torch.no_grad():
            for batch, metadata in loader:
                if self.device != "cpu":
                    batch = {k: v.to(self.device) for k, v in batch.items()}

                with torch.amp.autocast(
                    device_type="cuda" if "cuda" in self.device else "cpu",
                    dtype=self._config["dtype"],
                    enabled="cuda" in self.device,
                ):
                    forecast_hat = self._model(batch)

                if forecast_hat.ndim == 5:
                    forecast_hat = forecast_hat[:, 0]

                forecast_hat = forecast_hat.expand(1, 13, -1, -1)
                forecast_hat = torch.sigmoid(forecast_hat)

                # Cast to float32 BEFORE .cpu().numpy() — bfloat16 tensors
                # cannot convert to numpy directly.
                mask_tensor = forecast_hat[0, 0].to(torch.float32).cpu().numpy()

                ts_input  = _parse_metadata_ts(metadata, "timestamps_input",  0)
                ts_target = _parse_metadata_ts(metadata, "timestamps_targets", 0)

                stats = _extract_mask_stats(mask_tensor, threshold=SIGMOID_THRESHOLD)

                return {
                    "timestamp_input":  ts_input,
                    "timestamp_target": ts_target,
                    **stats,
                }

        return _empty_ar_result()


def _extract_mask_stats(mask: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    from scipy.ndimage import label, center_of_mass

    H, W = mask.shape
    binary = (mask > threshold).astype(np.uint8)
    labeled, n_regions = label(binary)

    if n_regions == 0:
        return _empty_ar_stats()

    centroids = []
    for region_id in range(1, n_regions + 1):
        region_mask = labeled == region_id
        area_pixels = int(region_mask.sum())
        area_frac   = area_pixels / (H * W)

        if area_frac < 0.0001:
            continue

        cy, cx = center_of_mass(region_mask)
        x_norm = (cx / W) * 2 - 1
        y_norm = (cy / H) * 2 - 1
        disk_proximity = min(1.0, (x_norm**2 + y_norm**2) ** 0.5)

        centroids.append({
            "x": round(x_norm, 4),
            "y": round(y_norm, 4),
            "area_frac": round(area_frac, 6),
            "disk_proximity": round(disk_proximity, 4),
        })

    if not centroids:
        return _empty_ar_stats()

    centroids.sort(key=lambda c: c["area_frac"], reverse=True)
    total_area = sum(c["area_frac"] for c in centroids)

    return {
        "active_region_count": len(centroids),
        "centroids": centroids,
        "total_area_frac": round(total_area, 6),
        "dominant_region": centroids[0],
    }


def _empty_ar_stats() -> dict[str, Any]:
    return {
        "active_region_count": 0,
        "centroids": [],
        "total_area_frac": 0.0,
        "dominant_region": None,
    }


def _empty_ar_result() -> dict[str, Any]:
    return {
        "timestamp_input": "",
        "timestamp_target": "",
        **_empty_ar_stats(),
    }


def _parse_metadata_ts(metadata: dict, key: str, idx: int) -> str:
    try:
        arr   = metadata[key]
        entry = arr[idx] if hasattr(arr, "__getitem__") else arr
        return str(np.datetime_as_string(
            np.asarray(entry).ravel()[0], unit="m"
        ))
    except Exception:
        return ""