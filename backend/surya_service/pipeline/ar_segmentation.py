"""
AR Segmentation inference pipeline.

Task dir:  downstream_examples/ar_segmentation/
Weights:   assets/ar_segmentation_weights.pth (nasa-ibm-ai4science/ar_segmentation_surya)
Model:     HelioSpectformer2D or UNet (from models.py in task dir)

Two call modes per tick:
  ARSegmentationPipeline.infer(solar_dt)         → current AR state
  ARSegmentationPipeline.infer(flare_target_dt)  → predicted AR state at flare time

Output (mask statistics only — PNG discarded):
  {
    "timestamp_input":  str,
    "timestamp_target": str,
    "active_region_count": int,
    "centroids": [{"x": float, "y": float, "area_frac": float, "disk_proximity": float}],
    "total_area_frac":  float,   # fraction of solar disk covered
    "dominant_region":  {"x": float, "y": float, "area_frac": float} | None
  }

Key implementation notes (from actual infer.py source):
  - forecast_hat shape after model: (1, 13, H, W) expanded, then sigmoid applied
  - The mask channel we care about is dim 0 (first channel post-sigmoid)
  - Centroids extracted via scipy.ndimage.label on thresholded sigmoid mask
  - Coordinates normalised to [-1, 1] solar disk space (0,0 = disk center)
  - metadata["timestamps_input"] / ["timestamps_targets"] are numpy datetime64 arrays
"""

import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline, _utc

log = logging.getLogger(__name__)

CHECKPOINT   = "assets/ar_segmentation_weights.pth"
SIGMOID_THRESHOLD = 0.5   # binarise the segmentation mask


class ARSegmentationPipeline(BaseInferencePipeline):
    task_dir    = "ar_segmentation"
    result_type = "ar_segmentation"

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")

        # Import model factory from task directory (added to sys.path by base)
        from infer import load_model
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = load_model(
            config=self._task_path / "config_infer.yaml",   # infer.py may take path
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[ar_segmentation] model loaded from %s", checkpoint_path)

    def _load_timestamp_index(self) -> list[datetime]:
        """
        Load timestamps from the HelioNetCDF index file.
        config["data"]["valid_data_path"] points to a NetCDF index.
        """
        index_path = self._config["data"].get("valid_data_path", "")
        full_path  = Path(index_path) if Path(index_path).is_absolute() \
                     else self._task_path / index_path

        if full_path.suffix in (".nc", ".netcdf", ".netCDF"):
            return self._index_from_netcdf(str(full_path))
        elif full_path.suffix == ".csv":
            return self._index_from_csv(str(full_path))
        else:
            log.warning("[ar_segmentation] Unknown index format: %s", full_path)
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
                    forecast_hat = forecast_hat[:, 0]  # (B, C, H, W)

                # Expand to 13 channels as in the original infer.py
                forecast_hat = forecast_hat.expand(1, 13, -1, -1)
                forecast_hat = torch.sigmoid(forecast_hat)

                # Use channel 0 as the segmentation mask
                mask_tensor = forecast_hat[0, 0].cpu().numpy()  # (H, W)

                # Extract timestamps from metadata
                ts_input  = _parse_metadata_ts(metadata, "timestamps_input",  0)
                ts_target = _parse_metadata_ts(metadata, "timestamps_targets", 0)

                # Extract mask statistics — this is all we need, no PNG saved
                stats = _extract_mask_stats(mask_tensor, threshold=SIGMOID_THRESHOLD)

                return {
                    "timestamp_input":     ts_input,
                    "timestamp_target":    ts_target,
                    **stats,
                }

        return _empty_ar_result()


# ── Mask statistics extraction ────────────────────────────────────────────────

def _extract_mask_stats(mask: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    """
    Extract active region statistics from a sigmoid probability mask.

    Converts continuous sigmoid output → binary mask → connected components.
    Coordinates are normalised to [-1, 1] with (0,0) at disk center.

    Returns:
        active_region_count, centroids list, total_area_frac, dominant_region
    """
    from scipy.ndimage import label, center_of_mass

    H, W = mask.shape
    binary = (mask > threshold).astype(np.uint8)

    # Label connected components
    labeled, n_regions = label(binary)

    if n_regions == 0:
        return _empty_ar_stats()

    centroids = []
    for region_id in range(1, n_regions + 1):
        region_mask  = labeled == region_id
        area_pixels  = int(region_mask.sum())
        area_frac    = area_pixels / (H * W)

        # Skip tiny noise regions (< 0.01% of image)
        if area_frac < 0.0001:
            continue

        cy, cx = center_of_mass(region_mask)

        # Normalise to [-1, 1] — (0,0) = disk center
        x_norm = (cx / W) * 2 - 1
        y_norm = (cy / H) * 2 - 1

        # Disk proximity: 0 = center, 1 = limb
        disk_proximity = min(1.0, (x_norm**2 + y_norm**2) ** 0.5)

        centroids.append({
            "x":              round(x_norm, 4),
            "y":              round(y_norm, 4),
            "area_frac":      round(area_frac, 6),
            "disk_proximity": round(disk_proximity, 4),
        })

    if not centroids:
        return _empty_ar_stats()

    # Sort by area descending
    centroids.sort(key=lambda c: c["area_frac"], reverse=True)
    total_area = sum(c["area_frac"] for c in centroids)

    return {
        "active_region_count": len(centroids),
        "centroids":           centroids,
        "total_area_frac":     round(total_area, 6),
        "dominant_region":     centroids[0],
    }


def _empty_ar_stats() -> dict[str, Any]:
    return {
        "active_region_count": 0,
        "centroids":           [],
        "total_area_frac":     0.0,
        "dominant_region":     None,
    }


def _empty_ar_result() -> dict[str, Any]:
    return {
        "timestamp_input":  "",
        "timestamp_target": "",
        **_empty_ar_stats(),
    }


def _parse_metadata_ts(metadata: dict, key: str, idx: int) -> str:
    """
    Extract a formatted timestamp from metadata dict.
    metadata[key] is a list of numpy datetime64 arrays (from format_metadata pattern).
    """
    try:
        arr   = metadata[key]
        entry = arr[idx] if hasattr(arr, "__getitem__") else arr
        # numpy datetime64 → string
        return str(np.datetime_as_string(
            np.asarray(entry).ravel()[0], unit="m"
        ))
    except Exception:
        return ""