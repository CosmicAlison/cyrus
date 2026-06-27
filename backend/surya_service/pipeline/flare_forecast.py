"""
Solar Flare Forecasting inference pipeline.

Task dir:  downstream_examples/solar_flare_forcasting/   (note typo in repo)
Weights:   assets/solar_flare_weights.pth
Model:     spectformer + LoRA (0.28% trainable params)

Output:
  {
    "prediction":    int,   # 0 = no flare, 1 = flare
    "time_input":    str,   # ISO timestamp of SDO observation
    "time_target":   str,   # ISO timestamp of prediction window
    "goes_class":    str,   # GOES class label from dataset (ground truth)
  }

Notes:
  - infer_single_sample is confirmed exposed in the flare tutorial imports
  - Dataset CSV has columns: timestep, max_goes_class, cumulative_index, label_max, label_cum
  - We locate the closest timestamp via CSV index then build a single-sample dataloader
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline, _utc

log = logging.getLogger(__name__)

CHECKPOINT = "assets/solar_flare_weights.pth"


class FlareForecastPipeline(BaseInferencePipeline):
    task_dir    = "solar_flare_forcasting"
    result_type = "flare_forecast"

    def _load_model(self) -> None:
        import torch
        from surya.utils.distributed import set_global_seed
        set_global_seed(42)

        self._config = self._load_config("config_infer.yaml")

        from infer import load_model
        checkpoint_path = str(self._task_path / CHECKPOINT)

        self._model = load_model(
            config=self._config,
            checkpoint_path=checkpoint_path,
            device=torch.device(self.device),
        )
        self._model.eval()
        log.info("[flare_forecast] model loaded from %s", checkpoint_path)

    def _load_timestamp_index(self) -> list[datetime]:
        """
        Load timestamps from the dataset CSV.
        CSV format: timestep, max_goes_class, cumulative_index, label_max, label_cum
        """
        import pandas as pd

        # Config may have test_path, test_file, or data_path
        data_cfg = self._config.get("data", {})
        csv_path = (
            data_cfg.get("test_path")
            or data_cfg.get("test_file")
            or data_cfg.get("data_path")
        )

        if not csv_path:
            # Search assets
            candidates = sorted((self._task_path / "assets").rglob("*.csv"))
            csv_path   = str(candidates[0]) if candidates else None

        if not csv_path:
            log.warning("[flare_forecast] no CSV index found")
            return []

        full = Path(csv_path) if Path(csv_path).is_absolute() \
               else self._task_path / csv_path

        return self._index_from_csv(str(full), col="timestep")

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        import torch
        from infer import get_dataloader, infer_single_sample

        loader, scalers = self._build_fixed_dataloader(sample_idx, data_type="test")

        all_preds = []
        all_meta  = []

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

                # Binary prediction: threshold at 0.5
                prob = float(torch.sigmoid(output).cpu().item()
                             if output.numel() == 1
                             else torch.sigmoid(output[:, 0]).cpu().mean().item())
                pred = 1 if prob >= 0.5 else 0
                all_preds.append((pred, prob))
                all_meta.append(metadata)

        if not all_preds:
            return {"prediction": 0, "probability": 0.0,
                    "time_input": "", "time_target": "", "goes_class": ""}

        pred, prob = all_preds[0]
        meta       = all_meta[0]

        ts_input  = _extract_ts(meta, "timestamps_input",   0)
        ts_target = _extract_ts(meta, "timestamps_targets",  0)
        goes      = _extract_str(meta, "max_goes_class", "goes_class", "label")

        return {
            "prediction":  pred,
            "probability": round(prob, 4),
            "time_input":  ts_input,
            "time_target": ts_target,
            "goes_class":  goes,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_ts(meta: dict, key: str, idx: int) -> str:
    try:
        arr = meta[key]
        val = arr[idx] if hasattr(arr, "__getitem__") else arr
        return str(np.datetime_as_string(np.asarray(val).ravel()[0], unit="m"))
    except Exception:
        return ""


def _extract_str(meta: dict, *keys: str) -> str:
    for k in keys:
        if k in meta:
            v = meta[k]
            return str(v[0] if hasattr(v, "__getitem__") else v)
    return ""