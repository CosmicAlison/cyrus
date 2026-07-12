"""
Solar Flare Forecasting inference pipeline.

Task dir:  downstream_examples/solar_flare_forcasting/   (note typo in repo)
Weights:   assets/solar_flare_weights.pth
Model:     spectformer + LoRA (0.28% trainable params)

"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.base import BaseInferencePipeline

log = logging.getLogger(__name__)

CHECKPOINT = "assets/solar_flare_weights.pth"


class FlareForecastPipeline(BaseInferencePipeline):
    task_dir    = "solar_flare_forcasting"
    result_type = "flare_forecast"

    dataset_class_name   = "SolarFlareDataset"
    extra_dataset_kwargs = {"flare_index_path": "flare_data_path"}
    index_path_key       = "valid_data_path"
    rollout_steps         = 0

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
        log.info("[flare_forecast] model loaded from %s", checkpoint_path)

    def _load_timestamp_index(self) -> list[datetime]:
        index_path = self._config["data"].get("valid_data_path")
        if not index_path:
            log.warning("[flare_forecast] no valid_data_path in config")
            return []
        return self._index_from_csv(index_path, col="timestep")

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        import torch

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