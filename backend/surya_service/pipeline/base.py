"""
Abstract base for all four Surya downstream inference pipelines.

Key facts learned from actual infer.py source (ar_segmentation):
  - Dataset is HelioNetCDFDataset, indexed by a NetCDF/CSV index file
  - Dataloader uses random_ids = torch.randperm — we override with fixed seed
    and a deterministic index derived from the solar clock timestamp
  - metadata dict contains timestamps_input / timestamps_targets as numpy datetime64
  - Models are LoRA-wrapped via PEFT on top of the 366M base weights
  - All tasks share the same config loading pattern:
      config = yaml.safe_load(config_infer.yaml)
      config["data"]["scalers"] = yaml.safe_load(scalers_path)
      config["dtype"] = torch.float32 | float16 | bfloat16
"""

import logging
import sys
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SURYA_REPO = Path("/surya")


class BaseInferencePipeline(ABC):
    task_dir:    str   # subdir under downstream_examples/
    result_type: str   # human label for logging

    def __init__(self, device: str = "cpu") -> None:
        self.device      = device
        self._task_path  = SURYA_REPO / "downstream_examples" / self.task_dir
        self._model      = None
        self._config     = None
        self._scalers    = None
        self._ts_index   = None   # sorted list[datetime] — loaded once

        # Make the task directory importable ("from infer import ...")
        task_str = str(self._task_path)
        if task_str not in sys.path:
            sys.path.insert(0, task_str)

        log.info("[%s] initialising (device=%s)", self.result_type, device)
        self._load_model()
        log.info("[%s] ready", self.result_type)

    # ── Abstract interface ────────────────────────────────────────────────────

    @abstractmethod
    def _load_model(self) -> None:
        """Load config, scalers, model weights into self._config/_scalers/_model."""
        ...

    @abstractmethod
    def _load_timestamp_index(self) -> list[datetime]:
        """
        Return sorted list of all timestamps available in the test/valid split.
        Used for closest-sample lookup — called once then cached.
        """
        ...

    @abstractmethod
    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        """Run inference on sample at position sample_idx. Return raw result dict."""
        ...

    # ── Public entry point ────────────────────────────────────────────────────

    def infer(self, solar_dt: datetime) -> dict[str, Any]:
        """
        Find the benchmark sample closest to solar_dt and run inference.
        Always returns a dict with at least:
            _pipeline, _requested_solar_dt, _matched_solar_dt
        """
        if self._ts_index is None:
            self._ts_index = self._load_timestamp_index()
            log.info(
                "[%s] timestamp index: %d entries (%s → %s)",
                self.result_type, len(self._ts_index),
                self._ts_index[0].date()  if self._ts_index else "?",
                self._ts_index[-1].date() if self._ts_index else "?",
            )

        idx      = self._closest_index(solar_dt)
        matched  = self._ts_index[idx] if self._ts_index else solar_dt

        log.debug("[%s] solar_dt=%s → idx=%d matched=%s",
                  self.result_type, solar_dt.isoformat(), idx, matched.isoformat())

        result = self._run_at_index(idx)
        result["_pipeline"]           = self.result_type
        result["_requested_solar_dt"] = solar_dt.isoformat()
        result["_matched_solar_dt"]   = matched.isoformat()
        return result

    # ── Shared helpers ────────────────────────────────────────────────────────

    def _closest_index(self, target: datetime) -> int:
        """Binary search on sorted timestamp index."""
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)
        index = self._ts_index
        if not index:
            return 0

        lo, hi = 0, len(index) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            ts  = _utc(index[mid])
            if ts < target:
                lo = mid + 1
            else:
                hi = mid

        # Compare both neighbours
        if lo > 0:
            if abs((_utc(index[lo - 1]) - target).total_seconds()) < \
               abs((_utc(index[lo])     - target).total_seconds()):
                return lo - 1
        return lo

    def _load_config(self, filename: str = "config_infer.yaml") -> dict:
        """Load YAML config + scalers, resolve dtype to torch type."""
        import torch
        import yaml

        path = self._task_path / filename
        if not path.exists():
            path = self._task_path / "config.yaml"

        config = yaml.safe_load(open(path))

        pretrained = (
            config.get("pretrained_path")
            or config.get("model", {}).get("pretrained_path")
        )
        
        pretrained = str((self._task_path / pretrained).resolve())
        
        config["pretrained_path"] = pretrained
        
        if "model" in config:
            config["model"]["pretrained_path"] = pretrained

        scalers_path = config.get("data", {}).get("scalers_path", "")
        sp = self._task_path / scalers_path
        if sp.exists():
            config["data"]["scalers"] = yaml.safe_load(open(sp))

        dtype_map = {
            "float16":  torch.float16,
            "bfloat16": torch.bfloat16,
            "float32":  torch.float32,
        }
        config["dtype"] = dtype_map.get(config.get("dtype", "float32"), torch.float32)
        return config

    def _build_fixed_dataloader(self, sample_idx: int, data_type: str = "test"):
        """
        Build a dataloader that returns exactly one sample at a known index.
        Overrides the random_ids logic in the upstream get_dataloader.
        """
        import torch
        from torch.utils.data import DataLoader, Subset
        from surya.utils.data import build_scalers
        from surya.datasets.helio import HelioNetCDFDataset

        scalers = build_scalers(info=self._config["data"]["scalers"])
        self._scalers = scalers

        index_key = "valid_data_path" if data_type == "valid" else "valid_data_path"
        index_path = self._config["data"].get(
            f"{data_type}_data_path",
            self._config["data"].get("valid_data_path"),
        )

        dataset = HelioNetCDFDataset(
            sdo_data_root_path=self._config["data"]["sdo_data_root_path"],
            index_path=index_path,
            time_delta_input_minutes=self._config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=self._config["data"]["time_delta_target_minutes"],
            n_input_timestamps=len(self._config["data"]["time_delta_input_minutes"]),
            rollout_steps=1,
            channels=self._config["data"]["channels"],
            scalers=scalers,
            phase=data_type,
        )

        # Clamp index to dataset length
        idx = min(sample_idx, len(dataset) - 1)

        try:
            from finetune import custom_collate_fn
            collate = custom_collate_fn
        except ImportError:
            collate = None

        loader = DataLoader(
            dataset=Subset(dataset, [idx]),
            batch_size=1,
            num_workers=0,
            pin_memory=False,
            shuffle=False,
            collate_fn=collate,
        )
        return loader, scalers

    def _index_from_netcdf(self, nc_path: str,
                            time_var: str = "time") -> list[datetime]:
        """Extract sorted datetime list from a NetCDF index file."""
        import netCDF4 as nc
        import numpy as np

        ds        = nc.Dataset(nc_path)
        raw_times = nc.num2date(ds[time_var][:], ds[time_var].units)
        ds.close()
        return sorted([
            datetime(t.year, t.month, t.day, t.hour, t.minute, t.second,
                     tzinfo=timezone.utc)
            for t in raw_times
        ])

    def _index_from_csv(self, csv_path: str,
                         col: str = "timestep") -> list[datetime]:
        """Extract sorted datetime list from a CSV index file."""
        import pandas as pd
        df = pd.read_csv(csv_path, parse_dates=[col])
        return sorted([
            t.to_pydatetime().replace(tzinfo=timezone.utc)
            for t in df[col]
        ])


def _utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)