"""
Abstract base for all four Surya downstream inference pipelines.
"""

import logging
import os
import sys
import importlib.util
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# NOTE: verify this matches your actual deployment path — during debugging
# this was /workspace/cyrus/backend/surya_service/Surya, not /surya.
# Override via env var so it doesn't need another code edit if it moves again.
SURYA_REPO = Path(os.environ.get("SURYA_REPO_PATH", "/surya"))


class BaseInferencePipeline(ABC):
    task_dir:    str    # subdir under downstream_examples/
    result_type: str    # human label for logging

    # ---- Per-pipeline overrides — set these on each subclass ----
    # Name of the task-specific Dataset subclass to use (e.g. "SolarFlareDataset").
    # Leave None to use the generic HelioNetCDFDataset directly (this is what
    # ar_segmentation's own reference infer.py does).
    dataset_class_name: str | None = None

    # Extra constructor kwargs the task-specific Dataset subclass needs,
    # mapped from {constructor_kwarg_name: config["data"] key}.
    # e.g. flare needs {"flare_index_path": "flare_data_path"}
    extra_dataset_kwargs: dict[str, str] = {}

    # Explicit config["data"] key to use for index_path. If None, falls back
    # to the generic "{data_type}_data_path" / "valid_data_path" chain.
    # EUV needed this set explicitly to "infer_data_path" — its config
    # doesn't use "valid_data_path" at all.
    index_path_key: str | None = None

    # rollout_steps passed to the Dataset constructor. This is NOT universal —
    # flare needs 0 (single-step +60min forecast), ar_segmentation's own
    # reference infer.py hardcodes 1. Get this wrong and filter_valid_indices()
    # silently requires timesteps that don't exist, producing an EMPTY
    # valid_indices list with no obvious error (IndexError way downstream).
    rollout_steps: int = 0

    def __init__(self, device: str = "cpu") -> None:
        self.device      = device
        self._task_path  = SURYA_REPO / "downstream_examples" / self.task_dir
        self._model      = None
        self._config     = None
        self._scalers    = None
        self._ts_index   = None   # sorted list[datetime] — loaded once

        task_str = str(self._task_path)
        if task_str not in sys.path:
            sys.path.insert(0, task_str)

        log.info("[%s] initialising (device=%s)", self.result_type, device)
        self._load_model()
        log.info("[%s] ready", self.result_type)

    # ── Abstract interface ────────────────────────────────────────────────

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

    def _run_at_index(self, sample_idx: int) -> dict[str, Any]:
        """Run inference on sample at position sample_idx. Return raw result dict."""
        raise NotImplementedError

    # ── Public entry point ────────────────────────────────────────────────

    def infer(self, solar_dt: datetime) -> dict[str, Any]:
        """
        Find the benchmark sample closest to solar_dt and run inference.
        Always returns a dict with at least:
            _pipeline, _requested_solar_dt, _matched_solar_dt
        """
        if self._ts_index is None:
            self._ts_index = self._load_timestamp_index()
            log.info(
                "[%s] timestamp index: %d entries (%s -> %s)",
                self.result_type, len(self._ts_index),
                self._ts_index[0].date()  if self._ts_index else "?",
                self._ts_index[-1].date() if self._ts_index else "?",
            )

        idx     = self._closest_index(solar_dt)
        matched = self._ts_index[idx] if self._ts_index else solar_dt

        log.debug("[%s] solar_dt=%s -> idx=%d matched=%s",
                  self.result_type, solar_dt.isoformat(), idx, matched.isoformat())

        result = self._run_at_index(idx)
        result["_pipeline"]           = self.result_type
        result["_requested_solar_dt"] = solar_dt.isoformat()
        result["_matched_solar_dt"]   = matched.isoformat()
        return result

    # ── Shared helpers ───────────────────────────────────────────────────

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

        if lo > 0:
            if abs((_utc(index[lo - 1]) - target).total_seconds()) < \
               abs((_utc(index[lo])     - target).total_seconds()):
                return lo - 1
        return lo

    def _import_task_module(self, filename: str = "infer.py"):
        """
        Load this task's infer.py (or another file) with its sibling modules
        (models.py, dataset.py, finetune.py) pre-registered in sys.modules
        under their literal bare names FIRST. This guarantees any bare
        `from dataset import X` / `from finetune import Y` inside infer.py —
        or inside our own code afterward — resolves to THIS task's files,
        regardless of what another pipeline loaded earlier and regardless
        of sys.path search order.
        """
        task_path = str(self._task_path)
        if task_path not in sys.path:
            sys.path.insert(0, task_path)

        for name in ("models", "dataset", "finetune"):
            file_path = self._task_path / f"{name}.py"
            if not file_path.exists():
                sys.modules.pop(name, None)
                continue
            spec = importlib.util.spec_from_file_location(name, file_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)

        cache_key = f"{self.task_dir}_{Path(filename).stem}"
        if cache_key in sys.modules:
            del sys.modules[cache_key]

        mod_path = self._task_path / filename
        spec = importlib.util.spec_from_file_location(cache_key, mod_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[cache_key] = module
        spec.loader.exec_module(module)
        return module

    def _load_config(self, filename: str = "config_infer.yaml") -> dict:
        """Load YAML config, resolve ALL relative paths, load scalers, resolve dtype."""
        import torch
        import yaml

        path = self._task_path / filename
        if not path.exists():
            path = self._task_path / "config.yaml"

        with open(path, "r") as f:
            config = yaml.safe_load(f)

        # ---- pretrained_path can live at top level OR under "model" ----
        pretrained = (
            config.get("pretrained_path")
            or config.get("model", {}).get("pretrained_path")
        )
        if pretrained:
            pretrained = str((self._task_path / pretrained).resolve())
            config["pretrained_path"] = pretrained
            if "model" in config:
                config["model"]["pretrained_path"] = pretrained

        # ---- Resolve EVERY relative path under data: ----
        # Deliberately NOT gated on a "./" prefix — configs are inconsistent
        # about this (e.g. "./assets/x" vs "assets/x" vs "../../data/x").
        data = config.get("data", {})
        for key, value in list(data.items()):
            if isinstance(value, str) and value and not Path(value).is_absolute():
                data[key] = str((self._task_path / value).resolve())

        # ---- Load scalers into memory ----
        scalers_path = data.get("scalers_path")
        if scalers_path:
            sp = Path(scalers_path)
            if sp.exists():
                with open(sp, "r") as f:
                    config["data"]["scalers"] = yaml.safe_load(f)
                config["data"]["scalers_path"] = str(sp)
            else:
                log.warning("[%s] scalers file not found: %s", self.result_type, sp)

        # ---- dtype string -> torch dtype ----
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

        scalers = build_scalers(info=self._config["data"]["scalers"])
        self._scalers = scalers

        # ---- resolve index path ----
        if self.index_path_key:
            index_path = self._config["data"].get(self.index_path_key)
        else:
            index_path = self._config["data"].get(
                f"{data_type}_data_path",
                self._config["data"].get("valid_data_path"),
            )
        if not index_path:
            raise ValueError(
                f"[{self.result_type}] could not resolve an index path from "
                f"config['data'] — set `index_path_key` on this pipeline class"
            )

        # ---- resolve dataset class via isolated per-task module import ----
        self._import_task_module()
        if self.dataset_class_name:
            DatasetClass = getattr(sys.modules["dataset"], self.dataset_class_name)
        else:
            from surya.datasets.helio import HelioNetCDFDataset
            DatasetClass = HelioNetCDFDataset

        extra_kwargs = {
            kwarg_name: self._config["data"][config_key]
            for kwarg_name, config_key in (self.extra_dataset_kwargs or {}).items()
        }

        dataset = DatasetClass(
            sdo_data_root_path=self._config["data"]["sdo_data_root_path"],
            index_path=index_path,
            time_delta_input_minutes=self._config["data"]["time_delta_input_minutes"],
            time_delta_target_minutes=self._config["data"]["time_delta_target_minutes"],
            n_input_timestamps=len(self._config["data"]["time_delta_input_minutes"]),
            rollout_steps=self.rollout_steps,
            channels=self._config["data"]["channels"],
            scalers=scalers,
            phase=data_type,
            **extra_kwargs,
        )

        # ---- clamp against the REAL valid-sample count, not len(dataset) ----
        # len(dataset) was observed to disagree with len(dataset.valid_indices)
        # in at least one case — clamping against len(dataset) let an
        # out-of-range index slip through and crash deeper in __getitem__.
        valid_indices = getattr(dataset, "valid_indices", None)
        valid_len = len(valid_indices) if valid_indices is not None else len(dataset)
        if valid_len == 0:
            raise ValueError(
                f"[{self.result_type}] dataset has 0 valid samples — check "
                f"rollout_steps ({self.rollout_steps}) and index density "
                f"against time_delta_input_minutes/time_delta_target_minutes"
            )
        idx = max(0, min(sample_idx, valid_len - 1))

        collate = getattr(sys.modules.get("finetune"), "custom_collate_fn", None)

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