"""
Maps real wall-clock time to a position inside a fixed 10-year solar
data window (2012-01-01 → 2022-01-01).

The window was chosen to include:
  - Solar Cycle 24 maximum (2013-2014) — frequent X/M-class flares
  - 2017-09 events (X8.2, X9.3 — strongest flares in a decade)
  - Solar Cycle 25 onset (2019-2022) — quieter baseline for contrast

Usage:
    from solar_clock import SolarClock
    clock = SolarClock()
    solar_now = clock.now()          # current solar timestamp
    solar_at  = clock.at(real_dt)   # solar timestamp for any real datetime

Override for demos:
    Set SOLAR_TIME_OVERRIDE=2017-09-06T12:00:00 in env to pin to a
    specific solar event (e.g. the X9.3 flare) for a dramatic demo.
"""

import logging
import os
from datetime import datetime, timedelta, timezone

log = logging.getLogger(__name__)


SOLAR_WINDOW_START = datetime(2012, 1, 1, tzinfo=timezone.utc)
SOLAR_WINDOW_END   = datetime(2022, 1, 1, tzinfo=timezone.utc)
SOLAR_WINDOW_SECS  = (SOLAR_WINDOW_END - SOLAR_WINDOW_START).total_seconds()

# The real-world anchor: deployment start date.
# Everything is measured relative to this.
# Override via DEPLOY_ANCHOR env var (ISO format).
_DEFAULT_ANCHOR = datetime(2025, 1, 1, tzinfo=timezone.utc)


class SolarClock:
    """
    Single source of truth for solar time across the entire surya_service.

    Real time advances 1:1 with solar time — after 10 years of real time
    the clock wraps back to 2012-01-01. In practice the hackathon runs
    for 3 weeks, placing solar time in early-to-mid 2012 (active period).
    """

    def __init__(self) -> None:
        anchor_str = os.environ.get("DEPLOY_ANCHOR")
        if anchor_str:
            self._anchor = datetime.fromisoformat(anchor_str).replace(tzinfo=timezone.utc)
        else:
            self._anchor = _DEFAULT_ANCHOR

        # Demo override — pins clock to a specific solar timestamp
        self._override_str = os.environ.get("SOLAR_TIME_OVERRIDE")
        if self._override_str:
            log.warning(
                "SOLAR_TIME_OVERRIDE active: solar time pinned to %s",
                self._override_str,
            )

        log.info(
            "SolarClock initialised | window: %s → %s | anchor: %s",
            SOLAR_WINDOW_START.date(),
            SOLAR_WINDOW_END.date(),
            self._anchor.date(),
        )

    def now(self) -> datetime:
        """Current solar timestamp."""
        if self._override_str:
            return datetime.fromisoformat(self._override_str).replace(tzinfo=timezone.utc)
        return self.at(datetime.now(timezone.utc))

    def at(self, real_dt: datetime) -> datetime:
        """
        Convert any real datetime to the corresponding solar timestamp.
        Wraps around the 10-year window automatically.
        """
        if real_dt.tzinfo is None:
            real_dt = real_dt.replace(tzinfo=timezone.utc)

        elapsed_secs = (real_dt - self._anchor).total_seconds()
        # Handle negative elapsed (real_dt before anchor) by normalising
        position_secs = elapsed_secs % SOLAR_WINDOW_SECS
        return SOLAR_WINDOW_START + timedelta(seconds=position_secs)

    def advance(self, solar_dt: datetime, real_delta: timedelta) -> datetime:
        """
        Advance a solar timestamp by a real-world timedelta,
        wrapping around the window if needed.
        """
        offset = (solar_dt - SOLAR_WINDOW_START).total_seconds()
        new_offset = (offset + real_delta.total_seconds()) % SOLAR_WINDOW_SECS
        return SOLAR_WINDOW_START + timedelta(seconds=new_offset)

    def info(self) -> dict:
        solar_now = self.now()
        return {
            "solar_now":          solar_now.isoformat(),
            "window_start":       SOLAR_WINDOW_START.isoformat(),
            "window_end":         SOLAR_WINDOW_END.isoformat(),
            "override_active":    self._override_str is not None,
            "override_value":     self._override_str,
            "anchor":             self._anchor.isoformat(),
        }


# Module-level singleton
clock = SolarClock()