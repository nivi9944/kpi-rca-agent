"""Analysis-engine settings that changed between versions, kept switchable so earlier results stay reproducible.

  v2: segment-scan threshold |hist_z| >= 4.25 (8.2% natural alarm rate), Welch t-test for continuous rates
  v3: threshold 4.0 (11.2% natural alarm rate, results/calibration.json; an accepted trade-off for recall) and,
      for money metrics (AOV, avg_item_price), Yuen's trimmed-mean test (20% trimming), robust to the few
      big-ticket orders that dominate a week's mean

Select with the KPI_ENGINE environment variable or `set_engine`; the default is v3.
"""
from __future__ import annotations

import os

ENGINES = {
    "v2": {"hist_z": 4.25, "money_rate_test": "welch"},
    "v3": {"hist_z": 4.0, "money_rate_test": "yuen"},
}
TRIM = 0.2
_current = {"name": os.getenv("KPI_ENGINE", "v3")}


def set_engine(name: str) -> None:
    if name not in ENGINES:
        raise ValueError(f"unknown engine {name!r}; known: {sorted(ENGINES)}")
    _current["name"] = name


def engine_name() -> str:
    return _current["name"]


def hist_z() -> float:
    return ENGINES[_current["name"]]["hist_z"]


def money_rate_test() -> str:
    return ENGINES[_current["name"]]["money_rate_test"]
