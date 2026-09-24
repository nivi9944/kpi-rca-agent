"""Charts for reports (matplotlib PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from metrics.metrics import metric_def, to_week  # noqa: E402
from tools.context import Investigation  # noqa: E402
from tools.drilldown import drill_table  # noqa: E402

BLUE, RED, GREY = "#2B6CB0", "#C53030", "#A0AEC0"


def make_chart(inv: Investigation, kind: str, metric: str, week: str, dimension: str | None = None,
               n_baseline: int = 4, weeks_back: int = 16, top_k: int = 8) -> dict:
    m = metric_def(metric)
    w = to_week(week)
    inv.chart_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.4), dpi=110)
    if kind == "series":
        s = inv.store.weekly(metric)
        s = s[(s["week"] <= w + pd.Timedelta(weeks=4)) & (s["week"] >= w - pd.Timedelta(weeks=weeks_back))]
        ax.plot(s["week"], s["value"], color=BLUE, marker="o", ms=3, lw=1.6)
        tgt = s[s["week"] == w]
        ax.scatter(tgt["week"], tgt["value"], color=RED, zorder=3, s=40, label=f"week {w:%Y-%m-%d}")
        ax.set_title(f"{m['label']} by week")
        ax.legend(frameon=False)
        fig.autofmt_xdate()
    elif kind == "segments":
        if not dimension:
            raise ValueError("dimension is required for a segments chart")
        t, _ = drill_table(inv.store, metric, dimension, w, n_baseline)
        t = t.head(top_k).iloc[::-1]
        colors = [RED if v < 0 else BLUE for v in t["delta"]]
        ax.barh(t["segment"].astype(str), t["contribution_pct"], color=colors)
        ax.axvline(0, color=GREY, lw=0.8)
        ax.set_xlabel("contribution to change (%)")
        ax.set_title(f"{m['label']}: contribution by {dimension}, week {w:%Y-%m-%d}")
    else:
        plt.close(fig)
        raise ValueError("kind must be 'series' or 'segments'")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    path = Path(inv.chart_dir) / f"{metric}_{kind}_{dimension or 'all'}_{w:%Y%m%d}_{len(inv.evidence) + 1}.png"
    fig.savefig(path)
    plt.close(fig)
    return {"chart_path": path.as_posix(), "kind": kind, "metric": metric, "week": w}
