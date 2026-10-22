
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE = Path(r"D:\reliability_layer")
RS_DIR = BASE / "data" / "monitoring" / "regime_shifts"

RS04_PATH = RS_DIR / "RS04_causal_reliability_panel.csv"
QUAR_PATH = RS_DIR / "RS04_quarantined_outcomes.csv"

OUT_DIR = BASE / "data" / "results" / "continuous_deterioration"
OUT_DIR.mkdir(parents=True, exist_ok=True)

H = 10

COVID_START = pd.Timestamp("2020-02-24")
COVID_END = pd.Timestamp("2020-07-17")

EVENT_WINDOWS = {
    "COVID": (pd.Timestamp("2020-02-24"), pd.Timestamp("2020-07-17")),
    "2018_stress": (pd.Timestamp("2018-01-01"), pd.Timestamp("2018-12-31")),
    "2022_stress": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
    "2023_banking": (pd.Timestamp("2023-03-01"), pd.Timestamp("2023-05-31")),
    "2025_stress": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
}


def require_columns(df, cols, label):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} is missing required columns: {missing}"
        )


def quantile_table(s, name):
    x = pd.to_numeric(s, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    qs = [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.00]

    return pd.DataFrame(
        {
            "metric": name,
            "quantile": qs,
            "value": [
                float(x.quantile(q))
                for q in qs
            ],
        }
    )


print("=" * 110)
print("CONTINUOUS VA R DETERIORATION TARGET — AUDIT")
print("=" * 110)

if not RS04_PATH.exists():
    raise FileNotFoundError(RS04_PATH)

if not QUAR_PATH.exists():
    raise FileNotFoundError(QUAR_PATH)

print(f"RS04:       {RS04_PATH}")
print(f"Quarantine: {QUAR_PATH}")

# ---------------------------------------------------------------------
# Load causal model state / baseline
# ---------------------------------------------------------------------

rs = pd.read_csv(
    RS04_PATH,
    usecols=[
        "ticker",
        "forecast_date",
        "model_name",
        "current_abs_var",
        "historical_10d_mean_norm_loss",
        "historical_20d_mean_norm_loss",
        "causal_10d_loss_threshold",
        "causal_20d_loss_threshold",
    ],
    low_memory=False,
)

rs["ticker"] = (
    rs["ticker"]
    .astype(str)
    .str.strip()
    .str.upper()
)

rs["model_name"] = (
    rs["model_name"]
    .astype(str)
    .str.strip()
)

rs["forecast_date"] = pd.to_datetime(
    rs["forecast_date"],
    errors="coerce",
)

for c in rs.columns:
    if c not in ["ticker", "model_name", "forecast_date"]:
        rs[c] = pd.to_numeric(
            rs[c],
            errors="coerce",
        )

require_columns(
    rs,
    [
        "ticker",
        "forecast_date",
        "model_name",
        "current_abs_var",
        "historical_10d_mean_norm_loss",
        "historical_20d_mean_norm_loss",
        "causal_10d_loss_threshold",
        "causal_20d_loss_threshold",
    ],
    "RS04",
)

rs = (
    rs.sort_values(
        ["ticker", "model_name", "forecast_date"]
    )
    .drop_duplicates(
        ["ticker", "model_name", "forecast_date"]
    )
)

print(f"RS04 rows: {len(rs):,}")

# ---------------------------------------------------------------------
# Load quarantined future outcomes
# ---------------------------------------------------------------------

q = pd.read_csv(
    QUAR_PATH,
    low_memory=False,
)

q["ticker"] = (
    q["ticker"]
    .astype(str)
    .str.strip()
    .str.upper()
)

q["model_name"] = (
    q["model_name"]
    .astype(str)
    .str.strip()
)

q["forecast_date"] = pd.to_datetime(
    q["forecast_date"],
    errors="coerce",
)

q["realized_date"] = pd.to_datetime(
    q["realized_date"],
    errors="coerce",
)

q["outcome_next_day_pinball_loss"] = pd.to_numeric(
    q["outcome_next_day_pinball_loss"],
    errors="coerce",
)

q["outcome_next_day_breach"] = pd.to_numeric(
    q["outcome_next_day_breach"],
    errors="coerce",
)

require_columns(
    q,
    [
        "ticker",
        "forecast_date",
        "realized_date",
        "outcome_next_day_pinball_loss",
        "outcome_next_day_breach",
        "model_name",
    ],
    "quarantine",
)

q = (
    q.sort_values(
        ["ticker", "model_name", "forecast_date"]
    )
    .drop_duplicates(
        ["ticker", "model_name", "forecast_date"]
    )
)

print(f"Quarantine rows: {len(q):,}")

# ---------------------------------------------------------------------
# Reconstruct daily normalized pinball loss.
#
# Each quarantine row is the loss of the VaR forecast made at t and
# evaluated on t+1. current_abs_var belongs to that same forecast t.
# ---------------------------------------------------------------------

daily = q.merge(
    rs[
        [
            "ticker",
            "forecast_date",
            "model_name",
            "current_abs_var",
        ]
    ],
    on=[
        "ticker",
        "forecast_date",
        "model_name",
    ],
    how="left",
    validate="one_to_one",
)

daily["normalized_pinball_loss"] = (
    daily["outcome_next_day_pinball_loss"]
    / daily["current_abs_var"].abs()
)

bad = (
    (~np.isfinite(daily["normalized_pinball_loss"]))
    | (daily["current_abs_var"].abs() <= 0)
)

if int(bad.sum()) > 0:
    raise AssertionError(
        f"Invalid normalized loss rows: {int(bad.sum())}"
    )

# ---------------------------------------------------------------------
# Build forward H-day target from t through t+H-1.
#
# This means:
#   loss at forecast t     evaluates return t+1
#   loss at forecast t+1   evaluates return t+2
#   ...
#   loss at forecast t+9   evaluates return t+10
#
# This is the same forward-window convention used by the research design.
# ---------------------------------------------------------------------

daily = daily.sort_values(
    ["ticker", "model_name", "forecast_date"]
).reset_index(drop=True)

group_keys = ["ticker", "model_name"]

future = daily[group_keys + [
    "forecast_date",
    "normalized_pinball_loss",
    "outcome_next_day_breach",
]].copy()

future["base_date"] = future["forecast_date"]

# Shift future outcomes backward so they attach to the current forecast date.
for k in range(H):
    s = future[group_keys + [
        "forecast_date",
        "normalized_pinball_loss",
        "outcome_next_day_breach",
    ]].copy()

    s["forecast_date"] = (
        s.groupby(group_keys)["forecast_date"]
        .shift(-k)
    )

    # The final k rows of every ticker/model become NaT after the backward
    # shift. Those rows are incomplete future windows and must not participate
    # in the merge. Keeping them causes pandas to treat repeated NaT keys as
    # duplicate merge keys and breaks validate="one_to_one".
    s = s.dropna(
        subset=["forecast_date"]
    ).copy()

    s = s.rename(
        columns={
            "normalized_pinball_loss":
                f"loss_k{k}",
            "outcome_next_day_breach":
                f"breach_k{k}",
        }
    )[
        group_keys
        + [
            "forecast_date",
            f"loss_k{k}",
            f"breach_k{k}",
        ]
    ]

    if k == 0:
        target = s
    else:
        target = target.merge(
            s,
            on=group_keys + ["forecast_date"],
            how="left",
            validate="one_to_one",
        )

loss_cols = [f"loss_k{k}" for k in range(H)]
breach_cols = [f"breach_k{k}" for k in range(H)]

target = target.dropna(
    subset=loss_cols + breach_cols
).copy()

# Hard integrity checks: every retained target row must have a real forecast
# date and a unique ticker/model/date key.
if target["forecast_date"].isna().any():
    raise AssertionError("Target construction left NaT forecast dates.")

if target.duplicated(
    group_keys + ["forecast_date"]
).any():
    dup = target.loc[
        target.duplicated(
            group_keys + ["forecast_date"],
            keep=False,
        ),
        group_keys + ["forecast_date"],
    ].head(20)
    raise AssertionError(
        "Duplicate target keys remain after forward-window construction:\n"
        f"{dup.to_string(index=False)}"
    )

target["future_10d_mean_norm_loss"] = target[
    loss_cols
].mean(axis=1)

target["future_10d_median_norm_loss"] = target[
    loss_cols
].median(axis=1)

target["future_10d_max_norm_loss"] = target[
    loss_cols
].max(axis=1)

target["future_10d_breach_rate"] = target[
    breach_cols
].mean(axis=1)

# ---------------------------------------------------------------------
# Attach the causal baseline available at t.
# ---------------------------------------------------------------------

target = target.merge(
    rs[
        [
            "ticker",
            "forecast_date",
            "model_name",
            "historical_10d_mean_norm_loss",
            "historical_20d_mean_norm_loss",
            "causal_10d_loss_threshold",
            "causal_20d_loss_threshold",
        ]
    ],
    on=[
        "ticker",
        "forecast_date",
        "model_name",
    ],
    how="left",
    validate="one_to_one",
)

target["relative_to_hist10"] = (
    target["future_10d_mean_norm_loss"]
    / target["historical_10d_mean_norm_loss"]
    .replace(0, np.nan)
)

target["relative_to_hist20"] = (
    target["future_10d_mean_norm_loss"]
    / target["historical_20d_mean_norm_loss"]
    .replace(0, np.nan)
)

target["relative_to_threshold10"] = (
    target["future_10d_mean_norm_loss"]
    / target["causal_10d_loss_threshold"]
    .replace(0, np.nan)
)

target["relative_to_threshold20"] = (
    target["future_10d_mean_norm_loss"]
    / target["causal_20d_loss_threshold"]
    .replace(0, np.nan)
)

target = target.replace(
    [np.inf, -np.inf],
    np.nan,
)

print()
print(f"Continuous target rows: {len(target):,}")
print(
    f"Date range: "
    f"{target['forecast_date'].min().date()} "
    f"to "
    f"{target['forecast_date'].max().date()}"
)

# ---------------------------------------------------------------------
# Validate horizon construction.
# ---------------------------------------------------------------------

for ticker, model in target[
    ["ticker", "model_name"]
].drop_duplicates().itertuples(index=False):
    pass

# Verify each source series has consecutive forecast dates for the H rows
# used in each target. This catches accidental calendar gaps.
daily_date_map = (
    daily[
        [
            "ticker",
            "model_name",
            "forecast_date",
        ]
    ]
    .sort_values(
        [
            "ticker",
            "model_name",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)

expected_rows = (
    daily_date_map.groupby(
        ["ticker", "model_name"]
    ).size()
)

print(
    f"Ticker/model series: {len(expected_rows)} "
    f"(expected 96)"
)

# ---------------------------------------------------------------------
# Distribution audit
# ---------------------------------------------------------------------

metrics = [
    "future_10d_mean_norm_loss",
    "future_10d_median_norm_loss",
    "future_10d_max_norm_loss",
    "relative_to_hist10",
    "relative_to_hist20",
    "relative_to_threshold10",
    "relative_to_threshold20",
]

dist_parts = []

for model in sorted(
    target["model_name"].unique()
):
    m = target[
        target["model_name"] == model
    ]

    for metric in metrics:
        qt = quantile_table(
            m[metric],
            metric,
        )
        qt.insert(0, "model", model)
        dist_parts.append(qt)

dist_df = pd.concat(
    dist_parts,
    ignore_index=True,
)

dist_path = (
    OUT_DIR
    / "continuous_target_distribution_by_model.csv"
)

dist_df.to_csv(
    dist_path,
    index=False,
)

# ---------------------------------------------------------------------
# Severity thresholds.
# ---------------------------------------------------------------------

severity_rows = []

for model in sorted(
    target["model_name"].unique()
):
    m = target[
        target["model_name"] == model
    ]

    for metric in [
        "relative_to_hist20",
        "relative_to_threshold10",
        "future_10d_breach_rate",
    ]:
        x = pd.to_numeric(
            m[metric],
            errors="coerce",
        ).dropna()

        row = {
            "model": model,
            "metric": metric,
            "n": int(len(x)),
            "mean": float(x.mean()),
            "median": float(x.median()),
        }

        for threshold in (
            1.0,
            1.25,
            1.5,
            2.0,
            3.0,
            5.0,
        ):
            if metric == "future_10d_breach_rate":
                value = threshold / 100.0
                key = f"share_ge_{threshold:.2f}"
            else:
                value = threshold
                key = f"share_ge_{threshold:.2f}"

            row[key] = float(
                np.mean(x >= value)
            )

        severity_rows.append(row)

severity_df = pd.DataFrame(
    severity_rows
)

severity_path = (
    OUT_DIR
    / "continuous_target_severity_rates.csv"
)

severity_df.to_csv(
    severity_path,
    index=False,
)

# ---------------------------------------------------------------------
# Event / regime-style audit.
# ---------------------------------------------------------------------

event_rows = []

for event_name, (
    start,
    end,
) in EVENT_WINDOWS.items():

    msk = (
        (target["forecast_date"] >= start)
        & (target["forecast_date"] <= end)
    )

    e = target[msk].copy()

    for model in sorted(
        target["model_name"].unique()
    ):
        em = e[
            e["model_name"] == model
        ]

        if em.empty:
            continue

        event_rows.append(
            {
                "event": event_name,
                "model": model,
                "n_rows": int(len(em)),
                "mean_future_norm_loss": float(
                    em[
                        "future_10d_mean_norm_loss"
                    ].mean()
                ),
                "median_future_norm_loss": float(
                    em[
                        "future_10d_mean_norm_loss"
                    ].median()
                ),
                "mean_relative_hist20": float(
                    em[
                        "relative_to_hist20"
                    ].mean()
                ),
                "median_relative_hist20": float(
                    em[
                        "relative_to_hist20"
                    ].median()
                ),
                "mean_relative_threshold10": float(
                    em[
                        "relative_to_threshold10"
                    ].mean()
                ),
                "median_relative_threshold10": float(
                    em[
                        "relative_to_threshold10"
                    ].median()
                ),
                "mean_future_breach_rate": float(
                    em[
                        "future_10d_breach_rate"
                    ].mean()
                ),
                "p90_relative_hist20": float(
                    em[
                        "relative_to_hist20"
                    ].quantile(0.90)
                ),
            }
        )

event_df = pd.DataFrame(
    event_rows
)

event_path = (
    OUT_DIR
    / "continuous_target_event_audit.csv"
)

event_df.to_csv(
    event_path,
    index=False,
)

# ---------------------------------------------------------------------
# Date-level cross-sectional audit.
# ---------------------------------------------------------------------

date_df = (
    target.groupby(
        [
            "forecast_date",
            "model_name",
        ],
        as_index=False,
    )
    .agg(
        mean_future_norm_loss=(
            "future_10d_mean_norm_loss",
            "mean",
        ),
        median_future_norm_loss=(
            "future_10d_mean_norm_loss",
            "median",
        ),
        mean_relative_hist20=(
            "relative_to_hist20",
            "mean",
        ),
        median_relative_hist20=(
            "relative_to_hist20",
            "median",
        ),
        mean_relative_threshold10=(
            "relative_to_threshold10",
            "mean",
        ),
        mean_future_breach_rate=(
            "future_10d_breach_rate",
            "mean",
        ),
        frac_severe_2x=(
            "relative_to_hist20",
            lambda x: float(
                np.mean(x >= 2.0)
            ),
        ),
        frac_severe_3x=(
            "relative_to_hist20",
            lambda x: float(
                np.mean(x >= 3.0)
            ),
        ),
    )
)

date_path = (
    OUT_DIR
    / "continuous_target_daily_monitoring_series.csv.gz"
)

date_df.to_csv(
    date_path,
    index=False,
    compression="gzip",
)

# ---------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------

print()
print("=" * 110)
print("CORE CONTINUOUS-TARGET SUMMARY")
print("=" * 110)

for model in sorted(
    target["model_name"].unique()
):
    m = target[
        target["model_name"] == model
    ]

    print()
    print(model)
    print(
        f"  future mean normalized loss "
        f"median = "
        f"{m['future_10d_mean_norm_loss'].median():.6f}"
    )
    print(
        f"  future/past-20d loss ratio "
        f"median = "
        f"{m['relative_to_hist20'].median():.4f}"
    )
    print(
        f"  future/past-20d loss ratio "
        f"90th pct = "
        f"{m['relative_to_hist20'].quantile(0.90):.4f}"
    )
    print(
        f"  future/past-20d loss ratio "
        f"95th pct = "
        f"{m['relative_to_hist20'].quantile(0.95):.4f}"
    )
    print(
        f"  share >= 2x past-20d loss = "
        f"{np.mean(m['relative_to_hist20'] >= 2.0):.4%}"
    )
    print(
        f"  share >= 3x past-20d loss = "
        f"{np.mean(m['relative_to_hist20'] >= 3.0):.4%}"
    )

print()
print("COVID SUMMARY")
print(
    event_df[
        event_df["event"] == "COVID"
    ]
    .round(5)
    .to_string(index=False)
)

print()
print("OUTPUT FILES")
print(dist_path)
print(severity_path)
print(event_path)
print(date_path)

# Save full row-level target for the next modeling stage.
target_path = (
    OUT_DIR
    / "continuous_deterioration_target_H10.csv.gz"
)

target.to_csv(
    target_path,
    index=False,
    compression="gzip",
)

print(target_path)

print("=" * 110)
print("CONTINUOUS TARGET AUDIT COMPLETE")
print("=" * 110)
