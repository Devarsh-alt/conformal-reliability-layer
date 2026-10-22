from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# CONFIG
# =============================================================================

ROOT = Path(r"D:\reliability_layer")

RS05_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS05"
)

CASE_DIR = (
    RS05_DIR
    / "case_study"
)

INPUT_PATH = (
    CASE_DIR
    / "RS05_COVID_corrected_date_level_S6.csv"
)

OUTPUT_PATH = (
    CASE_DIR
    / "RS05_COVID_corrected_lead_time_S6.csv"
)

MODELS = [
    "EWMA",
    "GARCH-t",
    "HS",
    "QRF",
]

TARGETS = [
    "T1",
    "T2",
]

EVENT_ONSET = {
    "T1": pd.Timestamp("2020-03-16"),
    "T2": pd.Timestamp("2020-02-12"),
}

PRE_EVENT_WINDOW = 20

SYSTEM_RED_FRACTION = 0.25


# =============================================================================
# LOAD
# =============================================================================

if not INPUT_PATH.exists():
    raise FileNotFoundError(INPUT_PATH)

df = pd.read_csv(INPUT_PATH)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

required = [
    "target",
    "model",
    "forecast_date",
    "red_fraction",
]

missing = [
    c for c in required
    if c not in df.columns
]

if missing:
    raise RuntimeError(
        f"Missing columns: {missing}"
    )

df = df[
    df["target"].isin(TARGETS)
    &
    df["model"].isin(MODELS)
].copy()


# =============================================================================
# TRADING-DAY LEAD
# =============================================================================

def calculate_lead(
    dates: np.ndarray,
    alert_date: pd.Timestamp,
    event_date: pd.Timestamp,
) -> int:

    dates = np.sort(
        pd.to_datetime(dates)
    )

    alert_idx = np.searchsorted(
        dates,
        alert_date,
    )

    event_idx = np.searchsorted(
        dates,
        event_date,
    )

    return int(
        event_idx - alert_idx
    )


# =============================================================================
# FIND FIRST ALERT WITHIN EXACT 20-DAY WINDOW
# =============================================================================

results = []

for target in TARGETS:

    event_date = EVENT_ONSET[target]

    print()
    print("=" * 100)
    print(
        f"{target} | EVENT ONSET = {event_date.date()}"
    )
    print("=" * 100)

    for model in MODELS:

        x = df[
            (df["target"] == target)
            &
            (df["model"] == model)
        ].sort_values(
            "forecast_date"
        ).copy()

        dates = np.sort(
            x["forecast_date"].unique()
        )

        event_idx = np.searchsorted(
            dates,
            event_date,
        )

        if event_idx >= len(dates):
            print(
                f"{model}: event date unavailable"
            )
            continue

        # Exactly previous 20 trading dates + event date.
        start_idx = max(
            0,
            event_idx - PRE_EVENT_WINDOW,
        )

        window_dates = dates[
            start_idx:event_idx
        ]

        window = x[
            x["forecast_date"].isin(
                window_dates
            )
        ].copy()

        # ---------------------------------------------------------------------
        # System-level RED:
        # >=25% of stocks RED.
        # ---------------------------------------------------------------------

        alerts = window[
            window["red_fraction"]
            >= SYSTEM_RED_FRACTION
        ].sort_values(
            "forecast_date"
        )

        if alerts.empty:

            first_alert = pd.NaT
            lead = np.nan
            detected = False

        else:

            first_alert = pd.Timestamp(
                alerts.iloc[0]["forecast_date"]
            )

            lead = calculate_lead(
                dates,
                first_alert,
                event_date,
            )

            detected = True

        # ---------------------------------------------------------------------
        # Probability / band information on first alert.
        # ---------------------------------------------------------------------

        if detected:

            alert_row = alerts.iloc[0]

            alert_probability = float(
                alert_row["mean_p"]
            ) if "mean_p" in alerts.columns else np.nan

            alert_red_fraction = float(
                alert_row["red_fraction"]
            )

        else:

            alert_probability = np.nan
            alert_red_fraction = np.nan

        # ---------------------------------------------------------------------
        # State on final pre-event date.
        # ---------------------------------------------------------------------

        if not window.empty:

            final_row = window.iloc[-1]

            final_probability = float(
                final_row["mean_p"]
            )

            final_red_fraction = float(
                final_row["red_fraction"]
            )

            final_amber_fraction = float(
                final_row["amber_fraction"]
            )

            final_green_fraction = float(
                final_row["green_fraction"]
            )

        else:

            final_probability = np.nan
            final_red_fraction = np.nan
            final_amber_fraction = np.nan
            final_green_fraction = np.nan

        result = {
            "target": target,
            "model": model,
            "event_onset": event_date,
            "pre_event_window_trading_days":
                PRE_EVENT_WINDOW,
            "window_start":
                window["forecast_date"].min()
                if not window.empty
                else pd.NaT,
            "window_end":
                window["forecast_date"].max()
                if not window.empty
                else pd.NaT,
            "first_system_red_alert":
                first_alert,
            "lead_days":
                lead,
            "detected":
                detected,
            "alert_probability":
                alert_probability,
            "alert_red_fraction":
                alert_red_fraction,
            "final_pre_event_probability":
                final_probability,
            "final_pre_event_red_fraction":
                final_red_fraction,
            "final_pre_event_amber_fraction":
                final_amber_fraction,
            "final_pre_event_green_fraction":
                final_green_fraction,
        }

        results.append(
            result
        )

        print()
        print(
            f"MODEL: {model}"
        )
        print(
            f"20-day window: "
            f"{result['window_start']} "
            f"to "
            f"{result['window_end']}"
        )
        print(
            f"First system RED: "
            f"{result['first_system_red_alert']}"
        )
        print(
            f"Lead: "
            f"{result['lead_days']} trading days"
        )
        print(
            f"Detected: "
            f"{result['detected']}"
        )
        print(
            f"Alert probability: "
            f"{result['alert_probability']}"
        )
        print(
            f"Final pre-event probability: "
            f"{result['final_pre_event_probability']}"
        )


# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    [
        "target",
        "model",
    ]
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)

print()
print("=" * 100)
print("CORRECTED COVID LEAD-TIME SUMMARY")
print("=" * 100)

print(
    results_df[
        [
            "target",
            "model",
            "event_onset",
            "window_start",
            "window_end",
            "first_system_red_alert",
            "lead_days",
            "detected",
            "alert_probability",
            "final_pre_event_probability",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    f"Saved:\n{OUTPUT_PATH}"
)

print()
print(
    "Definition:"
)
print(
    "First alert = >=25% of stocks in RED "
    "within exactly 20 trading days before event onset."
)

print(
    "No retraining or recalibration performed."
)