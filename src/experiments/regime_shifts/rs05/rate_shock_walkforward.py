from __future__ import annotations

"""
RS05 — WALK-FORWARD RATE-SHOCK EX-ANTE RELIABILITY CASE STUDY
==============================================================

Purpose
-------
Evaluate the already-defined RS05 S6 reliability monitor around a
pre-specified 2022 US monetary-policy rate-shock anchor.

IMPORTANT
---------
This script is an EVENT-STUDY layer only. It does NOT train or recalibrate
the monitor.

The predictions MUST come from rs05_walkforward.py, so every forecast used
here is genuinely out-of-sample / walk-forward.

Primary target:
    T1 = future 10-trading-day deterioration

Primary event anchor:
    2022-03-16 = first Federal Reserve rate hike of the 2022 tightening
    cycle.

The event date is fixed independently of RS05 predictions.

System RED:
    >= 25% of the 24 stocks are RED.

Pre-event warning window:
    exactly 20 available trading dates before the event.

The script reports:
    - first system RED alert
    - trading-day lead
    - probability at first RED
    - maximum pre-event probability
    - final pre-event probability
    - realized deterioration after onset
    - detection / miss

It also verifies that the input contains the expected S6/logit
walk-forward predictions.
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd


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

DEFAULT_INPUT = RS05_DIR / "RS05_oos_predictions_H10_main.csv.gz"
FALLBACK_INPUT = RS05_DIR / "RS05_oos_predictions_H10_calibration.csv.gz"

OUTPUT_DIR = RS05_DIR / "case_study"

MODELS = ["EWMA", "GARCH-t", "HS", "QRF"]

TARGET = "T1"
SPEC = "S6"
ALGO = "logit"

# Fixed independently of monitor output.
# 2022-03-16 = first Federal Reserve rate hike of the 2022 tightening cycle.
EVENT_ONSET = pd.Timestamp("2022-03-16")

PRE_EVENT_TRADING_DAYS = 20
POST_EVENT_CALENDAR_DAYS = 120

SYSTEM_RED_FRACTION = 0.25


# =============================================================================
# ARGUMENTS
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Walk-forward OOS prediction file. Defaults to H10_main, "
             "then falls back to H10_calibration."
    )

    parser.add_argument(
        "--event-date",
        type=str,
        default=EVENT_ONSET.strftime("%Y-%m-%d"),
        help="Fixed event anchor, YYYY-MM-DD."
    )

    parser.add_argument(
        "--post-days",
        type=int,
        default=POST_EVENT_CALENDAR_DAYS,
        help="Calendar days shown after event onset."
    )

    return parser.parse_args()


# =============================================================================
# LOAD
# =============================================================================

def choose_input(user_input):
    if user_input:
        path = Path(user_input)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    if DEFAULT_INPUT.exists():
        return DEFAULT_INPUT

    if FALLBACK_INPUT.exists():
        return FALLBACK_INPUT

    raise FileNotFoundError(
        "No walk-forward prediction file found.\n"
        f"Tried:\n  {DEFAULT_INPUT}\n  {FALLBACK_INPUT}"
    )


def load_predictions(path):
    print("=" * 110)
    print("RS05 — WALK-FORWARD RATE-SHOCK EX-ANTE RELIABILITY CASE STUDY")
    print("=" * 110)

    print(f"Input: {path}")
    print(f"Target: {TARGET}")
    print(f"Spec: {SPEC}")
    print(f"Algorithm: {ALGO}")
    print(f"Event onset: {EVENT_ONSET.date()}")
    print(f"System RED threshold: {SYSTEM_RED_FRACTION:.0%}")
    print(f"Pre-event window: {PRE_EVENT_TRADING_DAYS} trading days")

    df = pd.read_csv(path, compression="gzip", low_memory=False)

    required = [
        "target",
        "model",
        "algo",
        "spec",
        "forecast_date",
        "ticker",
        "y",
        "p_cal",
        "thr05",
        "thr10",
        "thr20",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n" + "\n".join(missing)
        )

    # Normalize.
    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"], errors="coerce"
    )

    df["target"] = (
        df["target"].astype(str).str.strip().str.upper()
    )

    df["model"] = (
        df["model"].astype(str).str.strip()
    )

    df["algo"] = (
        df["algo"].astype(str).str.strip().str.lower()
    )

    df["spec"] = (
        df["spec"].astype(str).str.strip().str.upper()
    )

    df["ticker"] = (
        df["ticker"].astype(str).str.strip().str.upper()
    )

    for c in ["y", "p_cal", "thr05", "thr10", "thr20"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[
        (df["target"] == TARGET)
        & (df["model"].isin(MODELS))
        & (df["spec"] == SPEC)
        & (df["algo"] == ALGO)
    ].dropna(
        subset=[
            "forecast_date",
            "ticker",
            "y",
            "p_cal",
            "thr05",
            "thr10",
            "thr20",
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No S6/logit/T1 rows remain after filtering."
        )

    # Integrity checks.
    df["y"] = df["y"].astype(int)

    if not set(df["y"].unique()).issubset({0, 1}):
        raise RuntimeError("Non-binary target values found.")

    if not np.isfinite(df["p_cal"]).all():
        raise RuntimeError("Non-finite p_cal values found.")

    if ((df["p_cal"] < 0) | (df["p_cal"] > 1)).any():
        raise RuntimeError("p_cal outside [0, 1].")

    if (
        (df["thr05"] < df["thr10"])
        | (df["thr10"] < df["thr20"])
    ).any():
        raise RuntimeError(
            "Frozen threshold ordering violated."
        )

    duplicates = int(
        df.duplicated(
            subset=[
                "target",
                "model",
                "forecast_date",
                "ticker",
            ]
        ).sum()
    )

    if duplicates:
        raise RuntimeError(
            f"Duplicate target/model/date/ticker rows: {duplicates}"
        )

    # We expect all four models.
    actual_models = sorted(df["model"].unique())
    if actual_models != sorted(MODELS):
        raise RuntimeError(
            f"Expected models {sorted(MODELS)}, found {actual_models}"
        )

    print()
    print(f"Rows after filtering: {len(df):,}")
    print(f"Date range: {df['forecast_date'].min().date()} "
          f"to {df['forecast_date'].max().date()}")
    print(f"Tickers: {df['ticker'].nunique()}")
    print(f"Models: {actual_models}")

    # Explicitly verify that the event is inside the available OOS period.
    if not (
        df["forecast_date"].min()
        <= EVENT_ONSET
        <= df["forecast_date"].max()
    ):
        raise RuntimeError(
            f"Event onset {EVENT_ONSET.date()} is outside the OOS "
            f"prediction range."
        )

    # Helpful warning if this is only a narrow smoke run.
    n_dates = df["forecast_date"].nunique()
    print(f"Unique OOS forecast dates: {n_dates}")

    return df


# =============================================================================
# BAND ASSIGNMENT
# =============================================================================

def assign_bands(df):
    x = df.copy()

    x["trust_band"] = np.select(
        [
            x["p_cal"] >= x["thr10"],
            x["p_cal"] >= x["thr20"],
        ],
        [
            "RED",
            "AMBER",
        ],
        default="GREEN",
    )

    x["critical_5pct"] = x["p_cal"] >= x["thr05"]

    return x


# =============================================================================
# DATE-LEVEL AGGREGATION
# =============================================================================

def build_date_level_panel(df):
    x = assign_bands(df)

    rows = []

    for (target, model, date), g in x.groupby(
        ["target", "model", "forecast_date"],
        sort=True,
        observed=True,
    ):
        rows.append(
            {
                "target": target,
                "model": model,
                "forecast_date": date,
                "n_tickers": int(len(g)),
                "mean_p": float(g["p_cal"].mean()),
                "median_p": float(g["p_cal"].median()),
                "realized_fraction": float(g["y"].mean()),
                "green_fraction": float(
                    (g["trust_band"] == "GREEN").mean()
                ),
                "amber_fraction": float(
                    (g["trust_band"] == "AMBER").mean()
                ),
                "red_fraction": float(
                    (g["trust_band"] == "RED").mean()
                ),
                "critical_fraction": float(
                    g["critical_5pct"].mean()
                ),
                "mean_thr20": float(g["thr20"].mean()),
                "mean_thr10": float(g["thr10"].mean()),
                "mean_thr05": float(g["thr05"].mean()),
            }
        )

    panel = pd.DataFrame(rows)

    if panel.empty:
        raise RuntimeError("Date-level panel is empty.")

    panel = panel.sort_values(
        ["target", "model", "forecast_date"]
    ).reset_index(drop=True)

    panel["system_red_alert"] = (
        panel["red_fraction"] >= SYSTEM_RED_FRACTION
    )

    return panel


# =============================================================================
# TRADING-DAY LEAD
# =============================================================================

def trading_day_difference(dates, earlier, later):
    dates = np.sort(pd.to_datetime(dates))

    earlier = pd.Timestamp(earlier)
    later = pd.Timestamp(later)

    earlier_idx = np.searchsorted(dates, earlier)
    later_idx = np.searchsorted(dates, later)

    return int(later_idx - earlier_idx)


# =============================================================================
# EVENT ANALYSIS
# =============================================================================

def analyze_model(panel, model, event_date):
    x = panel[
        (panel["target"] == TARGET)
        & (panel["model"] == model)
    ].sort_values("forecast_date").copy()

    dates = np.sort(x["forecast_date"].unique())

    event_idx = np.searchsorted(dates, event_date)

    if event_idx >= len(dates):
        return {
            "model": model,
            "event_onset": event_date,
            "detected": False,
            "reason": "event outside available dates",
        }

    # Exactly the previous 20 available trading dates.
    start_idx = max(0, event_idx - PRE_EVENT_TRADING_DAYS)
    pre_dates = dates[start_idx:event_idx]

    pre = x[
        x["forecast_date"].isin(pre_dates)
    ].copy()

    post_end = event_date + pd.Timedelta(
        days=POST_EVENT_CALENDAR_DAYS
    )

    post = x[
        (x["forecast_date"] >= event_date)
        & (x["forecast_date"] <= post_end)
    ].copy()

    alerts = pre[
        pre["system_red_alert"]
    ].sort_values("forecast_date")

    if alerts.empty:
        first_alert = pd.NaT
        lead = np.nan
        detected = False
        alert_probability = np.nan
        alert_red_fraction = np.nan
    else:
        first_alert = pd.Timestamp(
            alerts.iloc[0]["forecast_date"]
        )

        lead = trading_day_difference(
            dates,
            first_alert,
            event_date,
        )

        detected = True

        alert_probability = float(
            alerts.iloc[0]["mean_p"]
        )

        alert_red_fraction = float(
            alerts.iloc[0]["red_fraction"]
        )

    # Final pre-event state.
    if pre.empty:
        final_pre = {}
    else:
        final_row = pre.iloc[-1]

        final_pre = {
            "final_pre_event_probability":
                float(final_row["mean_p"]),
            "final_pre_event_red_fraction":
                float(final_row["red_fraction"]),
            "final_pre_event_amber_fraction":
                float(final_row["amber_fraction"]),
            "final_pre_event_green_fraction":
                float(final_row["green_fraction"]),
        }

    # Maximum pre-event probability.
    max_pre_probability = (
        float(pre["mean_p"].max())
        if not pre.empty else np.nan
    )

    # Outcome after event.
    if not post.empty:
        event_day_rows = post[
            post["forecast_date"] >= event_date
        ]

        event_day_realized = float(
            event_day_rows.iloc[0]["realized_fraction"]
        ) if not event_day_rows.empty else np.nan

        max_post_realized = float(
            post["realized_fraction"].max()
        )
    else:
        event_day_realized = np.nan
        max_post_realized = np.nan

    result = {
        "target": TARGET,
        "model": model,
        "event_onset": event_date,
        "pre_event_window_trading_days":
            PRE_EVENT_TRADING_DAYS,
        "window_start":
            pre["forecast_date"].min() if not pre.empty else pd.NaT,
        "window_end":
            pre["forecast_date"].max() if not pre.empty else pd.NaT,
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
        "max_pre_event_probability":
            max_pre_probability,
        "event_day_realized_fraction":
            event_day_realized,
        "max_post_event_realized_fraction":
            max_post_realized,
    }

    result.update(final_pre)

    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    global EVENT_ONSET

    args = parse_args()
    EVENT_ONSET = pd.Timestamp(args.event_date)

    input_path = choose_input(args.input)

    df = load_predictions(input_path)
    panel = build_date_level_panel(df)

    print()
    print("=" * 110)
    print("RATE-SHOCK EVENT ANALYSIS")
    print("=" * 110)

    results = []

    for model in MODELS:
        result = analyze_model(
            panel,
            model,
            EVENT_ONSET,
        )

        results.append(result)

        print()
        print(f"MODEL: {model}")
        print(f"Window: {result.get('window_start')} "
              f"to {result.get('window_end')}")
        print(f"First system RED: "
              f"{result.get('first_system_red_alert')}")
        print(f"Lead: "
              f"{result.get('lead_days')} trading days")
        print(f"Detected: "
              f"{result.get('detected')}")
        print(f"Alert probability: "
              f"{result.get('alert_probability')}")
        print(f"Maximum pre-event probability: "
              f"{result.get('max_pre_event_probability')}")
        print(f"Event-day realized deterioration: "
              f"{result.get('event_day_realized_fraction')}")
        print(f"Maximum post-event deterioration: "
              f"{result.get('max_post_event_realized_fraction')}")

    results_df = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "RS05_RATE_SHOCK_T1_walkforward_lead_time_S6.csv"
    )

    panel_path = (
        OUTPUT_DIR
        / "RS05_RATE_SHOCK_T1_walkforward_date_level_S6.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    panel[
        panel["forecast_date"]
        <= EVENT_ONSET + pd.Timedelta(days=args.post_days)
    ].to_csv(
        panel_path,
        index=False,
    )

    print()
    print("=" * 110)
    print("RATE-SHOCK SUMMARY")
    print("=" * 110)

    summary_cols = [
        "model",
        "event_onset",
        "first_system_red_alert",
        "lead_days",
        "detected",
        "alert_probability",
        "max_pre_event_probability",
        "event_day_realized_fraction",
        "max_post_event_realized_fraction",
    ]

    print(
        results_df[summary_cols].to_string(index=False)
    )

    print()
    print(f"Saved lead-time results:\n{output_path}")
    print(f"Saved date-level panel:\n{panel_path}")

    print()
    print("WALK-FORWARD INTEGRITY:")
    print("  This script does not retrain, recalibrate, or tune RS05.")
    print("  It consumes predictions generated by rs05_walkforward.py.")
    print("  The rate-shock event date is fixed independently of RS05 output.")
    print("  The pre-event warning uses only dates strictly before the event.")
    print("  Realized deterioration is evaluated after the event.")


if __name__ == "__main__":
    main()
