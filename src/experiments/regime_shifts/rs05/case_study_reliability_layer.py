from __future__ import annotations

"""
RS05 — CORRECTED COVID EX-ANTE RELIABILITY CASE STUDY
======================================================

Purpose
-------
Demonstrate the behavior of the already-generated OOS S6 reliability layer
around the COVID episode.

Two targets are handled separately because they have different natural
episode anchors:

T1
--
Future 10-day deterioration.

Fixed descriptive COVID stress onset:
    2020-03-16

T2
--
Future 10-day coverage failure.

Outcome-defined COVID failure onset:
    2020-02-12

System-level RED alert
----------------------
A system alert occurs only when:

    >= 25% of the 24 stocks are in RED

RED itself is defined using the frozen RS05 thr10 threshold.

IMPORTANT
---------
This script does NOT:

    - retrain RS05
    - recalibrate probabilities
    - estimate new thresholds
    - use future outcomes as predictors
    - choose the event onset from monitor probabilities
    - choose the alert date by looking for the largest probability spike

The event onset is fixed beforehand.

The pre-event panels show only probabilities and reliability-band states
available before the event onset.

Realized outcomes are shown separately after the event onset solely for
evaluation.

OUTPUTS
-------
case_study/
    RS05_COVID_corrected_date_level_S6.csv
    RS05_COVID_corrected_summary_S6.csv
    RS05_COVID_T1_corrected_summary_S6.csv
    RS05_COVID_T2_corrected_summary_S6.csv

    RS05_COVID_CORRECTED_<MODEL>_T1_S6.png
    RS05_COVID_CORRECTED_<MODEL>_T2_S6.png
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"D:\reliability_layer"
)

RS05_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS05"
)

INPUT_PATH = (
    RS05_DIR
    / "RS05_oos_predictions_H10_calibration.csv.gz"
)

OUTPUT_DIR = (
    RS05_DIR
    / "case_study"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# RS05 CONFIGURATION
# =============================================================================

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

SPEC = "S6"
ALGO = "logit"

HEADLINE_START = pd.Timestamp(
    "2020-01-01"
)

# -------------------------------------------------------------------------
# CASE-STUDY EVENT ANCHORS
# -------------------------------------------------------------------------

EVENT_ONSET = {
    "T1": pd.Timestamp("2020-03-16"),
    "T2": pd.Timestamp("2020-02-12"),
}

# Contextual transition start used for COVID T1.
T1_TRANSITION_START = pd.Timestamp(
    "2020-02-01"
)

# For T2, use the preceding month as visible warning history.
T2_WARNING_START = pd.Timestamp(
    "2020-01-01"
)

# -------------------------------------------------------------------------
# SYSTEM ALERT RULE
# -------------------------------------------------------------------------

# A system-level warning requires at least 25% of stocks to be RED.
SYSTEM_RED_FRACTION = 0.25

# -------------------------------------------------------------------------
# PLOT WINDOWS
# -------------------------------------------------------------------------

# Amount of time shown BEFORE event onset.
PRE_EVENT_CALENDAR_DAYS = 90

# Amount of time shown AFTER event onset.
POST_EVENT_CALENDAR_DAYS = 120


# =============================================================================
# OUTPUT FILES
# =============================================================================

DATE_LEVEL_OUT = (
    OUTPUT_DIR
    / "RS05_COVID_corrected_date_level_S6.csv"
)

SUMMARY_OUT = (
    OUTPUT_DIR
    / "RS05_COVID_corrected_summary_S6.csv"
)

T1_SUMMARY_OUT = (
    OUTPUT_DIR
    / "RS05_COVID_T1_corrected_summary_S6.csv"
)

T2_SUMMARY_OUT = (
    OUTPUT_DIR
    / "RS05_COVID_T2_corrected_summary_S6.csv"
)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_predictions() -> pd.DataFrame:

    print("=" * 110)
    print(
        "RS05 — CORRECTED COVID EX-ANTE RELIABILITY CASE STUDY"
    )
    print("=" * 110)

    print(
        f"Input:\n{INPUT_PATH}"
    )

    print()
    print(
        "Target event anchors:"
    )
    print(
        f"  T1 deterioration onset: "
        f"{EVENT_ONSET['T1'].date()}"
    )
    print(
        f"  T2 coverage-failure onset: "
        f"{EVENT_ONSET['T2'].date()}"
    )

    print()
    print(
        "System RED rule:"
    )
    print(
        f"  >= {SYSTEM_RED_FRACTION:.0%} "
        "of stocks in RED"
    )

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{INPUT_PATH}"
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        df = pd.read_csv(
            INPUT_PATH,
            compression="gzip",
            low_memory=False,
        )

    print()
    print(
        f"Raw rows: {len(df):,}"
    )

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

    missing = [
        c
        for c in required
        if c not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns:\n"
            + "\n".join(
                f"  - {c}"
                for c in missing
            )
        )

    # -------------------------------------------------------------------------
    # Normalize
    # -------------------------------------------------------------------------

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )

    df["target"] = (
        df["target"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["model"] = (
        df["model"]
        .astype(str)
        .str.strip()
    )

    df["algo"] = (
        df["algo"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["spec"] = (
        df["spec"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df["y"] = pd.to_numeric(
        df["y"],
        errors="coerce",
    )

    df["p_cal"] = pd.to_numeric(
        df["p_cal"],
        errors="coerce",
    )

    for c in [
        "thr05",
        "thr10",
        "thr20",
    ]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # -------------------------------------------------------------------------
    # Filter exact intended S6 OOS sample.
    # -------------------------------------------------------------------------

    df = df[
        (df["forecast_date"] >= HEADLINE_START)
        &
        df["target"].isin(TARGETS)
        &
        df["model"].isin(MODELS)
        &
        (df["spec"] == SPEC)
        &
        (df["algo"] == ALGO)
    ].dropna(
        subset=[
            "forecast_date",
            "ticker",
            "target",
            "model",
            "spec",
            "y",
            "p_cal",
            "thr05",
            "thr10",
            "thr20",
        ]
    ).copy()

    if df.empty:
        raise RuntimeError(
            "No rows remain after filtering."
        )

    # -------------------------------------------------------------------------
    # Integrity
    # -------------------------------------------------------------------------

    df["y"] = df["y"].astype(int)

    if not set(
        df["y"].unique()
    ).issubset({0, 1}):
        raise RuntimeError(
            "Non-binary y values detected."
        )

    p = df["p_cal"].to_numpy(
        dtype=float
    )

    if not np.isfinite(p).all():
        raise RuntimeError(
            "Non-finite p_cal detected."
        )

    if (
        (p < 0)
        |
        (p > 1)
    ).any():
        raise RuntimeError(
            "p_cal outside [0,1]."
        )

    bad_threshold_order = (
        (df["thr05"] < df["thr10"])
        |
        (df["thr10"] < df["thr20"])
    )

    if bad_threshold_order.any():
        raise RuntimeError(
            "Frozen threshold ordering violated."
        )

    # Duplicate ticker/date check.
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
            f"Duplicate target/model/date/ticker rows: "
            f"{duplicates}"
        )

    print(
        f"Filtered rows: {len(df):,}"
    )

    print(
        f"Date range: "
        f"{df['forecast_date'].min().date()} "
        f"to "
        f"{df['forecast_date'].max().date()}"
    )

    print(
        f"Tickers: {df['ticker'].nunique()}"
    )

    print(
        f"Models: {sorted(df['model'].unique())}"
    )

    print(
        f"Targets: {sorted(df['target'].unique())}"
    )

    return df


# =============================================================================
# BAND ASSIGNMENT
# =============================================================================

def assign_bands(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = df.copy()

    # Frozen RS05 thresholds.
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

    x["critical_5pct"] = (
        x["p_cal"] >= x["thr05"]
    )

    return x


# =============================================================================
# DATE-LEVEL PANEL
# =============================================================================

def build_date_level_panel(
    df: pd.DataFrame,
) -> pd.DataFrame:

    x = assign_bands(
        df
    )

    rows = []

    for (
        target,
        model,
        date,
    ), g in x.groupby(
        [
            "target",
            "model",
            "forecast_date",
        ],
        sort=True,
        observed=True,
    ):

        rows.append(
            {
                "target":
                    target,

                "model":
                    model,

                "forecast_date":
                    date,

                "n_tickers":
                    int(len(g)),

                "mean_p":
                    float(
                        g["p_cal"].mean()
                    ),

                "median_p":
                    float(
                        g["p_cal"].median()
                    ),

                "realized_fraction":
                    float(
                        g["y"].mean()
                    ),

                "green_fraction":
                    float(
                        (
                            g["trust_band"]
                            == "GREEN"
                        ).mean()
                    ),

                "amber_fraction":
                    float(
                        (
                            g["trust_band"]
                            == "AMBER"
                        ).mean()
                    ),

                "red_fraction":
                    float(
                        (
                            g["trust_band"]
                            == "RED"
                        ).mean()
                    ),

                "critical_fraction":
                    float(
                        g["critical_5pct"].mean()
                    ),

                "mean_thr20":
                    float(
                        g["thr20"].mean()
                    ),

                "mean_thr10":
                    float(
                        g["thr10"].mean()
                    ),

                "mean_thr05":
                    float(
                        g["thr05"].mean()
                    ),
            }
        )

    panel = pd.DataFrame(
        rows
    )

    if panel.empty:
        raise RuntimeError(
            "Date-level panel is empty."
        )

    panel = panel.sort_values(
        [
            "target",
            "model",
            "forecast_date",
        ]
    ).reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # System-level warning:
    #
    # >=25% of stocks in RED.
    # -------------------------------------------------------------------------

    panel["system_red_alert"] = (
        panel["red_fraction"]
        >= SYSTEM_RED_FRACTION
    )

    return panel


# =============================================================================
# TRADING-DAY DIFFERENCE
# =============================================================================

def trading_day_difference(
    dates: np.ndarray,
    earlier,
    later,
) -> int:

    dates = np.sort(
        pd.to_datetime(
            dates
        )
    )

    earlier = pd.Timestamp(
        earlier
    )

    later = pd.Timestamp(
        later
    )

    earlier_idx = np.searchsorted(
        dates,
        earlier,
    )

    later_idx = np.searchsorted(
        dates,
        later,
    )

    return int(
        later_idx - earlier_idx
    )


# =============================================================================
# FIRST SYSTEM RED ALERT
# =============================================================================

def first_system_red_alert(
    panel: pd.DataFrame,
    target: str,
    model: str,
    event_onset: pd.Timestamp,
) -> tuple[pd.Timestamp | pd.NaT, float]:

    dates = np.sort(
        panel[
            (panel["target"] == target)
            &
            (panel["model"] == model)
        ]["forecast_date"].unique()
    )

    # Strictly BEFORE the event onset.
    x = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] < event_onset)
        &
        (
            panel["system_red_alert"]
            == True
        )
    ].sort_values(
        "forecast_date"
    )

    if x.empty:
        return (
            pd.NaT,
            np.nan,
        )

    alert_date = pd.Timestamp(
        x.iloc[0]["forecast_date"]
    )

    lead = trading_day_difference(
        dates,
        alert_date,
        event_onset,
    )

    return (
        alert_date,
        float(lead),
    )


# =============================================================================
# FIRST CROSSING OF MEAN PROBABILITY THRESHOLDS
# =============================================================================

def first_probability_crossing(
    panel: pd.DataFrame,
    target: str,
    model: str,
    event_onset: pd.Timestamp,
    threshold_column: str,
) -> tuple[pd.Timestamp | pd.NaT, float]:

    dates = np.sort(
        panel[
            (panel["target"] == target)
            &
            (panel["model"] == model)
        ]["forecast_date"].unique()
    )

    x = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] < event_onset)
        &
        (
            panel["mean_p"]
            >=
            panel[threshold_column]
        )
    ].sort_values(
        "forecast_date"
    )

    if x.empty:
        return (
            pd.NaT,
            np.nan,
        )

    d = pd.Timestamp(
        x.iloc[0]["forecast_date"]
    )

    lead = trading_day_difference(
        dates,
        d,
        event_onset,
    )

    return (
        d,
        float(lead),
    )


# =============================================================================
# EVENT-DATE STATISTICS
# =============================================================================

def event_day_row(
    panel: pd.DataFrame,
    target: str,
    model: str,
    event_onset: pd.Timestamp,
) -> pd.Series | None:

    x = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (
            panel["forecast_date"]
            == event_onset
        )
    ]

    if x.empty:
        return None

    return x.iloc[0]


# =============================================================================
# CASE SUMMARY
# =============================================================================

def build_case_summary(
    panel: pd.DataFrame,
    target: str,
    model: str,
) -> dict:

    event_onset = EVENT_ONSET[
        target
    ]

    pre_start = (
        event_onset
        - pd.Timedelta(
            days=PRE_EVENT_CALENDAR_DAYS
        )
    )

    post_end = (
        event_onset
        + pd.Timedelta(
            days=POST_EVENT_CALENDAR_DAYS
        )
    )

    # -------------------------------------------------------------------------
    # PRE-EVENT SIGNAL
    # -------------------------------------------------------------------------

    pre = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] >= pre_start)
        &
        (panel["forecast_date"] < event_onset)
    ].sort_values(
        "forecast_date"
    ).copy()

    # -------------------------------------------------------------------------
    # POST-EVENT REALIZED OUTCOME
    # -------------------------------------------------------------------------

    post = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] >= event_onset)
        &
        (panel["forecast_date"] <= post_end)
    ].sort_values(
        "forecast_date"
    ).copy()

    if pre.empty:
        raise RuntimeError(
            f"No pre-event observations for "
            f"{target} / {model}"
        )

    # -------------------------------------------------------------------------
    # FIRST 25%-RED SYSTEM ALERT
    # -------------------------------------------------------------------------

    first_red_date, red_lead = (
        first_system_red_alert(
            panel,
            target,
            model,
            event_onset,
        )
    )

    # -------------------------------------------------------------------------
    # FIRST MEAN PROBABILITY THRESHOLD CROSSINGS
    # -------------------------------------------------------------------------

    first_amber_date, amber_lead = (
        first_probability_crossing(
            panel,
            target,
            model,
            event_onset,
            "mean_thr20",
        )
    )

    first_red_probability_date, red_probability_lead = (
        first_probability_crossing(
            panel,
            target,
            model,
            event_onset,
            "mean_thr10",
        )
    )

    first_severe_date, severe_lead = (
        first_probability_crossing(
            panel,
            target,
            model,
            event_onset,
            "mean_thr05",
        )
    )

    # -------------------------------------------------------------------------
    # PRE-EVENT PROBABILITY
    # -------------------------------------------------------------------------

    first_pre_p = float(
        pre.iloc[0]["mean_p"]
    )

    final_pre_p = float(
        pre.iloc[-1]["mean_p"]
    )

    max_pre_p = float(
        pre["mean_p"].max()
    )

    mean_pre_p = float(
        pre["mean_p"].mean()
    )

    # -------------------------------------------------------------------------
    # PRE-EVENT BAND STATE
    # -------------------------------------------------------------------------

    last_pre = pre.iloc[-1]

    if last_pre["red_fraction"] >= SYSTEM_RED_FRACTION:
        final_system_state = "RED"

    elif (
        last_pre["amber_fraction"]
        >= SYSTEM_RED_FRACTION
    ):
        final_system_state = "AMBER"

    else:
        final_system_state = "GREEN"

    # -------------------------------------------------------------------------
    # POST-EVENT REALIZED OUTCOME
    # -------------------------------------------------------------------------

    if post.empty:

        max_realized = np.nan
        mean_realized = np.nan
        event_day_realized = np.nan

    else:

        max_realized = float(
            post[
                "realized_fraction"
            ].max()
        )

        mean_realized = float(
            post[
                "realized_fraction"
            ].mean()
        )

        event_day = event_day_row(
            panel,
            target,
            model,
            event_onset,
        )

        if event_day is None:
            event_day_realized = np.nan
        else:
            event_day_realized = float(
                event_day[
                    "realized_fraction"
                ]
            )

    return {
        "target":
            target,

        "model":
            model,

        "event_onset":
            event_onset,

        "pre_event_start":
            pre_start,

        "pre_event_end":
            event_onset
            - pd.Timedelta(
                days=1
            ),

        "first_pre_event_p":
            first_pre_p,

        "final_pre_event_p":
            final_pre_p,

        "max_pre_event_p":
            max_pre_p,

        "mean_pre_event_p":
            mean_pre_p,

        "probability_change_start_to_final":
            final_pre_p
            - first_pre_p,

        "first_amber_probability_crossing":
            first_amber_date,

        "amber_probability_lead_days":
            amber_lead,

        "first_red_probability_crossing":
            first_red_probability_date,

        "red_probability_lead_days":
            red_probability_lead,

        "first_severe_probability_crossing":
            first_severe_date,

        "severe_probability_lead_days":
            severe_lead,

        "first_system_red_alert":
            first_red_date,

        "system_red_lead_days":
            red_lead,

        "final_pre_event_system_state":
            final_system_state,

        "event_day_realized_fraction":
            event_day_realized,

        "max_post_event_realized_fraction":
            max_realized,

        "mean_post_event_realized_fraction":
            mean_realized,

        "system_alert_detected_before_event":
            pd.notna(first_red_date),
    }


# =============================================================================
# PLOT
# =============================================================================

def plot_case(
    panel: pd.DataFrame,
    target: str,
    model: str,
) -> None:

    event_onset = EVENT_ONSET[
        target
    ]

    pre_start = (
        event_onset
        - pd.Timedelta(
            days=PRE_EVENT_CALENDAR_DAYS
        )
    )

    post_end = (
        event_onset
        + pd.Timedelta(
            days=POST_EVENT_CALENDAR_DAYS
        )
    )

    # -------------------------------------------------------------------------
    # PRE-EVENT DATA
    # -------------------------------------------------------------------------

    pre = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] >= pre_start)
        &
        (panel["forecast_date"] < event_onset)
    ].sort_values(
        "forecast_date"
    ).copy()

    # -------------------------------------------------------------------------
    # POST-EVENT DATA
    # -------------------------------------------------------------------------

    post = panel[
        (panel["target"] == target)
        &
        (panel["model"] == model)
        &
        (panel["forecast_date"] >= event_onset)
        &
        (panel["forecast_date"] <= post_end)
    ].sort_values(
        "forecast_date"
    ).copy()

    if pre.empty:
        print(
            f"No pre-event data for {target}/{model}"
        )
        return

    # -------------------------------------------------------------------------
    # FIRST SYSTEM RED ALERT
    # -------------------------------------------------------------------------

    first_red_date, red_lead = (
        first_system_red_alert(
            panel,
            target,
            model,
            event_onset,
        )
    )

    # -------------------------------------------------------------------------
    # FIGURE
    # -------------------------------------------------------------------------

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.45,
                1.0,
                1.0,
            ]
        },
    )

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]

    # =========================================================================
    # PANEL 1 — EX-ANTE PREDICTED PROBABILITY
    # =========================================================================

    ax1.plot(
        pre["forecast_date"],
        pre["mean_p"],
        linewidth=2,
        label="Mean calibrated probability",
    )

    ax1.plot(
        pre["forecast_date"],
        pre["mean_thr20"],
        linestyle="--",
        linewidth=1,
        label="thr20",
    )

    ax1.plot(
        pre["forecast_date"],
        pre["mean_thr10"],
        linestyle="--",
        linewidth=1,
        label="thr10",
    )

    ax1.plot(
        pre["forecast_date"],
        pre["mean_thr05"],
        linestyle=":",
        linewidth=1,
        label="thr05",
    )

    # Fixed event onset.
    ax1.axvline(
        event_onset,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Event onset"
        ),
    )

    # Correct system-level RED alert.
    if first_red_date is not None:

        ax1.axvline(
            first_red_date,
            linestyle=":",
            linewidth=1.8,
            label=(
                "First system RED alert "
                "(>=25% stocks)"
            ),
        )

    ax1.set_ylabel(
        "Predicted probability"
    )

    if target == "T1":
        title_target = (
            "T1 — Future 10-Day Deterioration"
        )
        onset_text = (
            "Fixed COVID stress onset"
        )
    else:
        title_target = (
            "T2 — Future 10-Day Coverage Failure"
        )
        onset_text = (
            "Outcome-defined COVID "
            "failure onset"
        )

    ax1.set_title(
        f"{model} — {title_target}\n"
        "Ex-Ante Reliability Signal"
    )

    ax1.grid(
        alpha=0.25
    )

    ax1.legend(
        loc="upper left"
    )

    # Explanation.
    ax1.text(
        0.99,
        0.95,
        "EX-ANTE ONLY\n"
        "Probability uses information available "
        "through each forecast date.",
        transform=ax1.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.15,
        ),
    )

    # =========================================================================
    # PANEL 2 — SYSTEM RELIABILITY-BAND COMPOSITION
    # =========================================================================

    ax2.stackplot(
        pre["forecast_date"],
        pre["green_fraction"],
        pre["amber_fraction"],
        pre["red_fraction"],
        labels=[
            "GREEN",
            "AMBER",
            "RED",
        ],
        alpha=0.80,
    )

    # 25% alert threshold on RED fraction.
    ax2.axhline(
        SYSTEM_RED_FRACTION,
        linestyle="--",
        linewidth=1,
        label="System RED threshold (25%)",
    )

    ax2.axvline(
        event_onset,
        linestyle="--",
        linewidth=1.5,
    )

    if first_red_date is not None:

        ax2.axvline(
            first_red_date,
            linestyle=":",
            linewidth=1.8,
        )

    ax2.set_ylim(
        0,
        1,
    )

    ax2.set_ylabel(
        "Fraction of stocks"
    )

    ax2.set_title(
        "Ex-Ante Reliability-Band Composition"
    )

    ax2.grid(
        alpha=0.20
    )

    ax2.legend(
        loc="upper left"
    )

    # =========================================================================
    # PANEL 3 — POST-EVENT REALIZED OUTCOME
    # =========================================================================

    if not post.empty:

        ax3.plot(
            post["forecast_date"],
            post["realized_fraction"],
            linewidth=2,
            label=(
                "Realized outcome fraction"
            ),
        )

    ax3.axvline(
        event_onset,
        linestyle="--",
        linewidth=1.5,
        label=onset_text,
    )

    ax3.set_ylabel(
        "Realized fraction"
    )

    ax3.set_xlabel(
        "Forecast date"
    )

    ax3.set_title(
        "Post-Onset Realized Outcome — "
        "Evaluation Only"
    )

    ax3.grid(
        alpha=0.25
    )

    ax3.legend(
        loc="upper left"
    )

    ax3.text(
        0.99,
        0.95,
        "EX-POST EVALUATION\n"
        "Realized outcomes are revealed here "
        "only to evaluate the forecasts.",
        transform=ax3.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.15,
        ),
    )

    # =========================================================================
    # SHARED ANNOTATION
    # =========================================================================

    if first_red_date is not None:

        annotation = (
            f"First system RED: "
            f"{first_red_date.date()}\n"
            f"Lead: {int(red_lead)} trading days"
        )

        ax1.text(
            0.99,
            0.72,
            annotation,
            transform=ax1.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                alpha=0.15,
            ),
        )

    fig.tight_layout()

    filename = (
        OUTPUT_DIR
        / (
            f"RS05_COVID_CORRECTED_"
            f"{model}_{target}_S6.png"
        )
    )

    fig.savefig(
        filename,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )

    print(
        f"Saved plot: {filename}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    # -------------------------------------------------------------------------
    # LOAD
    # -------------------------------------------------------------------------

    df = load_predictions()

    # -------------------------------------------------------------------------
    # BUILD DATE PANEL
    # -------------------------------------------------------------------------

    panel = build_date_level_panel(
        df
    )

    panel.to_csv(
        DATE_LEVEL_OUT,
        index=False,
    )

    print()
    print(
        f"Saved date-level panel:\n"
        f"{DATE_LEVEL_OUT}"
    )

    # -------------------------------------------------------------------------
    # BUILD SUMMARIES
    # -------------------------------------------------------------------------

    results = []

    for target in TARGETS:

        for model in MODELS:

            result = build_case_summary(
                panel=panel,
                target=target,
                model=model,
            )

            results.append(
                result
            )

    summary = pd.DataFrame(
        results
    )

    summary = summary.sort_values(
        [
            "target",
            "model",
        ]
    )

    summary.to_csv(
        SUMMARY_OUT,
        index=False,
    )

    summary[
        summary["target"] == "T1"
    ].to_csv(
        T1_SUMMARY_OUT,
        index=False,
    )

    summary[
        summary["target"] == "T2"
    ].to_csv(
        T2_SUMMARY_OUT,
        index=False,
    )

    # -------------------------------------------------------------------------
    # PRINT T1
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "CORRECTED COVID CASE STUDY — T1"
    )
    print("=" * 110)

    print(
        summary[
            summary["target"] == "T1"
        ][
            [
                "model",
                "event_onset",
                "first_pre_event_p",
                "final_pre_event_p",
                "max_pre_event_p",
                "first_system_red_alert",
                "system_red_lead_days",
                "final_pre_event_system_state",
                "event_day_realized_fraction",
                "max_post_event_realized_fraction",
                "system_alert_detected_before_event",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # PRINT T2
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "CORRECTED COVID CASE STUDY — T2"
    )
    print("=" * 110)

    print(
        summary[
            summary["target"] == "T2"
        ][
            [
                "model",
                "event_onset",
                "first_pre_event_p",
                "final_pre_event_p",
                "max_pre_event_p",
                "first_system_red_alert",
                "system_red_lead_days",
                "final_pre_event_system_state",
                "event_day_realized_fraction",
                "max_post_event_realized_fraction",
                "system_alert_detected_before_event",
            ]
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # GENERATE FIGURES
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "GENERATING CORRECTED COVID FIGURES"
    )
    print("=" * 110)

    for target in TARGETS:

        for model in MODELS:

            plot_case(
                panel=panel,
                target=target,
                model=model,
            )

    # -------------------------------------------------------------------------
    # FINAL INTEGRITY
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print(
        "FINAL INTEGRITY CHECK"
    )
    print("=" * 110)

    expected = (
        len(TARGETS)
        * len(MODELS)
    )

    actual = (
        panel[
            [
                "target",
                "model",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    print(
        f"Expected target/model combinations: "
        f"{expected}"
    )

    print(
        f"Actual target/model combinations:   "
        f"{actual}"
    )

    if expected != actual:
        raise RuntimeError(
            "Missing target/model combinations."
        )

    # -------------------------------------------------------------------------
    # Verify no pre-event probability enters post-event side.
    # -------------------------------------------------------------------------

    for target in TARGETS:

        onset = EVENT_ONSET[
            target
        ]

        bad = panel[
            (
                panel["target"]
                == target
            )
            &
            (
                panel["forecast_date"]
                < onset
            )
        ]

        if bad.empty:
            raise RuntimeError(
                f"No pre-event rows for {target}"
            )

        max_pre_date = bad[
            "forecast_date"
        ].max()

        if max_pre_date >= onset:
            raise RuntimeError(
                f"Pre-event panel leakage for {target}"
            )

        print(
            f"{target}: latest ex-ante date = "
            f"{max_pre_date.date()} | "
            f"event onset = {onset.date()}"
        )

    print()
    print(
        "Thresholds remain frozen."
    )

    print(
        "No RS05 training or recalibration performed."
    )

    print(
        "System RED requires >=25% of stocks in RED."
    )

    print()
    print(
        "DONE."
    )


if __name__ == "__main__":
    main()