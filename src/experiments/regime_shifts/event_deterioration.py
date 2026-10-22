"""
RS02 — EVENT-BASED VaR DETERIORATION ANALYSIS

Purpose
-------
Measure how existing VaR models deteriorate during a major market
regime shift.

First event:
    COVID-19 market shock, 2020

Models:
    HS
    EWMA
    GARCH-t
    QRF

The experiment is descriptive/event-based.

It DOES NOT:
    - train a new model
    - use future information
    - define a predictive reliability score
    - classify regimes using the future target

Those tasks belong to later RS experiments.

Event windows
-------------
Pre-event baseline:
    2019-01-01 -> 2020-01-31

Transition:
    2020-02-01 -> 2020-03-15

Stress:
    2020-03-16 -> 2020-05-31

Recovery:
    2020-06-01 -> 2020-12-31

For each ticker/model we calculate:
    - observations
    - breach count
    - breach rate
    - mean pinball loss
    - median pinball loss
    - mean absolute VaR
    - mean VaR
    - deterioration in breach rate
    - deterioration in pinball loss
    - relative deterioration

Outputs
-------
RS02_event_deterioration.csv
RS02_event_summary.csv
RS02_event_deterioration.log
"""

from pathlib import Path
import logging

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"D:\reliability_layer")

PANEL_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_long_history_var_panel.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
)

OUTPUT_DETAIL = (
    OUTPUT_DIR
    / "RS02_event_deterioration.csv"
)

OUTPUT_SUMMARY = (
    OUTPUT_DIR
    / "RS02_event_summary.csv"
)

OUTPUT_LOG = (
    OUTPUT_DIR
    / "RS02_event_deterioration.log"
)


# ============================================================
# EVENT DEFINITION
# ============================================================

EVENT_NAME = "COVID-19"

EVENT_WINDOWS = {
    "pre_event": (
        pd.Timestamp("2019-01-01"),
        pd.Timestamp("2020-01-31"),
    ),

    "transition": (
        pd.Timestamp("2020-02-01"),
        pd.Timestamp("2020-03-15"),
    ),

    "stress": (
        pd.Timestamp("2020-03-16"),
        pd.Timestamp("2020-05-31"),
    ),

    "recovery": (
        pd.Timestamp("2020-06-01"),
        pd.Timestamp("2020-12-31"),
    ),
}


# ============================================================
# MODELS
# ============================================================

MODELS = {
    "HS": {
        "var": "hs_var",
        "breach": "hs_breach",
        "pinball": "hs_pinball_loss",
    },

    "EWMA": {
        "var": "ewma_var",
        "breach": "ewma_breach",
        "pinball": "ewma_pinball_loss",
    },

    "GARCH-t": {
        "var": "garch_var",
        "breach": "garch_breach",
        "pinball": "garch_pinball_loss",
    },

    "QRF": {
        "var": "qrf_var",
        "breach": "qrf_breach",
        "pinball": "qrf_pinball_loss",
    },
}


# ============================================================
# LOGGING
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            OUTPUT_LOG,
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def assign_event_period(date):
    """
    Assign each forecast date to one of the predefined COVID
    event periods.

    Dates outside the event-study window are excluded.
    """

    for period, (start, end) in EVENT_WINDOWS.items():

        if start <= date <= end:
            return period

    return None


def safe_relative_change(event_value, baseline_value):
    """
    Relative change from baseline.

    Returns NaN if the baseline is zero or unavailable.
    """

    if (
        pd.isna(event_value)
        or pd.isna(baseline_value)
        or baseline_value == 0
    ):
        return np.nan

    return (
        (event_value - baseline_value)
        / abs(baseline_value)
    )


def calculate_metrics(df, model_name):
    """
    Calculate model performance metrics for a dataframe subset.
    """

    columns = MODELS[model_name]

    valid = df[
        df[columns["var"]].notna()
        & df[columns["breach"]].notna()
        & df[columns["pinball"]].notna()
    ].copy()

    if len(valid) == 0:

        return {
            "n_obs": 0,
            "breach_count": np.nan,
            "breach_rate": np.nan,
            "mean_pinball_loss": np.nan,
            "median_pinball_loss": np.nan,
            "mean_abs_var": np.nan,
            "mean_var": np.nan,
        }

    return {
        "n_obs": int(len(valid)),

        "breach_count": int(
            valid[columns["breach"]].sum()
        ),

        "breach_rate": float(
            valid[columns["breach"]].mean()
        ),

        "mean_pinball_loss": float(
            valid[columns["pinball"]].mean()
        ),

        "median_pinball_loss": float(
            valid[columns["pinball"]].median()
        ),

        "mean_abs_var": float(
            valid[columns["var"]].abs().mean()
        ),

        "mean_var": float(
            valid[columns["var"]].mean()
        ),
    }


# ============================================================
# LOAD PANEL
# ============================================================

logger.info("=" * 100)
logger.info("RS02 — COVID-19 EVENT DETERIORATION ANALYSIS")
logger.info("=" * 100)

logger.info("")
logger.info("Loading RS01 panel:")
logger.info(str(PANEL_PATH))

if not PANEL_PATH.exists():

    raise FileNotFoundError(
        f"RS01 panel not found:\n{PANEL_PATH}"
    )


panel = pd.read_csv(
    PANEL_PATH,
    parse_dates=[
        "forecast_date",
        "realized_date",
    ],
)


# ============================================================
# BASIC PANEL CHECK
# ============================================================

logger.info("")
logger.info("=" * 100)
logger.info("PANEL CHECK")
logger.info("=" * 100)

logger.info(
    f"Rows: {len(panel):,}"
)

logger.info(
    f"Tickers: {panel['ticker'].nunique()}"
)

logger.info(
    f"Forecast date: "
    f"{panel['forecast_date'].min().date()} "
    f"to "
    f"{panel['forecast_date'].max().date()}"
)

logger.info(
    f"Realized date: "
    f"{panel['realized_date'].min().date()} "
    f"to "
    f"{panel['realized_date'].max().date()}"
)


# ============================================================
# ASSIGN EVENT PERIOD
# ============================================================

panel["event_period"] = (
    panel["forecast_date"]
    .apply(assign_event_period)
)


event_panel = panel[
    panel["event_period"].notna()
].copy()


logger.info("")
logger.info(
    f"Event-study observations: "
    f"{len(event_panel):,}"
)

logger.info("")
logger.info("Observations by event period:")

logger.info(
    "\n"
    + event_panel["event_period"]
    .value_counts()
    .sort_index()
    .to_string()
)


# ============================================================
# DETAILED MODEL/PERIOD ANALYSIS
# ============================================================

detail_rows = []


for ticker in sorted(
    event_panel["ticker"].unique()
):

    ticker_df = event_panel[
        event_panel["ticker"] == ticker
    ].copy()

    for model_name in MODELS:

        # ----------------------------------------------------
        # Baseline
        # ----------------------------------------------------

        baseline_df = ticker_df[
            ticker_df["event_period"]
            == "pre_event"
        ]

        baseline = calculate_metrics(
            baseline_df,
            model_name,
        )

        # ----------------------------------------------------
        # Each event period
        # ----------------------------------------------------

        for period in [
            "pre_event",
            "transition",
            "stress",
            "recovery",
        ]:

            period_df = ticker_df[
                ticker_df["event_period"]
                == period
            ]

            metrics = calculate_metrics(
                period_df,
                model_name,
            )

            row = {
                "event": EVENT_NAME,
                "ticker": ticker,
                "model": model_name,
                "period": period,

                "start_date":
                    EVENT_WINDOWS[period][0],

                "end_date":
                    EVENT_WINDOWS[period][1],

                "n_obs":
                    metrics["n_obs"],

                "breach_count":
                    metrics["breach_count"],

                "breach_rate":
                    metrics["breach_rate"],

                "mean_pinball_loss":
                    metrics["mean_pinball_loss"],

                "median_pinball_loss":
                    metrics["median_pinball_loss"],

                "mean_abs_var":
                    metrics["mean_abs_var"],

                "mean_var":
                    metrics["mean_var"],
            }

            # ------------------------------------------------
            # Deterioration relative to pre-event baseline
            # ------------------------------------------------

            row[
                "breach_rate_change"
            ] = (
                metrics["breach_rate"]
                - baseline["breach_rate"]
            )

            row[
                "breach_rate_relative_change"
            ] = safe_relative_change(
                metrics["breach_rate"],
                baseline["breach_rate"],
            )

            row[
                "pinball_loss_change"
            ] = (
                metrics["mean_pinball_loss"]
                - baseline["mean_pinball_loss"]
            )

            row[
                "pinball_loss_relative_change"
            ] = safe_relative_change(
                metrics["mean_pinball_loss"],
                baseline["mean_pinball_loss"],
            )

            row[
                "abs_var_change"
            ] = (
                metrics["mean_abs_var"]
                - baseline["mean_abs_var"]
            )

            row[
                "abs_var_relative_change"
            ] = safe_relative_change(
                metrics["mean_abs_var"],
                baseline["mean_abs_var"],
            )

            detail_rows.append(row)


detail = pd.DataFrame(
    detail_rows
)


# ============================================================
# POOLED MODEL/PERIOD SUMMARY
# ============================================================

summary_rows = []


for model_name in MODELS:

    baseline_df = event_panel[
        event_panel["event_period"]
        == "pre_event"
    ]

    baseline = calculate_metrics(
        baseline_df,
        model_name,
    )

    for period in [
        "pre_event",
        "transition",
        "stress",
        "recovery",
    ]:

        period_df = event_panel[
            event_panel["event_period"]
            == period
        ]

        metrics = calculate_metrics(
            period_df,
            model_name,
        )

        summary_rows.append(
            {
                "event": EVENT_NAME,
                "model": model_name,
                "period": period,

                "n_obs":
                    metrics["n_obs"],

                "breach_count":
                    metrics["breach_count"],

                "breach_rate":
                    metrics["breach_rate"],

                "mean_pinball_loss":
                    metrics["mean_pinball_loss"],

                "median_pinball_loss":
                    metrics["median_pinball_loss"],

                "mean_abs_var":
                    metrics["mean_abs_var"],

                "mean_var":
                    metrics["mean_var"],

                "breach_rate_change":
                    (
                        metrics["breach_rate"]
                        - baseline["breach_rate"]
                    ),

                "breach_rate_relative_change":
                    safe_relative_change(
                        metrics["breach_rate"],
                        baseline["breach_rate"],
                    ),

                "pinball_loss_change":
                    (
                        metrics["mean_pinball_loss"]
                        - baseline["mean_pinball_loss"]
                    ),

                "pinball_loss_relative_change":
                    safe_relative_change(
                        metrics["mean_pinball_loss"],
                        baseline["mean_pinball_loss"],
                    ),

                "abs_var_change":
                    (
                        metrics["mean_abs_var"]
                        - baseline["mean_abs_var"]
                    ),

                "abs_var_relative_change":
                    safe_relative_change(
                        metrics["mean_abs_var"],
                        baseline["mean_abs_var"],
                    ),
            }
        )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# PRINT MAIN RESULTS
# ============================================================

logger.info("")
logger.info("=" * 100)
logger.info("COVID EVENT — POOLED RESULTS")
logger.info("=" * 100)

display_columns = [
    "model",
    "period",
    "n_obs",
    "breach_rate",
    "mean_pinball_loss",
    "mean_abs_var",
    "breach_rate_change",
    "pinball_loss_change",
]


logger.info(
    "\n"
    + summary[
        display_columns
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# STRESS-ONLY COMPARISON
# ============================================================

logger.info("")
logger.info("=" * 100)
logger.info("COVID STRESS PERIOD — MODEL COMPARISON")
logger.info("=" * 100)

stress_summary = summary[
    summary["period"] == "stress"
].copy()

stress_summary = stress_summary.sort_values(
    "breach_rate",
    ascending=False,
)

logger.info(
    "\n"
    + stress_summary[
        [
            "model",
            "breach_rate",
            "mean_pinball_loss",
            "mean_abs_var",
            "breach_rate_relative_change",
            "pinball_loss_relative_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# MODEL-SPECIFIC TICKER STRESS SUMMARY
# ============================================================

logger.info("")
logger.info("=" * 100)
logger.info("COVID STRESS — TICKER-LEVEL BREACH RATES")
logger.info("=" * 100)

ticker_stress = detail[
    detail["period"] == "stress"
].copy()

ticker_stress = ticker_stress.sort_values(
    [
        "model",
        "breach_rate",
    ],
    ascending=[
        True,
        False,
    ],
)

logger.info(
    "\n"
    + ticker_stress[
        [
            "ticker",
            "model",
            "n_obs",
            "breach_rate",
            "mean_pinball_loss",
            "breach_rate_relative_change",
            "pinball_loss_relative_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# FIND WORST DETERIORATION
# ============================================================

logger.info("")
logger.info("=" * 100)
logger.info("LARGEST STRESS-PERIOD DETERIORATION")
logger.info("=" * 100)

worst = ticker_stress.sort_values(
    "pinball_loss_relative_change",
    ascending=False,
).head(20)

logger.info(
    "\n"
    + worst[
        [
            "ticker",
            "model",
            "breach_rate",
            "breach_rate_relative_change",
            "mean_pinball_loss",
            "pinball_loss_relative_change",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

detail.to_csv(
    OUTPUT_DETAIL,
    index=False,
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)


logger.info("")
logger.info("=" * 100)
logger.info("OUTPUTS SAVED")
logger.info("=" * 100)

logger.info(
    f"Detailed results:\n{OUTPUT_DETAIL}"
)

logger.info(
    f"Summary results:\n{OUTPUT_SUMMARY}"
)

logger.info(
    f"Log:\n{OUTPUT_LOG}"
)

logger.info("")
logger.info("=" * 100)
logger.info("RS02 COVID EVENT ANALYSIS COMPLETE")
logger.info("=" * 100)