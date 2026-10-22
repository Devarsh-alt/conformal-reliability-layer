from pathlib import Path
import logging
import numpy as np
import pandas as pd


# =============================================================================
# RS02B — MULTI-EVENT REGIME-SHIFT DETERIORATION ANALYSIS
# =============================================================================

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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DETAIL_PATH = OUTPUT_DIR / "RS02B_multi_event_deterioration.csv"
SUMMARY_PATH = OUTPUT_DIR / "RS02B_event_summary.csv"
COMPARISON_PATH = OUTPUT_DIR / "RS02B_model_comparison.csv"
LOG_PATH = OUTPUT_DIR / "RS02B_multi_event_deterioration.log"


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# =============================================================================
# MODELS
# =============================================================================

MODELS = {
    "HS": {
        "var": "hs_var",
        "breach": "hs_breach",
        "loss": "hs_pinball_loss",
    },
    "EWMA": {
        "var": "ewma_var",
        "breach": "ewma_breach",
        "loss": "ewma_pinball_loss",
    },
    "GARCH-t": {
        "var": "garch_var",
        "breach": "garch_breach",
        "loss": "garch_pinball_loss",
    },
    "QRF": {
        "var": "qrf_var",
        "breach": "qrf_breach",
        "loss": "qrf_pinball_loss",
    },
}


# =============================================================================
# EVENT DEFINITIONS
#
# These are descriptive event windows.
# RS03 will later replace these hand-defined windows with data-driven
# regime identification using market conditions.
# =============================================================================

EVENTS = {
    "2018_volatility_shock": {
        "pre_event": ("2017-01-01", "2017-12-31"),
        "transition": ("2018-01-01", "2018-02-28"),
        "stress": ("2018-03-01", "2018-04-30"),
        "recovery": ("2018-05-01", "2018-12-31"),
    },

    "2022_rate_tightening": {
        "pre_event": ("2021-01-01", "2021-12-31"),
        "transition": ("2022-01-01", "2022-03-31"),
        "stress": ("2022-04-01", "2022-09-30"),
        "recovery": ("2022-10-01", "2022-12-31"),
    },

    "2023_banking_stress": {
        "pre_event": ("2022-06-01", "2023-01-31"),
        "transition": ("2023-02-01", "2023-03-15"),
        "stress": ("2023-03-16", "2023-04-30"),
        "recovery": ("2023-05-01", "2023-05-31"),
    },

    "2025_volatility_episode": {
        "pre_event": ("2024-01-01", "2024-12-31"),
        "transition": ("2025-01-01", "2025-03-31"),
        "stress": ("2025-04-01", "2025-05-31"),
        "recovery": ("2025-06-01", "2025-06-30"),
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def assign_event_period(date, event_definition):
    """
    Assign one forecast date to a descriptive event period.
    """

    date = pd.Timestamp(date)

    for period, (start, end) in event_definition.items():

        start = pd.Timestamp(start)
        end = pd.Timestamp(end)

        if start <= date <= end:
            return period

    return None


def safe_relative_change(event_value, baseline_value):
    """
    Relative change:

        (event - baseline) / baseline

    Returns NaN when the baseline is zero or unavailable.
    """

    if pd.isna(event_value) or pd.isna(baseline_value):
        return np.nan

    if baseline_value == 0:
        return np.nan

    return (event_value - baseline_value) / baseline_value


def calculate_metrics(df, model_name):
    """
    Calculate model performance metrics for a dataframe.
    """

    cols = MODELS[model_name]

    x = df[[
        cols["var"],
        cols["breach"],
        cols["loss"],
    ]].dropna()

    if len(x) == 0:
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
        "n_obs": len(x),

        "breach_count": int(
            x[cols["breach"]].sum()
        ),

        "breach_rate": float(
            x[cols["breach"]].mean()
        ),

        "mean_pinball_loss": float(
            x[cols["loss"]].mean()
        ),

        "median_pinball_loss": float(
            x[cols["loss"]].median()
        ),

        "mean_abs_var": float(
            x[cols["var"]].abs().mean()
        ),

        "mean_var": float(
            x[cols["var"]].mean()
        ),
    }


def add_deterioration_metrics(row, baseline):
    """
    Add changes relative to the pre-event baseline.
    """

    row["breach_rate_change"] = (
        row["breach_rate"] - baseline["breach_rate"]
        if pd.notna(row["breach_rate"])
        and pd.notna(baseline["breach_rate"])
        else np.nan
    )

    row["breach_rate_relative_change"] = safe_relative_change(
        row["breach_rate"],
        baseline["breach_rate"],
    )

    row["pinball_loss_change"] = (
        row["mean_pinball_loss"]
        - baseline["mean_pinball_loss"]
        if pd.notna(row["mean_pinball_loss"])
        and pd.notna(baseline["mean_pinball_loss"])
        else np.nan
    )

    row["pinball_loss_relative_change"] = safe_relative_change(
        row["mean_pinball_loss"],
        baseline["mean_pinball_loss"],
    )

    row["abs_var_change"] = (
        row["mean_abs_var"]
        - baseline["mean_abs_var"]
        if pd.notna(row["mean_abs_var"])
        and pd.notna(baseline["mean_abs_var"])
        else np.nan
    )

    row["abs_var_relative_change"] = safe_relative_change(
        row["mean_abs_var"],
        baseline["mean_abs_var"],
    )

    return row


# =============================================================================
# START
# =============================================================================

log.info("=" * 100)
log.info("RS02B — MULTI-EVENT REGIME-SHIFT DETERIORATION ANALYSIS")
log.info("=" * 100)

log.info("")
log.info("Loading RS01 panel:")
log.info(str(PANEL_PATH))

if not PANEL_PATH.exists():
    raise FileNotFoundError(
        f"RS01 panel not found:\n{PANEL_PATH}"
    )

panel = pd.read_csv(PANEL_PATH)

panel["forecast_date"] = pd.to_datetime(
    panel["forecast_date"]
)

panel["realized_date"] = pd.to_datetime(
    panel["realized_date"]
)


# =============================================================================
# PANEL CHECK
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("PANEL CHECK")
log.info("=" * 100)

log.info(f"Rows: {len(panel):,}")
log.info(f"Tickers: {panel['ticker'].nunique()}")
log.info(
    f"Forecast date: "
    f"{panel['forecast_date'].min().date()} "
    f"to "
    f"{panel['forecast_date'].max().date()}"
)
log.info(
    f"Realized date: "
    f"{panel['realized_date'].min().date()} "
    f"to "
    f"{panel['realized_date'].max().date()}"
)

required_columns = ["ticker", "forecast_date", "realized_date"]

for model in MODELS.values():
    required_columns.extend([
        model["var"],
        model["breach"],
        model["loss"],
    ])

missing_columns = [
    c for c in required_columns
    if c not in panel.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# =============================================================================
# EVENT ANALYSIS
# =============================================================================

all_detail_rows = []
all_summary_rows = []
all_comparison_rows = []


for event_name, event_definition in EVENTS.items():

    log.info("")
    log.info("=" * 100)
    log.info(f"EVENT: {event_name}")
    log.info("=" * 100)

    # -------------------------------------------------------------------------
    # Assign event periods
    # -------------------------------------------------------------------------

    event_panel = panel.copy()

    event_panel["event_period"] = event_panel[
        "forecast_date"
    ].apply(
        lambda x: assign_event_period(
            x,
            event_definition,
        )
    )

    event_panel = event_panel[
        event_panel["event_period"].notna()
    ].copy()

    log.info(
        f"Event-study observations: "
        f"{len(event_panel):,}"
    )

    if len(event_panel) == 0:
        log.warning(
            f"No observations found for {event_name}. Skipping."
        )
        continue

    period_counts = (
        event_panel["event_period"]
        .value_counts()
        .reindex(
            ["pre_event", "transition", "stress", "recovery"]
        )
    )

    log.info("")
    log.info("Observations by event period:")
    log.info("")
    log.info(str(period_counts))


    # -------------------------------------------------------------------------
    # MODEL × TICKER × PERIOD
    # -------------------------------------------------------------------------

    for ticker in sorted(event_panel["ticker"].unique()):

        ticker_df = event_panel[
            event_panel["ticker"] == ticker
        ]

        for model_name in MODELS:

            baseline_df = ticker_df[
                ticker_df["event_period"] == "pre_event"
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

                period_df = ticker_df[
                    ticker_df["event_period"] == period
                ]

                metrics = calculate_metrics(
                    period_df,
                    model_name,
                )

                row = {
                    "event": event_name,
                    "ticker": ticker,
                    "model": model_name,
                    "period": period,

                    **metrics,
                }

                row = add_deterioration_metrics(
                    row,
                    baseline,
                )

                all_detail_rows.append(row)


    # -------------------------------------------------------------------------
    # POOLED EVENT SUMMARY
    # -------------------------------------------------------------------------

    log.info("")
    log.info("-" * 100)
    log.info(f"{event_name.upper()} — POOLED RESULTS")
    log.info("-" * 100)

    for model_name in MODELS:

        model_event_df = event_panel.copy()

        baseline_df = model_event_df[
            model_event_df["event_period"] == "pre_event"
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

            period_df = model_event_df[
                model_event_df["event_period"] == period
            ]

            metrics = calculate_metrics(
                period_df,
                model_name,
            )

            row = {
                "event": event_name,
                "model": model_name,
                "period": period,
                **metrics,
            }

            row = add_deterioration_metrics(
                row,
                baseline,
            )

            all_summary_rows.append(row)


    # -------------------------------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------------------------------

    stress_rows = [
        r
        for r in all_summary_rows
        if r["event"] == event_name
        and r["period"] == "stress"
    ]

    comparison_df = pd.DataFrame(stress_rows)

    if not comparison_df.empty:

        comparison_df = comparison_df.sort_values(
            "pinball_loss_relative_change",
            ascending=False,
        )

        log.info("")
        log.info("-" * 100)
        log.info(
            f"{event_name.upper()} — STRESS PERIOD MODEL COMPARISON"
        )
        log.info("-" * 100)
        log.info("")

        display_cols = [
            "model",
            "breach_rate",
            "mean_pinball_loss",
            "mean_abs_var",
            "breach_rate_relative_change",
            "pinball_loss_relative_change",
        ]

        log.info(
            comparison_df[
                display_cols
            ].to_string(
                index=False,
                float_format=lambda x: f"{x:.6f}",
            )
        )

        for _, row in comparison_df.iterrows():

            all_comparison_rows.append(
                row.to_dict()
            )


# =============================================================================
# SAVE RESULTS
# =============================================================================

detail_df = pd.DataFrame(
    all_detail_rows
)

summary_df = pd.DataFrame(
    all_summary_rows
)

comparison_df = pd.DataFrame(
    all_comparison_rows
)


detail_df.to_csv(
    DETAIL_PATH,
    index=False,
)

summary_df.to_csv(
    SUMMARY_PATH,
    index=False,
)

comparison_df.to_csv(
    COMPARISON_PATH,
    index=False,
)


# =============================================================================
# CROSS-EVENT SUMMARY
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("CROSS-EVENT STRESS COMPARISON")
log.info("=" * 100)

if not comparison_df.empty:

    cross_event = (
        comparison_df[
            [
                "event",
                "model",
                "breach_rate",
                "mean_pinball_loss",
                "breach_rate_relative_change",
                "pinball_loss_relative_change",
            ]
        ]
        .sort_values(
            [
                "event",
                "pinball_loss_relative_change",
            ],
            ascending=[True, False],
        )
    )

    log.info("")
    log.info(
        cross_event.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# =============================================================================
# BEST / WORST MODEL BY EVENT
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("WORST STRESS-PERIOD DETERIORATION BY EVENT")
log.info("=" * 100)

if not comparison_df.empty:

    for event_name in comparison_df["event"].unique():

        event_df = comparison_df[
            comparison_df["event"] == event_name
        ].copy()

        event_df = event_df.sort_values(
            "pinball_loss_relative_change",
            ascending=False,
        )

        worst = event_df.iloc[0]

        log.info(
            f"{event_name}: "
            f"{worst['model']} | "
            f"pinball relative change = "
            f"{worst['pinball_loss_relative_change']:.4f} | "
            f"breach rate = "
            f"{worst['breach_rate']:.4f}"
        )


# =============================================================================
# OUTPUT SUMMARY
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("OUTPUTS SAVED")
log.info("=" * 100)

log.info(f"Detailed results:")
log.info(str(DETAIL_PATH))

log.info(f"Summary results:")
log.info(str(SUMMARY_PATH))

log.info(f"Model comparison:")
log.info(str(COMPARISON_PATH))

log.info(f"Log:")
log.info(str(LOG_PATH))

log.info("")
log.info("=" * 100)
log.info("RS02B MULTI-EVENT ANALYSIS COMPLETE")
log.info("=" * 100)