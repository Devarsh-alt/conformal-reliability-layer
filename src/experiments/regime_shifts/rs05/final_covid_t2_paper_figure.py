from __future__ import annotations

"""
RS05 — FINAL COVID T2 PAPER FIGURE
===================================

T2:
    Future 10-day coverage failure

COVID T2 event onset:
    2020-02-12

Ex-ante warning window:
    Exactly 20 trading days before the event onset.

System RED alert:
    >= 25% of stocks in RED

Figure:
    Four model panels:
        EWMA
        GARCH-t
        HS
        QRF

Left y-axis:
    Predicted probability of future coverage failure

Right y-axis:
    Fraction of stocks in RED
    Realized coverage-failure fraction

The forecast probability and RED state are shown only before the event
onset. Realized coverage failure is shown only after the event onset.

No retraining.
No recalibration.
No threshold tuning.
"""


# =============================================================================
# IMPORTS
# =============================================================================

from pathlib import Path

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

CASE_DIR = (
    RS05_DIR
    / "case_study"
)

DATE_LEVEL_FILE = (
    CASE_DIR
    / "RS05_COVID_corrected_date_level_S6.csv"
)

LEAD_TIME_FILE = (
    CASE_DIR
    / "RS05_COVID_corrected_lead_time_S6.csv"
)

SUMMARY_FILE = (
    CASE_DIR
    / "RS05_COVID_corrected_summary_S6.csv"
)


# =============================================================================
# OUTPUTS
# =============================================================================

FIGURE_PNG = (
    CASE_DIR
    / "RS05_COVID_T2_FINAL_PAPER_FIGURE_S6.png"
)

FIGURE_PDF = (
    CASE_DIR
    / "RS05_COVID_T2_FINAL_PAPER_FIGURE_S6.pdf"
)

TABLE_CSV = (
    CASE_DIR
    / "RS05_COVID_T2_FINAL_PAPER_TABLE_S6.csv"
)

TABLE_TEX = (
    CASE_DIR
    / "RS05_COVID_T2_FINAL_PAPER_TABLE_S6.tex"
)


# =============================================================================
# DESIGN
# =============================================================================

MODELS = [
    "EWMA",
    "GARCH-t",
    "HS",
    "QRF",
]

TARGET = "T2"

EVENT_ONSET = pd.Timestamp(
    "2020-02-12"
)

PRE_EVENT_TRADING_DAYS = 20

POST_EVENT_END = pd.Timestamp(
    "2020-05-31"
)

SYSTEM_RED_THRESHOLD = 0.25


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    for path in [
        DATE_LEVEL_FILE,
        LEAD_TIME_FILE,
        SUMMARY_FILE,
    ]:

        if not path.exists():
            raise FileNotFoundError(
                f"Missing file:\n{path}"
            )

    panel = pd.read_csv(
        DATE_LEVEL_FILE
    )

    lead = pd.read_csv(
        LEAD_TIME_FILE
    )

    summary = pd.read_csv(
        SUMMARY_FILE
    )

    panel["forecast_date"] = pd.to_datetime(
        panel["forecast_date"]
    )

    lead["event_onset"] = pd.to_datetime(
        lead["event_onset"]
    )

    lead["first_system_red_alert"] = pd.to_datetime(
        lead["first_system_red_alert"]
    )

    summary["event_onset"] = pd.to_datetime(
        summary["event_onset"]
    )

    summary["first_system_red_alert"] = pd.to_datetime(
        summary["first_system_red_alert"]
    )

    return (
        panel,
        lead,
        summary,
    )


# =============================================================================
# EXACT 20-DAY PRE-EVENT WINDOW
# =============================================================================

def get_pre_event_dates(
    panel: pd.DataFrame,
    model: str,
) -> np.ndarray:

    dates = np.sort(
        panel[
            (panel["target"] == TARGET)
            &
            (panel["model"] == model)
        ]["forecast_date"].unique()
    )

    event_idx = np.searchsorted(
        dates,
        EVENT_ONSET,
    )

    start_idx = max(
        0,
        event_idx
        - PRE_EVENT_TRADING_DAYS,
    )

    return dates[
        start_idx:event_idx
    ]


# =============================================================================
# PAPER TABLE
# =============================================================================

def build_paper_table(
    lead: pd.DataFrame,
    summary: pd.DataFrame,
) -> pd.DataFrame:

    l = lead[
        lead["target"] == TARGET
    ].copy()

    s = summary[
        summary["target"] == TARGET
    ].copy()

    table = l.merge(
        s[
            [
                "model",
                "max_pre_event_p",
                "final_pre_event_p",
                "event_day_realized_fraction",
                "max_post_event_realized_fraction",
                "final_pre_event_system_state",
            ]
        ],
        on="model",
        how="left",
    )

    table = table[
        [
            "model",
            "first_system_red_alert",
            "lead_days",
            "alert_probability",
            "final_pre_event_probability",
            "max_pre_event_p",
            "final_pre_event_system_state",
            "event_day_realized_fraction",
            "max_post_event_realized_fraction",
            "detected",
        ]
    ].copy()

    table = table.rename(
        columns={
            "model":
                "Model",

            "first_system_red_alert":
                "First system RED",

            "lead_days":
                "Lead (trading days)",

            "alert_probability":
                "P(coverage failure) at first RED",

            "final_pre_event_probability":
                "P(coverage failure) at final pre-event date",

            "max_pre_event_p":
                "Maximum pre-event probability",

            "final_pre_event_system_state":
                "Final system state",

            "event_day_realized_fraction":
                "Realized coverage failure at onset",

            "max_post_event_realized_fraction":
                "Maximum post-onset coverage failure",

            "detected":
                "Detected before onset",
        }
    )

    return table.sort_values(
        "Model"
    )


# =============================================================================
# FIGURE
# =============================================================================

def make_figure(
    panel: pd.DataFrame,
    lead: pd.DataFrame,
):

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
    )

    axes = axes.flatten()

    for ax_left, model in zip(
        axes,
        MODELS,
    ):

        x = panel[
            (panel["target"] == TARGET)
            &
            (panel["model"] == model)
        ].sort_values(
            "forecast_date"
        ).copy()

        pre_dates = get_pre_event_dates(
            panel,
            model,
        )

        if len(pre_dates) == 0:
            continue

        pre_start = pd.Timestamp(
            pre_dates[0]
        )

        # ---------------------------------------------------------------------
        # PRE-EVENT
        # ---------------------------------------------------------------------

        pre = x[
            (x["forecast_date"] >= pre_start)
            &
            (x["forecast_date"] < EVENT_ONSET)
        ].copy()

        # ---------------------------------------------------------------------
        # POST-EVENT
        # ---------------------------------------------------------------------

        post = x[
            (x["forecast_date"] >= EVENT_ONSET)
            &
            (x["forecast_date"] <= POST_EVENT_END)
        ].copy()

        # ---------------------------------------------------------------------
        # FIRST SYSTEM RED
        # ---------------------------------------------------------------------

        l = lead[
            (lead["target"] == TARGET)
            &
            (lead["model"] == model)
        ]

        if l.empty:
            alert_date = pd.NaT
            lead_days = np.nan
        else:
            alert_date = pd.Timestamp(
                l.iloc[0]["first_system_red_alert"]
            )
            lead_days = float(
                l.iloc[0]["lead_days"]
            )

        # ---------------------------------------------------------------------
        # RIGHT AXIS
        # ---------------------------------------------------------------------

        ax_right = ax_left.twinx()

        # ---------------------------------------------------------------------
        # EX-ANTE PROBABILITY
        # ---------------------------------------------------------------------

        probability_line, = ax_left.plot(
            pre["forecast_date"],
            pre["mean_p"],
            linewidth=2.2,
            label=(
                "Predicted P(future coverage failure)"
            ),
        )

        # Thresholds on probability axis.
        ax_left.plot(
            pre["forecast_date"],
            pre["mean_thr20"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
            label="thr20",
        )

        ax_left.plot(
            pre["forecast_date"],
            pre["mean_thr10"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.65,
            label="thr10",
        )

        ax_left.plot(
            pre["forecast_date"],
            pre["mean_thr05"],
            linestyle=":",
            linewidth=0.9,
            alpha=0.65,
            label="thr05",
        )

        # ---------------------------------------------------------------------
        # RED FRACTION — RIGHT AXIS
        # ---------------------------------------------------------------------

        red_line, = ax_right.plot(
            pre["forecast_date"],
            pre["red_fraction"],
            linestyle="--",
            linewidth=1.8,
            label="Fraction of stocks RED",
        )

        ax_right.axhline(
            SYSTEM_RED_THRESHOLD,
            linestyle=":",
            linewidth=1.2,
            label="System RED threshold",
        )

        # ---------------------------------------------------------------------
        # REALIZED FAILURE — ONLY AFTER EVENT
        # ---------------------------------------------------------------------

        if not post.empty:

            realized_line, = ax_right.plot(
                post["forecast_date"],
                post["realized_fraction"],
                linewidth=2.2,
                label=(
                    "Realized coverage-failure fraction"
                ),
            )

        else:

            realized_line = None

        # ---------------------------------------------------------------------
        # EVENT ONSET
        # ---------------------------------------------------------------------

        ax_left.axvline(
            EVENT_ONSET,
            linestyle="-.",
            linewidth=1.5,
            label="Coverage-failure onset",
        )

        # ---------------------------------------------------------------------
        # FIRST SYSTEM RED
        # ---------------------------------------------------------------------

        if pd.notna(alert_date):

            ax_left.axvline(
                alert_date,
                linestyle=":",
                linewidth=1.8,
                label="First system RED",
            )

        # ---------------------------------------------------------------------
        # AXIS LIMITS
        # ---------------------------------------------------------------------

        probability_max = max(
            float(pre["mean_p"].max()),
            float(pre["mean_thr05"].max()),
        )

        # Give probability some headroom.
        probability_upper = max(
            0.01,
            probability_max * 1.20,
        )

        ax_left.set_ylim(
            0,
            probability_upper,
        )

        ax_right.set_ylim(
            0,
            1.02,
        )

        # ---------------------------------------------------------------------
        # LABELS
        # ---------------------------------------------------------------------

        ax_left.set_ylabel(
            "Predicted probability"
        )

        ax_right.set_ylabel(
            "Fraction of stocks"
        )

        ax_left.set_xlabel(
            "Forecast date"
        )

        ax_left.set_title(
            model,
            fontsize=13,
            fontweight="bold",
        )

        ax_left.grid(
            alpha=0.20
        )

        # ---------------------------------------------------------------------
        # COMBINED LEGEND
        # ---------------------------------------------------------------------

        handles_left, labels_left = (
            ax_left.get_legend_handles_labels()
        )

        handles_right, labels_right = (
            ax_right.get_legend_handles_labels()
        )

        handles = (
            handles_left
            + handles_right
        )

        labels = (
            labels_left
            + labels_right
        )

        seen = set()

        final_handles = []
        final_labels = []

        for h, label in zip(
            handles,
            labels,
        ):

            if label not in seen:

                seen.add(label)

                final_handles.append(
                    h
                )

                final_labels.append(
                    label
                )

        ax_left.legend(
            final_handles,
            final_labels,
            loc="upper left",
            fontsize=8,
            frameon=False,
        )

        # ---------------------------------------------------------------------
        # EX-ANTE / EX-POST ANNOTATION
        # ---------------------------------------------------------------------

        ax_left.text(
            0.98,
            0.95,
            (
                "Ex-ante forecast\n"
                "uses information through t"
            ),
            transform=ax_left.transAxes,
            ha="right",
            va="top",
            fontsize=8.5,
            bbox=dict(
                boxstyle="round",
                alpha=0.12,
            ),
        )

        # ---------------------------------------------------------------------
        # LEAD ANNOTATION
        # ---------------------------------------------------------------------

        if pd.notna(alert_date):

            ax_left.text(
                0.98,
                0.72,
                (
                    f"First RED: "
                    f"{alert_date.strftime('%Y-%m-%d')}\n"
                    f"Lead: "
                    f"{int(lead_days)} trading days"
                ),
                transform=ax_left.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox=dict(
                    boxstyle="round",
                    alpha=0.12,
                ),
            )

    # =========================================================================
    # TITLE
    # =========================================================================

    fig.suptitle(
        "COVID Episode: Ex-Ante Coverage-Failure Reliability Monitoring",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.965,
        (
            "S6 monitor | T2 = future 10-day coverage failure | "
            "20-trading-day pre-event window"
        ),
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.025,
        (
            "Left axis: predicted probability. "
            "Right axis: RED-stock fraction and realized coverage failure. "
            "Realized outcomes appear only after the event onset."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.055,
            1,
            0.94,
        ]
    )

    # =========================================================================
    # SAVE
    # =========================================================================

    fig.savefig(
        FIGURE_PNG,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGURE_PDF,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 110)
    print(
        "RS05 — FINAL COVID T2 PAPER FIGURE + TABLE"
    )
    print("=" * 110)

    print(
        f"T2 event onset: "
        f"{EVENT_ONSET.date()}"
    )

    print(
        "System RED threshold: "
        f"{SYSTEM_RED_THRESHOLD:.0%}"
    )

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    panel, lead, summary = load_data()

    panel = panel[
        panel["target"] == TARGET
    ].copy()

    lead = lead[
        lead["target"] == TARGET
    ].copy()

    summary = summary[
        summary["target"] == TARGET
    ].copy()

    # -------------------------------------------------------------------------
    # Model check
    # -------------------------------------------------------------------------

    actual_models = sorted(
        panel["model"].unique()
    )

    expected_models = sorted(
        MODELS
    )

    if actual_models != expected_models:
        raise RuntimeError(
            f"Expected models {expected_models}, "
            f"found {actual_models}"
        )

    # -------------------------------------------------------------------------
    # Paper table
    # -------------------------------------------------------------------------

    table = build_paper_table(
        lead,
        summary,
    )

    table.to_csv(
        TABLE_CSV,
        index=False,
    )

    # -------------------------------------------------------------------------
    # LaTeX
    # -------------------------------------------------------------------------

    latex_table = table.copy()

    latex_table["First system RED"] = (
        pd.to_datetime(
            latex_table["First system RED"]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )

    for col in [
        "P(coverage failure) at first RED",
        "P(coverage failure) at final pre-event date",
        "Maximum pre-event probability",
        "Realized coverage failure at onset",
        "Maximum post-onset coverage failure",
    ]:

        latex_table[col] = (
            latex_table[col]
            .map(
                lambda x:
                f"{x:.3f}"
            )
        )

    latex = latex_table.to_latex(
        index=False,
        escape=True,
    )

    TABLE_TEX.write_text(
        latex,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Figure
    # -------------------------------------------------------------------------

    make_figure(
        panel,
        lead,
    )

    # -------------------------------------------------------------------------
    # Console
    # -------------------------------------------------------------------------

    print()
    print(
        "FINAL T2 PAPER TABLE"
    )

    print(
        table.to_string(
            index=False
        )
    )

    print()
    print(
        f"PNG:\n{FIGURE_PNG}"
    )

    print(
        f"PDF:\n{FIGURE_PDF}"
    )

    print(
        f"CSV:\n{TABLE_CSV}"
    )

    print(
        f"LaTeX:\n{TABLE_TEX}"
    )

    print()
    print(
        "No retraining performed."
    )

    print(
        "No recalibration performed."
    )

    print(
        "No threshold tuning performed."
    )

    print(
        "T2 event onset fixed at 2020-02-12."
    )

    print(
        "System RED = >=25% of stocks in RED."
    )

    print()
    print(
        "DONE."
    )


if __name__ == "__main__":
    main()