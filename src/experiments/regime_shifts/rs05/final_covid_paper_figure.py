from __future__ import annotations

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

CASE_SUMMARY_FILE = (
    CASE_DIR
    / "RS05_COVID_corrected_summary_S6.csv"
)


# =============================================================================
# OUTPUT
# =============================================================================

FIGURE_PNG = (
    CASE_DIR
    / "RS05_COVID_T1_FINAL_PAPER_FIGURE_S6.png"
)

FIGURE_PDF = (
    CASE_DIR
    / "RS05_COVID_T1_FINAL_PAPER_FIGURE_S6.pdf"
)

TABLE_CSV = (
    CASE_DIR
    / "RS05_COVID_T1_FINAL_PAPER_TABLE_S6.csv"
)

TABLE_TEX = (
    CASE_DIR
    / "RS05_COVID_T1_FINAL_PAPER_TABLE_S6.tex"
)


# =============================================================================
# FIXED DESIGN
# =============================================================================

MODELS = [
    "EWMA",
    "GARCH-t",
    "HS",
    "QRF",
]

TARGET = "T1"

# Fixed descriptive COVID stress onset.
EVENT_ONSET = pd.Timestamp(
    "2020-03-16"
)

# Only show the 20 trading-day pre-event warning window,
# plus the subsequent stress period.
PRE_EVENT_TRADING_DAYS = 20

POST_EVENT_END = pd.Timestamp(
    "2020-05-31"
)

SYSTEM_RED_THRESHOLD = 0.25


# =============================================================================
# LOAD
# =============================================================================

def load_data():

    if not DATE_LEVEL_FILE.exists():
        raise FileNotFoundError(
            DATE_LEVEL_FILE
        )

    if not LEAD_TIME_FILE.exists():
        raise FileNotFoundError(
            LEAD_TIME_FILE
        )

    if not CASE_SUMMARY_FILE.exists():
        raise FileNotFoundError(
            CASE_SUMMARY_FILE
        )

    panel = pd.read_csv(
        DATE_LEVEL_FILE
    )

    lead = pd.read_csv(
        LEAD_TIME_FILE
    )

    summary = pd.read_csv(
        CASE_SUMMARY_FILE
    )

    panel["forecast_date"] = pd.to_datetime(
        panel["forecast_date"]
    )

    lead["event_onset"] = pd.to_datetime(
        lead["event_onset"]
    )

    lead["window_start"] = pd.to_datetime(
        lead["window_start"]
    )

    lead["window_end"] = pd.to_datetime(
        lead["window_end"]
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
# DETERMINE EXACT 20-DAY WINDOW
# =============================================================================

def get_pre_event_dates(
    panel: pd.DataFrame,
    model: str,
) -> np.ndarray:

    dates = np.sort(
        panel[
            (
                panel["target"]
                == TARGET
            )
            &
            (
                panel["model"]
                == model
            )
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

    table["first_system_red_alert"] = pd.to_datetime(
        table["first_system_red_alert"]
    )

    table["event_onset"] = pd.to_datetime(
        table["event_onset"]
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
                "P(deterioration) at first RED",

            "final_pre_event_probability":
                "P(deterioration) at final pre-event date",

            "max_pre_event_p":
                "Maximum pre-event probability",

            "final_pre_event_system_state":
                "Final system state",

            "event_day_realized_fraction":
                "Realized deterioration at onset",

            "max_post_event_realized_fraction":
                "Maximum post-onset deterioration",

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
        sharex=False,
        sharey=True,
    )

    axes = axes.flatten()

    for ax, model in zip(
        axes,
        MODELS,
    ):

        x = panel[
            (
                panel["target"]
                == TARGET
            )
            &
            (
                panel["model"]
                == model
            )
        ].copy()

        x = x.sort_values(
            "forecast_date"
        )

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
        # PRE-EVENT FORECASTS
        # ---------------------------------------------------------------------

        pre = x[
            (x["forecast_date"] >= pre_start)
            &
            (x["forecast_date"] < EVENT_ONSET)
        ].copy()

        # ---------------------------------------------------------------------
        # POST-EVENT REALIZED OUTCOME
        # ---------------------------------------------------------------------

        post = x[
            (x["forecast_date"] >= EVENT_ONSET)
            &
            (x["forecast_date"] <= POST_EVENT_END)
        ].copy()

        # ---------------------------------------------------------------------
        # Lead-time information
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
        # Light separation between ex-ante and ex-post areas
        # ---------------------------------------------------------------------

        ax.axvspan(
            pre_start,
            EVENT_ONSET,
            alpha=0.05,
        )

        ax.axvspan(
            EVENT_ONSET,
            POST_EVENT_END,
            alpha=0.025,
        )

        # ---------------------------------------------------------------------
        # Probability
        # ---------------------------------------------------------------------

        ax.plot(
            pre["forecast_date"],
            pre["mean_p"],
            linewidth=2.2,
            label="Predicted P(future deterioration)",
        )

        # ---------------------------------------------------------------------
        # System RED fraction
        # ---------------------------------------------------------------------

        ax.plot(
            pre["forecast_date"],
            pre["red_fraction"],
            linestyle="--",
            linewidth=1.7,
            label="Fraction of stocks RED",
        )

        # ---------------------------------------------------------------------
        # 25% system RED threshold
        # ---------------------------------------------------------------------

        ax.axhline(
            SYSTEM_RED_THRESHOLD,
            linestyle=":",
            linewidth=1.2,
            label="System RED threshold",
        )

        # ---------------------------------------------------------------------
        # Probability threshold lines
        #
        # Use the model-specific mean thresholds.
        # ---------------------------------------------------------------------

        ax.plot(
            pre["forecast_date"],
            pre["mean_thr20"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            label="thr20",
        )

        ax.plot(
            pre["forecast_date"],
            pre["mean_thr10"],
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
            label="thr10",
        )

        ax.plot(
            pre["forecast_date"],
            pre["mean_thr05"],
            linestyle=":",
            linewidth=0.9,
            alpha=0.6,
            label="thr05",
        )

        # ---------------------------------------------------------------------
        # Realized deterioration — ONLY after event onset.
        # ---------------------------------------------------------------------

        if not post.empty:

            ax.plot(
                post["forecast_date"],
                post["realized_fraction"],
                linewidth=2.2,
                label="Realized future deterioration",
            )

        # ---------------------------------------------------------------------
        # Fixed COVID event onset
        # ---------------------------------------------------------------------

        ax.axvline(
            EVENT_ONSET,
            linestyle="-.",
            linewidth=1.5,
            label="COVID stress onset",
        )

        # ---------------------------------------------------------------------
        # First system RED alert
        # ---------------------------------------------------------------------

        if pd.notna(alert_date):

            ax.axvline(
                alert_date,
                linestyle=":",
                linewidth=1.8,
                label="First system RED",
            )

        # ---------------------------------------------------------------------
        # Formatting
        # ---------------------------------------------------------------------

        ax.set_title(
            model,
            fontsize=13,
            fontweight="bold",
        )

        ax.set_ylim(
            0,
            1.02,
        )

        ax.set_ylabel(
            "Probability / fraction"
        )

        ax.set_xlabel(
            "Forecast date"
        )

        ax.grid(
            alpha=0.20
        )

        # ---------------------------------------------------------------------
        # Annotation
        # ---------------------------------------------------------------------

        if pd.notna(alert_date):

            text = (
                f"First RED: "
                f"{alert_date.strftime('%d %b %Y')}\n"
                f"Lead: "
                f"{int(lead_days)} trading days"
            )

        else:

            text = (
                "No system RED before onset"
            )

        ax.text(
            0.98,
            0.96,
            text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(
                boxstyle="round",
                alpha=0.12,
            ),
        )

    # -------------------------------------------------------------------------
    # One shared legend.
    # -------------------------------------------------------------------------

    handles, labels = axes[0].get_legend_handles_labels()

    # Remove duplicate labels while preserving order.
    seen = set()
    unique_handles = []
    unique_labels = []

    for h, label in zip(
        handles,
        labels,
    ):
        if label not in seen:
            seen.add(label)
            unique_handles.append(h)
            unique_labels.append(label)

    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.005),
        frameon=False,
    )

    fig.suptitle(
        "COVID Episode: Ex-Ante VaR Reliability Monitoring",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    fig.text(
        0.5,
        0.965,
        (
            "S6 monitor | T1 = future 10-day deterioration | "
            "pre-event forecasts use information available through each date"
        ),
        ha="center",
        fontsize=10,
    )

    fig.text(
        0.5,
        0.035,
        (
            "Dashed line = fraction of stocks in RED; "
            "solid probability is ex-ante; realized deterioration is shown only after the event onset."
        ),
        ha="center",
        fontsize=9,
    )

    plt.tight_layout(
        rect=[
            0,
            0.07,
            1,
            0.94,
        ]
    )

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
        "RS05 — FINAL COVID T1 PAPER FIGURE + TABLE"
    )
    print("=" * 110)

    panel, lead, summary = load_data()

    # -------------------------------------------------------------------------
    # Restrict to T1.
    # -------------------------------------------------------------------------

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
    # Check exact model coverage.
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
    # Make paper table.
    # -------------------------------------------------------------------------

    table = build_paper_table(
        lead,
        summary,
    )

    table.to_csv(
        TABLE_CSV,
        index=False,
    )

    # LaTeX-ready copy.
    table_tex = table.copy()

    table_tex["First system RED"] = (
        pd.to_datetime(
            table_tex["First system RED"]
        ).dt.strftime(
            "%Y-%m-%d"
        )
    )

    table_tex["P(deterioration) at first RED"] = (
        table_tex[
            "P(deterioration) at first RED"
        ]
        .map(
            lambda x:
            f"{x:.3f}"
        )
    )

    table_tex["P(deterioration) at final pre-event date"] = (
        table_tex[
            "P(deterioration) at final pre-event date"
        ]
        .map(
            lambda x:
            f"{x:.3f}"
        )
    )

    table_tex["Maximum pre-event probability"] = (
        table_tex[
            "Maximum pre-event probability"
        ]
        .map(
            lambda x:
            f"{x:.3f}"
        )
    )

    table_tex["Realized deterioration at onset"] = (
        table_tex[
            "Realized deterioration at onset"
        ]
        .map(
            lambda x:
            f"{x:.3f}"
        )
    )

    table_tex["Maximum post-onset deterioration"] = (
        table_tex[
            "Maximum post-onset deterioration"
        ]
        .map(
            lambda x:
            f"{x:.3f}"
        )
    )

    latex = table_tex.to_latex(
        index=False,
        escape=True,
    )

    TABLE_TEX.write_text(
        latex,
        encoding="utf-8",
    )

    # -------------------------------------------------------------------------
    # Figure.
    # -------------------------------------------------------------------------

    make_figure(
        panel,
        lead,
    )

    # -------------------------------------------------------------------------
    # Console output.
    # -------------------------------------------------------------------------

    print()
    print(
        "FINAL T1 PAPER TABLE"
    )
    print("-" * 110)

    print(
        table.to_string(
            index=False
        )
    )

    print()
    print(
        f"Figure PNG:\n{FIGURE_PNG}"
    )

    print(
        f"Figure PDF:\n{FIGURE_PDF}"
    )

    print(
        f"Table CSV:\n{TABLE_CSV}"
    )

    print(
        f"Table LaTeX:\n{TABLE_TEX}"
    )

    print()
    print(
        "Fixed COVID onset: "
        f"{EVENT_ONSET.date()}"
    )

    print(
        "System RED threshold: "
        f"{SYSTEM_RED_THRESHOLD:.0%}"
    )

    print(
        "No retraining/recalibration performed."
    )

    print(
        "DONE."
    )


if __name__ == "__main__":
    main()