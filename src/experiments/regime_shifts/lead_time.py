from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest


# =============================================================================
# PATHS
# =============================================================================

DEFAULT_PREDICTIONS = Path(
    r"D:\reliability_layer\data\monitoring\regime_shifts\RS05"
    r"\RS05_oos_predictions_H10_main.csv.gz"
)

OUT_DIR = Path(
    r"D:\reliability_layer\data\monitoring\regime_shifts\RS05"
)

# Headline analysis begins here, matching RS05a/RS05b.
HEADLINE_START = pd.Timestamp("2020-01-01")

# Outcome-defined failure event:
# fraction of 24 tickers with coverage_failure_Hd == 1.
MAIN_FAILURE_FRACTION = 0.50

# Sensitivity event definitions.
FAILURE_FRACTION_SENSITIVITIES = [0.25, 0.75]

# Merge event pieces separated by at most this many trading days.
MAX_GAP_DAYS = 5

# Look back this many trading dates before each event onset for an early warning.
PRE_ONSET_DAYS = 20

# Main operational alert rule:
# at least 25% of tickers for that VaR model are in the S6 10%-budget alert.
MAIN_ALERT_FRACTION = 0.25

# Sensitivity alert fractions.
ALERT_FRACTION_SENSITIVITIES = [0.10, 0.50]

# Main alert budget is thr10. We also report thr05 and thr20.
ALERT_BUDGET_COLS = {
    "5pct": "thr05",
    "10pct": "thr10",
    "20pct": "thr20",
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{path}"
        )

    df = pd.read_csv(
        path,
        compression="gzip",
        low_memory=False,
    )

    required = {
        "forecast_date",
        "ticker",
        "model",
        "target",
        "H",
        "spec",
        "algo",
        "y",
        "p_cal",
        "thr05",
        "thr10",
        "thr20",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"Prediction file missing required columns: {missing}"
        )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce",
    )
    df["y"] = pd.to_numeric(
        df["y"],
        errors="coerce",
    )
    df["p_cal"] = pd.to_numeric(
        df["p_cal"],
        errors="coerce",
    )

    for c in ["thr05", "thr10", "thr20"]:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "forecast_date",
            "y",
            "p_cal",
        ]
    ).copy()

    return df


# =============================================================================
# EVENT EXTRACTION
# =============================================================================

def build_failure_date_series(
    df: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    """
    Date-level fraction of tickers whose outcome y=1.

    Because the input is restricted to S6/T2, y represents the
    outcome-defined coverage-failure target for the selected horizon.
    """
    m = df[df["model"] == model_name].copy()

    if m.empty:
        return pd.DataFrame()

    daily = (
        m.groupby("forecast_date")
        .agg(
            n_tickers=("ticker", "nunique"),
            failure_fraction=("y", "mean"),
            mean_p=("p_cal", "mean"),
            median_p=("p_cal", "median"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    daily["complete_24_tickers"] = daily["n_tickers"].eq(24)

    return daily


def extract_events(
    daily: pd.DataFrame,
    failure_fraction: float,
    max_gap_days: int,
) -> pd.DataFrame:
    """
    Identify contiguous periods where the failure fraction exceeds the
    selected threshold, merging pieces separated by <= max_gap_days
    trading dates.

    This works entirely from the realized T2 outcome series and does not
    use the monitor probability to define the events.
    """
    if daily.empty:
        return pd.DataFrame()

    x = daily[
        daily["complete_24_tickers"]
        & (daily["forecast_date"] >= HEADLINE_START)
    ].copy()

    x["is_failure"] = (
        x["failure_fraction"] >= failure_fraction
    )

    event_rows: list[dict] = []
    i = 0
    event_id = 0

    while i < len(x):
        if not bool(x.iloc[i]["is_failure"]):
            i += 1
            continue

        event_id += 1
        start_i = i
        end_i = i

        # Extend through failures, allowing small gaps.
        while end_i + 1 < len(x):
            next_i = end_i + 1

            if bool(x.iloc[next_i]["is_failure"]):
                end_i = next_i
                continue

            # Count trading-date gaps from current date to the next date.
            gap = next_i - end_i - 1

            if gap <= max_gap_days:
                # Search ahead for another failure. If found within the
                # allowed gap, merge the pieces.
                found = None
                j = next_i + 1

                while j < len(x) and (j - next_i - 1) <= max_gap_days:
                    if bool(x.iloc[j]["is_failure"]):
                        found = j
                        break
                    j += 1

                if found is not None:
                    end_i = found
                    continue

            break

        block = x.iloc[start_i : end_i + 1].copy()

        event_rows.append(
            {
                "event_id": event_id,
                "start_date": block["forecast_date"].min(),
                "end_date": block["forecast_date"].max(),
                "n_dates": int(len(block)),
                "max_failure_fraction": float(
                    block["failure_fraction"].max()
                ),
                "mean_failure_fraction": float(
                    block["failure_fraction"].mean()
                ),
                "max_mean_p": float(
                    block["mean_p"].max()
                ),
                "n_dates_over_threshold": int(
                    block["is_failure"].sum()
                ),
            }
        )

        i = end_i + 1

    return pd.DataFrame(event_rows)


# =============================================================================
# EVENT-LEVEL ALERT / LEAD-TIME ANALYSIS
# =============================================================================

def add_alert_columns(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create date-level alert fractions using each day's calibrated S6
    probabilities and fold-frozen alert thresholds.
    """
    x = model_df.copy()

    for label, col in ALERT_BUDGET_COLS.items():
        x[f"alert_fraction_{label}"] = (
            x["p_cal"].to_numpy(dtype=float)
            >= x[col].to_numpy(dtype=float)
        )

    # Convert row-level booleans to date-level fractions.
    agg = {
        "n_tickers": ("ticker", "nunique"),
        "mean_p": ("p_cal", "mean"),
        "median_p": ("p_cal", "median"),
        "failure_fraction": ("y", "mean"),
        "alert_fraction_5pct": (
            "alert_fraction_5pct",
            "mean",
        ),
        "alert_fraction_10pct": (
            "alert_fraction_10pct",
            "mean",
        ),
        "alert_fraction_20pct": (
            "alert_fraction_20pct",
            "mean",
        ),
    }

    return (
        x.groupby("forecast_date")
        .agg(**agg)
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )


def trading_dates_between(
    date_series: pd.Series,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> int:
    dates = (
        pd.to_datetime(
            date_series,
            errors="coerce",
        )
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    window = dates[
        (dates >= start_date)
        & (dates <= end_date)
    ]

    return max(len(window) - 1, 0)


def analyze_events_for_model(
    model_name: str,
    model_predictions: pd.DataFrame,
    events: pd.DataFrame,
    pre_onset_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each outcome-defined failure event, identify first alerts in the
    20-trading-day pre-onset window and compute lead time.

    Main policy:
      alert if >=25% of tickers exceed each ticker's fold-frozen thr10.

    Sensitivities:
      alert if >=10% or >=50% exceed thr10.

    We also report analogous alert timing for thr05 and thr20.
    """
    date_level = add_alert_columns(
        model_predictions
    )

    all_dates = date_level["forecast_date"]

    result_rows: list[dict] = []
    alert_series_rows: list[dict] = []

    for _, event in events.iterrows():
        start = pd.Timestamp(event["start_date"])
        end = pd.Timestamp(event["end_date"])

        prior_dates = all_dates[
            all_dates < start
        ].drop_duplicates().sort_values()

        prior_dates = prior_dates.tail(
            pre_onset_days
        )

        window_dates = pd.concat(
            [
                prior_dates,
                pd.Series([start]),
            ],
            ignore_index=True,
        ).drop_duplicates().sort_values()

        window = date_level[
            date_level["forecast_date"].isin(
                window_dates
            )
        ].copy()

        row = {
            "model": model_name,
            "event_id": int(event["event_id"]),
            "failure_start": start,
            "failure_end": end,
            "failure_n_dates": int(event["n_dates"]),
            "failure_peak_fraction": float(
                event["max_failure_fraction"]
            ),
            "pre_onset_days": int(len(prior_dates)),
        }

        # First alerts for each probability budget and cross-sectional
        # aggregation threshold.
        for budget_label in ALERT_BUDGET_COLS:
            frac_col = (
                f"alert_fraction_{budget_label}"
            )

            for alert_frac in [
                MAIN_ALERT_FRACTION,
                *ALERT_FRACTION_SENSITIVITIES,
            ]:
                label = (
                    f"{budget_label}_cross{int(alert_frac * 100)}"
                )

                eligible = window[
                    window[frac_col] >= alert_frac
                ]

                if eligible.empty:
                    row[f"{label}_date"] = pd.NaT
                    row[f"{label}_lead_days"] = math.nan
                    row[f"{label}_hit"] = 0
                else:
                    first_alert = pd.Timestamp(
                        eligible["forecast_date"].min()
                    )

                    lead = trading_dates_between(
                        all_dates,
                        first_alert,
                        start,
                    )

                    row[f"{label}_date"] = first_alert
                    row[f"{label}_lead_days"] = (
                        lead
                    )
                    row[f"{label}_hit"] = 1

        result_rows.append(row)

        # Keep the date-level signal for plotting / diagnostics.
        tmp = window.copy()
        tmp["model"] = model_name
        tmp["event_id"] = int(event["event_id"])
        tmp["failure_start"] = start
        alert_series_rows.extend(
            tmp.to_dict("records")
        )

    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(alert_series_rows),
    )


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_event_hits(
    event_results: pd.DataFrame,
) -> pd.DataFrame:
    if event_results.empty:
        return pd.DataFrame()

    rows = []

    main_budget = "10pct"
    main_cross = int(
        MAIN_ALERT_FRACTION * 100
    )

    for model_name, g in event_results.groupby(
        "model"
    ):
        for cross_frac in [
            main_cross,
            *[
                int(x * 100)
                for x in ALERT_FRACTION_SENSITIVITIES
            ],
        ]:
            col = (
                f"{main_budget}_cross{cross_frac}_hit"
            )

            hits = g[col].astype(int)
            n = len(hits)
            k = int(hits.sum())

            if n:
                p_value = float(
                    binomtest(
                        k,
                        n,
                        p=0.5,
                        alternative="greater",
                    ).pvalue
                )
            else:
                p_value = math.nan

            lead_col = (
                f"{main_budget}_cross{cross_frac}_lead_days"
            )

            positive_leads = pd.to_numeric(
                g.loc[
                    g[col] == 1,
                    lead_col,
                ],
                errors="coerce",
            ).dropna()

            rows.append(
                {
                    "model": model_name,
                    "alert_budget": main_budget,
                    "alert_cross_section": cross_frac / 100.0,
                    "n_events": n,
                    "n_hits": k,
                    "hit_rate": (
                        k / n
                        if n
                        else math.nan
                    ),
                    "sign_test_p_gt_0.5": p_value,
                    "mean_lead_days": (
                        float(
                            positive_leads.mean()
                        )
                        if not positive_leads.empty
                        else math.nan
                    ),
                    "median_lead_days": (
                        float(
                            positive_leads.median()
                        )
                        if not positive_leads.empty
                        else math.nan
                    ),
                    "min_lead_days": (
                        float(
                            positive_leads.min()
                        )
                        if not positive_leads.empty
                        else math.nan
                    ),
                    "max_lead_days": (
                        float(
                            positive_leads.max()
                        )
                        if not positive_leads.empty
                        else math.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def find_false_alarm_runs(
    date_level: pd.DataFrame,
    events: pd.DataFrame,
    pre_onset_days: int,
    alert_threshold_fraction: float = MAIN_ALERT_FRACTION,
    budget_label: str = "10pct",
) -> pd.DataFrame:
    """
    Find contiguous alert runs outside the 20-day pre-onset window and
    failure event itself.

    This is a descriptive false-alarm measure, not a confirmatory test.
    """
    if date_level.empty:
        return pd.DataFrame()

    x = date_level.copy()

    alert_col = (
        f"alert_fraction_{budget_label}"
    )

    x["event_or_pre_event"] = False

    for _, event in events.iterrows():
        start = pd.Timestamp(event["start_date"])
        end = pd.Timestamp(event["end_date"])

        dates = x["forecast_date"]
        prior = dates[dates < start].drop_duplicates().sort_values().tail(
            pre_onset_days
        )

        mask = (
            x["forecast_date"].between(
                start,
                end,
                inclusive="both",
            )
            | x["forecast_date"].isin(prior)
        )

        x.loc[mask, "event_or_pre_event"] = True

    x["alert"] = (
        x[alert_col] >= alert_threshold_fraction
    )

    # A false alarm is an alert date outside event/pre-event windows.
    fa = x[
        x["alert"]
        & (~x["event_or_pre_event"])
    ][
        [
            "forecast_date",
            alert_col,
            "mean_p",
            "failure_fraction",
        ]
    ].copy()

    if fa.empty:
        return pd.DataFrame(
            columns=[
                "run_id",
                "start_date",
                "end_date",
                "n_dates",
                "max_alert_fraction",
                "max_mean_p",
                "max_failure_fraction",
            ]
        )

    fa = fa.sort_values(
        "forecast_date"
    ).reset_index(drop=True)

    rows = []
    start_idx = 0
    run_id = 0

    for i in range(1, len(fa)):
        gap = i - (i - 1) - 1

        # Consecutive rows in this filtered table correspond to consecutive
        # observed alert dates; use actual date spacing only as a diagnostic.
        date_gap = (
            fa.loc[i, "forecast_date"]
            - fa.loc[i - 1, "forecast_date"]
        ).days

        if date_gap > 7:
            block = fa.iloc[start_idx:i]
            run_id += 1
            rows.append(
                {
                    "run_id": run_id,
                    "start_date": block["forecast_date"].min(),
                    "end_date": block["forecast_date"].max(),
                    "n_dates": int(len(block)),
                    "max_alert_fraction": float(
                        block[
                            alert_col
                        ].max()
                    ),
                    "max_mean_p": float(
                        block["mean_p"].max()
                    ),
                    "max_failure_fraction": float(
                        block[
                            "failure_fraction"
                        ].max()
                    ),
                }
            )
            start_idx = i

    block = fa.iloc[start_idx:]
    run_id += 1
    rows.append(
        {
            "run_id": run_id,
            "start_date": block["forecast_date"].min(),
            "end_date": block["forecast_date"].max(),
            "n_dates": int(len(block)),
            "max_alert_fraction": float(
                block[alert_col].max()
            ),
            "max_mean_p": float(
                block["mean_p"].max()
            ),
            "max_failure_fraction": float(
                block["failure_fraction"].max()
            ),
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "RS05c: outcome-defined VaR failure event and early-warning "
            "lead-time analysis using RS05a S6 OOS probabilities."
        )
    )
    parser.add_argument(
        "--predictions",
        default=str(DEFAULT_PREDICTIONS),
    )
    parser.add_argument(
        "--models",
        default="ALL",
        help="ALL or comma-separated, e.g. HS,GARCH-t",
    )
    parser.add_argument(
        "--failure-fraction",
        type=float,
        default=MAIN_FAILURE_FRACTION,
        help="Main event onset threshold, default 0.50.",
    )
    parser.add_argument(
        "--max-gap-days",
        type=int,
        default=MAX_GAP_DAYS,
    )
    parser.add_argument(
        "--pre-onset-days",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    global_pre_onset_days = int(args.pre_onset_days)

    path = Path(args.predictions)
    df = load_predictions(path)

    # Main event analysis uses the actual sparse failure outcome and S6.
    p = df[
        (df["target"] == "T2")
        & (df["spec"] == "S6")
        & (df["algo"] == "logit")
        & (df["forecast_date"] >= HEADLINE_START)
    ].copy()

    if p.empty:
        raise RuntimeError(
            "No T2/S6/logit predictions found."
        )

    available_models = sorted(
        p["model"].astype(str).unique().tolist()
    )

    if args.models.upper() == "ALL":
        models = available_models
    else:
        models = [
            x.strip()
            for x in args.models.split(",")
            if x.strip()
        ]

        missing = [
            x for x in models
            if x not in available_models
        ]
        if missing:
            raise ValueError(
                f"Requested models missing: {missing}. "
                f"Available={available_models}"
            )

    print("=" * 110)
    print("RS05c OUTCOME-DEFINED VA R FAILURE / EARLY-WARNING STUDY")
    print("=" * 110)
    print(f"Input:                  {path}")
    print(f"Target:                 T2 / coverage_failure")
    print(f"Monitor:                S6 / logit")
    print(f"Headline start:         {HEADLINE_START.date()}")
    print(f"Failure onset fraction: {args.failure_fraction:.2f}")
    print(f"Gap merge:              {args.max_gap_days} trading dates")
    print(f"Pre-onset window:       {args.pre_onset_days} trading dates")
    print(f"Main alert:             >=25% tickers over thr10")
    print(f"Models:                 {models}")
    print()

    all_event_results = []
    all_series = []
    all_events = []
    all_false_alarm = []

    for model_name in models:
        print("=" * 110)
        print(f"MODEL: {model_name}")
        print("=" * 110)

        m = p[p["model"] == model_name].copy()

        # Sanity check: one observation per ticker/date.
        key = [
            "forecast_date",
            "ticker",
        ]
        duplicates = int(
            m.duplicated(key).sum()
        )

        if duplicates:
            raise ValueError(
                f"{model_name}: {duplicates:,} duplicate "
                f"ticker/date rows in S6 T2 predictions."
            )

        daily = build_failure_date_series(
            m,
            model_name,
        )

        print(
            f"Dates: {len(daily):,} | "
            f"Complete 24-ticker dates: "
            f"{daily['complete_24_tickers'].sum():,}"
        )

        # Main event threshold.
        events = extract_events(
            daily,
            failure_fraction=args.failure_fraction,
            max_gap_days=args.max_gap_days,
        )

        print(
            f"Main outcome-defined failure events: "
            f"{len(events)}"
        )

        if not events.empty:
            print(events.to_string(index=False))

        # Sensitivity event thresholds.
        for sens in FAILURE_FRACTION_SENSITIVITIES:
            sens_events = extract_events(
                daily,
                failure_fraction=sens,
                max_gap_days=args.max_gap_days,
            )

            print(
                f"Sensitivity failure threshold {sens:.2f}: "
                f"{len(sens_events)} events"
            )

            if not sens_events.empty:
                sens_events = sens_events.copy()
                sens_events["event_threshold"] = sens
                sens_events["model"] = model_name
                all_events.append(sens_events)

        if not events.empty:
            tmp_events = events.copy()
            tmp_events["event_threshold"] = args.failure_fraction
            tmp_events["model"] = model_name
            all_events.append(tmp_events)

            PRE_ONSET_DAYS = global_pre_onset_days

            er, series = analyze_events_for_model(
                model_name=model_name,
                model_predictions=m,
                events=events,
                pre_onset_days=global_pre_onset_days,
            )

            if not er.empty:
                all_event_results.append(er)

            if not series.empty:
                all_series.append(series)

        date_level = add_alert_columns(m)

        fa = find_false_alarm_runs(
            date_level,
            events,
            pre_onset_days=global_pre_onset_days,
            alert_threshold_fraction=MAIN_ALERT_FRACTION,
            budget_label="10pct",
        )

        if not fa.empty:
            fa["model"] = model_name
            all_false_alarm.append(fa)

    # -------------------------------------------------------------------------
    # SAVE
    # -------------------------------------------------------------------------

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    events_df = (
        pd.concat(
            all_events,
            ignore_index=True,
        )
        if all_events
        else pd.DataFrame()
    )

    event_results_df = (
        pd.concat(
            all_event_results,
            ignore_index=True,
        )
        if all_event_results
        else pd.DataFrame()
    )

    series_df = (
        pd.concat(
            all_series,
            ignore_index=True,
        )
        if all_series
        else pd.DataFrame()
    )

    false_alarm_df = (
        pd.concat(
            all_false_alarm,
            ignore_index=True,
        )
        if all_false_alarm
        else pd.DataFrame()
    )

    summary_df = summarize_event_hits(
        event_results_df
    )

    events_path = (
        OUT_DIR
        / "RS05c_outcome_defined_failure_events_H10.csv"
    )

    results_path = (
        OUT_DIR
        / "RS05c_event_leadtime_H10.csv"
    )

    series_path = (
        OUT_DIR
        / "RS05c_event_daily_series_H10.csv"
    )

    summary_path = (
        OUT_DIR
        / "RS05c_event_leadtime_summary_H10.csv"
    )

    false_alarm_path = (
        OUT_DIR
        / "RS05c_false_alarm_runs_H10.csv"
    )

    events_df.to_csv(
        events_path,
        index=False,
    )

    event_results_df.to_csv(
        results_path,
        index=False,
    )

    series_df.to_csv(
        series_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    false_alarm_df.to_csv(
        false_alarm_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # PRINT MAIN RESULTS
    # -------------------------------------------------------------------------

    print()
    print("=" * 110)
    print("RS05c MAIN EVENT RESULTS")
    print("=" * 110)

    if summary_df.empty:
        print(
            "No outcome-defined failure events were detected "
            "at the selected threshold."
        )
    else:
        print(
            summary_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.6g}",
            )
        )

    print()
    print("=" * 110)
    print("RS05c OUTPUTS")
    print("=" * 110)
    print(events_path)
    print(results_path)
    print(series_path)
    print(summary_path)
    print(false_alarm_path)

    print()
    print("IMPORTANT INTERPRETATION:")
    print(
        "Events are defined from realized T2 coverage failures, "
        "not from VIX/RV regimes or monitor probabilities."
    )
    print(
        "A lead of 0 means the first alert occurs on the failure onset date."
    )
    print(
        "A positive lead means the monitor crossed the alert rule "
        "before the outcome-defined failure onset."
    )
    print(
        "Event-level results are descriptive/low-power; the earlier "
        "HAC/block-bootstrap RS05b tests remain the confirmatory "
        "predictive-performance evidence."
    )


if __name__ == "__main__":
    main()
