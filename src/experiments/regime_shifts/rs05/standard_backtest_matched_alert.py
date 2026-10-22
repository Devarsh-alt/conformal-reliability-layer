from __future__ import annotations

"""
RS05H — PRACTITIONER-STYLE VaR BACKTESTING BASELINES
====================================================

Purpose
-------
Test whether the proposed RS05 reliability monitor adds information beyond
standard exception-based VaR backtesting diagnostics.

Signals
-------
1) Rolling Kupiec / unconditional-coverage (UC) test
2) Rolling Christoffersen conditional-coverage (CC) test
3) Rolling DQ-style lagged-hit diagnostic
4) Alpha-adjusted traffic-light-style exception-count signal
5) Optional VIX-only signal for the same matched-alert comparison

Important methodological point
------------------------------
The classical Basel traffic-light thresholds (0-4 / 5-9 / >=10 exceptions)
were designed for 99% VaR, not 95% VaR. We therefore DO NOT apply those
literal thresholds as the primary benchmark.

Instead we use an alpha-adjusted traffic-light-style rule for 5% VaR over
a 250-observation rolling window. The GREEN/AMBER/RED cutoffs are obtained
from exact Binomial(250, 0.05) upper quantiles:

    GREEN : exceptions <= q_0.95  = 18
    AMBER : 19 .. q_0.99 = 21
    RED   : exceptions >= 22

The raw exception fraction remains the continuous signal used to enforce
the same matched alert budget.

Causality
---------
For a forecast made at date t:
- all historical exception diagnostics use realized breaches known by t
- no t+1 or future outcome is used
- test-fold thresholds for the statistical signals are calibrated only from
  the preceding 252 forecast dates
- event outcomes are used ONLY for evaluation

Evaluation
----------
Same outcome-defined event framework as the existing RS05 matched-alert
study:
- H=10
- T2 coverage failure
- daily event if >=50% of the 24 stocks fail
- 20-trading-day pre-event window
- system alert if >=25% stocks are alerted
- nominal matched alert budget = 10% of ticker-days

Outputs
-------
Written to:
D:\reliability_layer\data\monitoring\regime_shifts\RS05

This script is deliberately independent of the RS05 monitor predictions for
signal construction. The existing RS05 prediction file is used only to
obtain the common T2 outcome labels and their forecast-date universe.
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2, binom, binomtest


# =============================================================================
# PATHS / CONFIG
# =============================================================================

BASE = Path(r"D:\reliability_layer")
RS_DIR = BASE / "data" / "monitoring" / "regime_shifts"
RS05_DIR = RS_DIR / "RS05"

PRED_PATH = RS05_DIR / "RS05_oos_predictions_H10_main.csv.gz"
QUAR_PATH = RS_DIR / "RS04_quarantined_outcomes.csv"

OUT_DIR = RS05_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADLINE_START = pd.Timestamp("2020-01-01")

ALPHA = 0.05
H = 10
ROLLING_N = 250
CAL_LEN = 252
MIN_HISTORY = 250
MATCHED_BUDGET = 0.10
CROSS_SECTION = 0.25
FAILURE_FRACTION = 0.50
PRE_ONSET_DATES = 20
MAX_GAP_TRADING_DATES = 5

MODELS = ["EWMA", "GARCH-t", "HS", "QRF"]

# DQ-style lag order.
DQ_LAGS = 5


# =============================================================================
# BASIC HELPERS
# =============================================================================

def logit_like_prob(x: float) -> float:
    return min(max(float(x), 1e-12), 1.0 - 1e-12)


def emit(msg: str = "") -> None:
    print(msg, flush=True)


def require_cols(df: pd.DataFrame, cols: list[str], label: str) -> None:
    missing = sorted(set(cols) - set(df.columns))
    if missing:
        raise KeyError(f"{label} missing required columns: {missing}")


# =============================================================================
# COMMON RS05 OUTCOME / FOLD DATA
# =============================================================================

def load_prediction_outcomes() -> pd.DataFrame:
    if not PRED_PATH.exists():
        raise FileNotFoundError(PRED_PATH)

    p = pd.read_csv(PRED_PATH, compression="gzip", low_memory=False)

    require_cols(
        p,
        ["forecast_date", "ticker", "model", "target", "H", "algo", "y"],
        "RS05 prediction file",
    )

    p["forecast_date"] = pd.to_datetime(p["forecast_date"], errors="coerce")
    p["ticker"] = p["ticker"].astype(str).str.upper().str.strip()
    p["model"] = p["model"].astype(str).str.strip()
    p["target"] = p["target"].astype(str).str.strip()
    p["algo"] = p["algo"].astype(str).str.strip()
    p["H"] = pd.to_numeric(p["H"], errors="coerce")
    p["y"] = pd.to_numeric(p["y"], errors="coerce")

    p = p[
        (p["target"] == "T2")
        & (p["H"] == H)
        & (p["algo"] == "logit")
        & (p["forecast_date"] >= HEADLINE_START)
        & (p["model"].isin(MODELS))
    ].copy()

    p = p.dropna(
        subset=["forecast_date", "ticker", "model", "y"]
    )

    # Signal evaluation outcome must be unique at ticker/model/date.
    p = (
        p.drop_duplicates(
            ["forecast_date", "ticker", "model"]
        )
        .sort_values(
            ["model", "ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    if p.empty:
        raise RuntimeError(
            "No T2/H10/logit rows found in the RS05 main prediction file."
        )

    # Every model should have the same date/ticker outcome panel.
    counts = (
        p.groupby("model")
        .agg(
            n_rows=("ticker", "size"),
            n_dates=("forecast_date", "nunique"),
            n_tickers=("ticker", "nunique"),
        )
        .reset_index()
    )

    if not (counts["n_tickers"] == 24).all():
        raise AssertionError(
            "Expected 24 tickers per model. Counts:\n"
            f"{counts.to_string(index=False)}"
        )

    return p


def load_quarantine_breaches() -> pd.DataFrame:
    if not QUAR_PATH.exists():
        raise FileNotFoundError(QUAR_PATH)

    q = pd.read_csv(QUAR_PATH, low_memory=False)

    require_cols(
        q,
        [
            "forecast_date",
            "realized_date",
            "ticker",
            "model_name",
            "outcome_next_day_breach",
        ],
        "RS04 quarantine",
    )

    q["forecast_date"] = pd.to_datetime(q["forecast_date"], errors="coerce")
    q["realized_date"] = pd.to_datetime(q["realized_date"], errors="coerce")
    q["ticker"] = q["ticker"].astype(str).str.upper().str.strip()
    q["model_name"] = q["model_name"].astype(str).str.strip()
    q["outcome_next_day_breach"] = pd.to_numeric(
        q["outcome_next_day_breach"], errors="coerce"
    )

    q = q[
        q["model_name"].isin(MODELS)
    ].dropna(
        subset=[
            "forecast_date",
            "realized_date",
            "ticker",
            "model_name",
            "outcome_next_day_breach",
        ]
    ).copy()

    q = (
        q.drop_duplicates(
            ["forecast_date", "ticker", "model_name"]
        )
        .sort_values(
            ["model_name", "ticker", "realized_date"]
        )
        .reset_index(drop=True)
    )

    # A breach evaluated on realized date d is known at d.
    q["breach"] = q["outcome_next_day_breach"].astype(int)

    return q


def make_breach_panel(q: pd.DataFrame) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Returns one realized-date breach series per (model, ticker).
    """
    panel = {}

    for (model, ticker), g in q.groupby(
        ["model_name", "ticker"],
        sort=True,
    ):
        x = (
            g[["realized_date", "breach"]]
            .drop_duplicates("realized_date")
            .sort_values("realized_date")
            .reset_index(drop=True)
        )

        if len(x) < MIN_HISTORY:
            continue

        panel[(model, ticker)] = x

    if len(panel) != 96:
        raise AssertionError(
            f"Expected 96 model/ticker breach series, got {len(panel)}"
        )

    return panel


def build_trading_calendar(outcomes: pd.DataFrame) -> pd.DatetimeIndex:
    dates = (
        pd.to_datetime(outcomes["forecast_date"])
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    return pd.DatetimeIndex(dates)


# =============================================================================
# ROLLING TEST STATISTICS
# =============================================================================

def kupiec_pvalue(hits: np.ndarray, alpha: float = ALPHA) -> float:
    """
    Exact binomial two-sided p-value for the UC null.

    This is more numerically stable for a sparse 5% breach process than
    directly evaluating an LR expression when x=0 or x=n.
    """
    n = int(len(hits))
    x = int(np.sum(hits))

    if n == 0:
        return np.nan

    return float(
        binomtest(
            x,
            n,
            p=alpha,
            alternative="two-sided",
        ).pvalue
    )


def christoffersen_cc_pvalue(
    hits: np.ndarray,
    alpha: float = ALPHA,
) -> float:
    """
    Christoffersen conditional-coverage p-value.

    LR_CC = LR_UC + LR_IND
    where the independence component uses the two-state Markov transition
    likelihood.
    """
    h = np.asarray(hits, dtype=int)

    if len(h) < 2:
        return np.nan

    n = len(h)
    x = int(h.sum())

    # UC likelihood.
    pi0 = alpha
    phat = x / n

    def ll_bernoulli(successes: int, total: int, p: float) -> float:
        p = min(max(float(p), 1e-12), 1.0 - 1e-12)
        s = int(successes)
        f = int(total - successes)
        return s * math.log(p) + f * math.log(1.0 - p)

    ll0 = ll_bernoulli(x, n, pi0)
    ll1 = ll_bernoulli(x, n, phat)

    lr_uc = max(0.0, 2.0 * (ll1 - ll0))

    # Transition counts.
    prev = h[:-1]
    curr = h[1:]

    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))

    n0 = n00 + n01
    n1 = n10 + n11

    pi01 = n01 / n0 if n0 else 0.0
    pi11 = n11 / n1 if n1 else 0.0

    ll_ind_null = ll_bernoulli(
        n01 + n11,
        n00 + n01 + n10 + n11,
        (n01 + n11) / max(n0 + n1, 1),
    )

    ll_ind_alt = (
        ll_bernoulli(n01, n0, pi01)
        + ll_bernoulli(n11, n1, pi11)
    )

    lr_ind = max(
        0.0,
        2.0 * (ll_ind_alt - ll_ind_null),
    )

    lr_cc = lr_uc + lr_ind

    return float(
        chi2.sf(
            lr_cc,
            df=2,
        )
    )


def dq_style_pvalue(
    hits: np.ndarray,
    alpha: float = ALPHA,
    lags: int = DQ_LAGS,
) -> float:
    """
    Engle–Manganelli-style DQ diagnostic using an intercept and lagged
    violation indicators.

    This implementation intentionally uses ONLY the historical hit process.
    It is therefore a conservative "lagged-hit DQ-style" baseline rather
    than a feature-rich DQ regression.

    DQ statistic:
        n * R^2
    compared with chi-square(df = number of regressors).

    A low p-value means the historical exception process contains
    statistically detectable dependence/inconsistency.
    """
    h = np.asarray(hits, dtype=float)

    if len(h) <= lags + 10:
        return np.nan

    y = h[lags:] - alpha
    X_parts = [np.ones(len(y))]

    for k in range(1, lags + 1):
        X_parts.append(h[lags - k : -k])

    X = np.column_stack(X_parts)

    # Drop rows with non-finite values, though the construction should be
    # finite.
    finite = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y = y[finite]
    X = X[finite]

    n = len(y)
    if n <= X.shape[1] + 2:
        return np.nan

    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        fitted = X @ beta

        sse = float(np.sum((y - fitted) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))

        if sst <= 1e-14:
            return np.nan

        r2 = max(0.0, min(1.0, 1.0 - sse / sst))
        stat = n * r2

        return float(
            chi2.sf(
                stat,
                df=X.shape[1],
            )
        )
    except np.linalg.LinAlgError:
        return np.nan


def traffic_light_category(
    exception_count: int,
    n: int = ROLLING_N,
    alpha: float = ALPHA,
) -> str:
    """
    Alpha-adjusted traffic-light-style category.

    For the fixed 250-observation / 5% setting:
       q95 = 18
       q99 = 21

    We compute the quantiles rather than hard-code the values, which also
    makes the function safe if ROLLING_N / alpha are changed.
    """
    q95 = int(binom.ppf(0.95, n, alpha))
    q99 = int(binom.ppf(0.99, n, alpha))

    if exception_count <= q95:
        return "GREEN"
    if exception_count <= q99:
        return "AMBER"
    return "RED"


# =============================================================================
# BUILD DAILY BACKTEST SCORES
# =============================================================================

def build_scores_for_series(
    series: pd.DataFrame,
    evaluation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    For every forecast date t, compute rolling diagnostics using breaches
    observed through t.

    The historical breach series is on realized dates. We therefore select
    observations with realized_date <= t.

    Returns one row per evaluation date.
    """
    s = series.copy().sort_values("realized_date").reset_index(drop=True)

    rows = []

    for d in evaluation_dates:
        hist = s[
            s["realized_date"] <= d
        ].tail(ROLLING_N)

        if len(hist) < MIN_HISTORY:
            continue

        hits = hist["breach"].to_numpy(dtype=int)
        x = int(hits.sum())

        p_uc = kupiec_pvalue(hits)
        p_cc = christoffersen_cc_pvalue(hits)
        p_dq = dq_style_pvalue(hits)

        def score_from_p(p: float) -> float:
            if not np.isfinite(p):
                return np.nan
            return float(-math.log10(max(p, 1e-12)))

        rows.append(
            {
                "forecast_date": d,
                "kupiec_p": p_uc,
                "kupiec_score": score_from_p(p_uc),
                "christoffersen_p": p_cc,
                "christoffersen_score": score_from_p(p_cc),
                "dq_p": p_dq,
                "dq_score": score_from_p(p_dq),
                "exception_count_250": x,
                "exception_rate_250": x / ROLLING_N,
                "traffic_light": traffic_light_category(x),
                "traffic_score": x / ROLLING_N,
            }
        )

    return pd.DataFrame(rows)


def build_all_scores(
    breach_panel: dict[tuple[str, str], pd.DataFrame],
    evaluation_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    pieces = []

    for (model, ticker), s in breach_panel.items():
        d = build_scores_for_series(
            s,
            evaluation_dates,
        )

        if d.empty:
            continue

        d["model"] = model
        d["ticker"] = ticker

        pieces.append(d)

        emit(
            f"  {model:7s} {ticker:5s}: "
            f"{len(d):,} diagnostic dates"
        )

    if not pieces:
        raise RuntimeError("No rolling backtest diagnostics were generated.")

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    out = out.sort_values(
        ["model", "ticker", "forecast_date"]
    ).reset_index(drop=True)

    return out


# =============================================================================
# FOLD-SPECIFIC ALERT THRESHOLDS
# =============================================================================

def quarterly_folds(
    dates: pd.DatetimeIndex,
) -> list[tuple[str, pd.DatetimeIndex, pd.Timestamp]]:
    idx = pd.DatetimeIndex(
        dates.drop_duplicates().sort_values()
    )

    periods = pd.PeriodIndex(idx, freq="Q")

    out = []

    for q in sorted(periods.unique()):
        test_dates = idx[periods == q]

        if len(test_dates) < 10:
            continue

        out.append(
            (
                str(q),
                test_dates,
                pd.Timestamp(test_dates.min()),
            )
        )

    return out


def calibration_thresholds(
    score_df: pd.DataFrame,
    score_col: str,
    test_dates: pd.DatetimeIndex,
) -> dict[pd.Timestamp, float]:
    """
    One fold-specific threshold calibrated from the preceding CAL_LEN
    forecast dates. We use >= for alerts after calibration.

    Because statistical p-value scores are discrete/tied, the realized
    alert rate can differ from exactly 10%. This is reported rather than
    hidden.
    """
    result = {}

    all_dates = pd.DatetimeIndex(
        score_df["forecast_date"]
        .drop_duplicates()
        .sort_values()
    )

    for q, q_dates, q_start in quarterly_folds(test_dates):
        eligible = all_dates[
            all_dates < q_start
        ]

        if len(eligible) < CAL_LEN:
            continue

        cal_dates = eligible[-CAL_LEN:]

        cal = score_df[
            score_df["forecast_date"].isin(cal_dates)
        ][score_col].dropna()

        if len(cal) < 100:
            continue

        # Target the upper 10% of historical severity.
        threshold = float(
            cal.quantile(1.0 - MATCHED_BUDGET)
        )

        for d in q_dates:
            result[pd.Timestamp(d)] = threshold

    return result


def apply_fold_thresholds(
    score_df: pd.DataFrame,
    score_col: str,
) -> pd.DataFrame:
    out = score_df.copy()

    threshold_map = {}
    for _, model_block in out.groupby("model"):
        test_dates = pd.DatetimeIndex(
            sorted(
                model_block["forecast_date"]
                .dropna()
                .drop_duplicates()
            )
        )

        local = calibration_thresholds(
            score_df=model_block,
            score_col=score_col,
            test_dates=test_dates,
        )
        threshold_map.update(local)

    out["threshold"] = out["forecast_date"].map(threshold_map)

    out["alert"] = (
        out[score_col].notna()
        & out["threshold"].notna()
        & (out[score_col] >= out["threshold"])
    )

    return out


# =============================================================================
# EVENT DEFINITION / MATCHED ALERT ANALYSIS
# =============================================================================

def build_failure_daily(outcomes: pd.DataFrame, model: str) -> pd.DataFrame:
    x = outcomes[outcomes["model"] == model].copy()

    daily = (
        x.groupby("forecast_date")
        .agg(
            n_tickers=("ticker", "nunique"),
            failure_fraction=("y", "mean"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    daily["complete_24_tickers"] = daily["n_tickers"].eq(24)
    return daily


def extract_events(
    daily: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    x = daily[
        daily["complete_24_tickers"]
        & (daily["forecast_date"] >= HEADLINE_START)
    ].copy()

    x["is_failure"] = x["failure_fraction"] >= threshold

    events = []
    i = 0
    event_id = 0

    while i < len(x):
        if not bool(x.iloc[i]["is_failure"]):
            i += 1
            continue

        event_id += 1
        start_i = i
        end_i = i

        while end_i + 1 < len(x):
            next_i = end_i + 1

            if bool(x.iloc[next_i]["is_failure"]):
                end_i = next_i
                continue

            gap = next_i - end_i - 1

            if gap <= MAX_GAP_TRADING_DATES:
                found = None
                j = next_i + 1

                while j < len(x) and (j - next_i - 1) <= MAX_GAP_TRADING_DATES:
                    if bool(x.iloc[j]["is_failure"]):
                        found = j
                        break
                    j += 1

                if found is not None:
                    end_i = found
                    continue

            break

        block = x.iloc[start_i : end_i + 1].copy()

        events.append(
            {
                "event_id": event_id,
                "start_date": block["forecast_date"].min(),
                "end_date": block["forecast_date"].max(),
                "n_dates": int(len(block)),
                "max_failure_fraction": float(block["failure_fraction"].max()),
                "mean_failure_fraction": float(block["failure_fraction"].mean()),
            }
        )

        i = end_i + 1

    return pd.DataFrame(events)


def trading_lead(
    all_dates: pd.Series | pd.DatetimeIndex,
    alert_date: pd.Timestamp,
    event_date: pd.Timestamp,
) -> int:
    dates = pd.DatetimeIndex(
        pd.to_datetime(pd.Series(all_dates), errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    window = dates[
        (dates >= alert_date)
        & (dates <= event_date)
    ]

    return max(len(window) - 1, 0)


def signal_daily(
    x: pd.DataFrame,
) -> pd.DataFrame:
    daily = (
        x.groupby("forecast_date")
        .agg(
            n_tickers=("ticker", "nunique"),
            alert_fraction=("alert", "mean"),
            mean_score=("signal_score", "mean"),
        )
        .reset_index()
        .sort_values("forecast_date")
        .reset_index(drop=True)
    )

    daily["complete_24_tickers"] = daily["n_tickers"].eq(24)
    return daily


def analyze_signal(
    signal_name: str,
    model: str,
    daily_signal: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = daily_signal["forecast_date"]

    event_rows = []
    protected_dates: set[pd.Timestamp] = set()

    for _, event in events.iterrows():
        start = pd.Timestamp(event["start_date"])
        end = pd.Timestamp(event["end_date"])

        prior = (
            all_dates[all_dates < start]
            .drop_duplicates()
            .sort_values()
            .tail(PRE_ONSET_DATES)
        )

        for d in prior:
            protected_dates.add(pd.Timestamp(d))
        for d in pd.date_range(start, end, freq="B"):
            # Only actual trading dates in the signal panel will be used below.
            protected_dates.add(pd.Timestamp(d))

        window_dates = pd.concat(
            [
                prior,
                pd.Series([start]),
            ],
            ignore_index=True,
        ).drop_duplicates().sort_values()

        window = daily_signal[
            daily_signal["forecast_date"].isin(window_dates)
        ].copy()

        eligible = window[
            window["alert_fraction"] >= CROSS_SECTION
        ]

        if eligible.empty:
            first_alert = pd.NaT
            lead = np.nan
            hit = 0
        else:
            first_alert = pd.Timestamp(
                eligible["forecast_date"].min()
            )
            lead = trading_lead(
                all_dates,
                first_alert,
                start,
            )
            hit = int(first_alert < start)

        event_rows.append(
            {
                "model": model,
                "signal": signal_name,
                "event_id": int(event["event_id"]),
                "failure_start": start,
                "failure_end": end,
                "failure_peak_fraction": float(event["max_failure_fraction"]),
                "first_alert_date": first_alert,
                "hit": hit,
                "lead_days": lead,
            }
        )

    # False alarm runs outside event/pre-event windows.
    work = daily_signal.copy()

    # Use the precise event/pre-event windows to exclude alerts.
    protected = set()
    for _, event in events.iterrows():
        start = pd.Timestamp(event["start_date"])
        end = pd.Timestamp(event["end_date"])
        prior = (
            all_dates[all_dates < start]
            .drop_duplicates()
            .sort_values()
            .tail(PRE_ONSET_DATES)
        )
        protected.update(pd.Timestamp(d) for d in prior)
        protected.update(
            pd.Timestamp(d)
            for d in all_dates[
                (all_dates >= start) & (all_dates <= end)
            ]
        )

    work["protected"] = work["forecast_date"].map(
        lambda d: pd.Timestamp(d) in protected
    )
    work["system_alert"] = work["alert_fraction"] >= CROSS_SECTION

    fa = work[
        work["system_alert"]
        & (~work["protected"])
    ].copy()

    false_runs = []

    if not fa.empty:
        fa = fa.sort_values("forecast_date").reset_index(drop=True)
        start_idx = 0
        run_id = 0

        for i in range(1, len(fa)):
            if (
                fa.loc[i, "forecast_date"]
                - fa.loc[i - 1, "forecast_date"]
            ).days > 7:
                block = fa.iloc[start_idx:i]
                run_id += 1

                false_runs.append(
                    {
                        "model": model,
                        "signal": signal_name,
                        "run_id": run_id,
                        "start_date": block["forecast_date"].min(),
                        "end_date": block["forecast_date"].max(),
                        "n_dates": int(len(block)),
                        "max_alert_fraction": float(
                            block["alert_fraction"].max()
                        ),
                        "max_mean_score": float(
                            block["mean_score"].max()
                        ),
                    }
                )

                start_idx = i

        block = fa.iloc[start_idx:]
        run_id += 1

        false_runs.append(
            {
                "model": model,
                "signal": signal_name,
                "run_id": run_id,
                "start_date": block["forecast_date"].min(),
                "end_date": block["forecast_date"].max(),
                "n_dates": int(len(block)),
                "max_alert_fraction": float(
                    block["alert_fraction"].max()
                ),
                "max_mean_score": float(
                    block["mean_score"].max()
                ),
            }
        )

    return (
        pd.DataFrame(event_rows),
        pd.DataFrame(false_runs),
    )


def summarize_events(
    event_results: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    if event_results.empty:
        return pd.DataFrame()

    for (model, signal), g in event_results.groupby(
        ["model", "signal"]
    ):
        n = len(g)
        hits = int(g["hit"].sum())

        leads = pd.to_numeric(
            g.loc[g["hit"] == 1, "lead_days"],
            errors="coerce",
        ).dropna()

        rows.append(
            {
                "model": model,
                "signal": signal,
                "n_events": n,
                "n_hits": hits,
                "hit_rate": hits / n if n else np.nan,
                "sign_test_p_gt_0.5": (
                    float(
                        binomtest(
                            hits,
                            n,
                            p=0.5,
                            alternative="greater",
                        ).pvalue
                    )
                    if n
                    else np.nan
                ),
                "mean_lead_days": (
                    float(leads.mean())
                    if len(leads)
                    else np.nan
                ),
                "median_lead_days": (
                    float(leads.median())
                    if len(leads)
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--models",
        default="ALL",
        help="ALL or comma-separated: EWMA,GARCH-t,HS,QRF",
    )

    args = parser.parse_args()

    outcomes = load_prediction_outcomes()
    q = load_quarantine_breaches()
    breach_panel = make_breach_panel(q)

    if args.models.upper() == "ALL":
        models = MODELS.copy()
    else:
        models = [
            x.strip()
            for x in args.models.split(",")
            if x.strip()
        ]

        missing = sorted(set(models) - set(MODELS))
        if missing:
            raise ValueError(f"Unknown models: {missing}")

    forecast_dates = build_trading_calendar(outcomes)

    emit("=" * 112)
    emit("RS05H — STANDARD VA R BACKTESTING BASELINES")
    emit("=" * 112)
    emit(f"Outcome file: {PRED_PATH}")
    emit(f"Breach history: {QUAR_PATH}")
    emit(f"Rolling window: {ROLLING_N} observations")
    emit(f"VaR alpha: {ALPHA:.2%}")
    emit(f"Matched alert budget: {MATCHED_BUDGET:.0%}")
    emit(f"System alert threshold: >= {CROSS_SECTION:.0%} stocks")
    emit(
        "Traffic light: "
        f"GREEN <= {int(binom.ppf(.95, ROLLING_N, ALPHA))}, "
        f"AMBER <= {int(binom.ppf(.99, ROLLING_N, ALPHA))}, "
        f"RED >= {int(binom.ppf(.99, ROLLING_N, ALPHA)) + 1}"
    )
    emit()

    scores = build_all_scores(
        breach_panel,
        forecast_dates,
    )

    score_path = (
        OUT_DIR
        / "RS05H_rolling_backtest_scores_H10.csv.gz"
    )
    scores.to_csv(
        score_path,
        index=False,
        compression="gzip",
    )

    signals = []

    signal_specs = {
        "Kupiec": "kupiec_score",
        "Christoffersen": "christoffersen_score",
        "DQ": "dq_score",
        "TrafficLight": "traffic_score",
    }

    all_event_results = []
    all_false_runs = []
    all_burden_rows = []
    all_event_tables = []

    for model in models:
        model_scores = scores[
            scores["model"] == model
        ].copy()

        emit()
        emit("=" * 112)
        emit(f"MODEL: {model}")
        emit("=" * 112)

        model_outcomes = outcomes[
            outcomes["model"] == model
        ].copy()

        daily_failure = build_failure_daily(
            outcomes,
            model,
        )

        events = extract_events(
            daily_failure,
            FAILURE_FRACTION,
        )

        event_table = events.copy()
        event_table["model"] = model
        all_event_tables.append(event_table)

        emit(
            f"Outcome-defined failure events: {len(events)}"
        )

        for signal_name, score_col in signal_specs.items():
            m = model_scores[
                [
                    "forecast_date",
                    "ticker",
                    score_col,
                ]
            ].copy()

            m = m.rename(
                columns={score_col: "signal_score"}
            )

            # Calibration thresholds are computed from preceding 252 dates
            # independently for each model.
            m = apply_fold_thresholds(
                model_scores.rename(
                    columns={score_col: "signal_score"}
                ),
                "signal_score",
            )

            m = m[
                [
                    "forecast_date",
                    "ticker",
                    "model",
                    "signal_score",
                    "threshold",
                    "alert",
                ]
            ].copy()

            m = m[
                m["forecast_date"] >= HEADLINE_START
            ].copy()

            m["signal"] = signal_name

            signals.append(m)

            daily = signal_daily(m)

            event_results, false_runs = analyze_signal(
                signal_name,
                model,
                daily,
                events,
            )

            if not event_results.empty:
                all_event_results.append(event_results)

            if not false_runs.empty:
                all_false_runs.append(false_runs)

            ticker_day_alert_rate = float(
                m["alert"].mean()
            )

            daily_complete = daily[
                daily["complete_24_tickers"]
            ].copy()

            system_alert_rate = float(
                np.mean(
                    daily_complete["alert_fraction"]
                    >= CROSS_SECTION
                )
            )

            all_burden_rows.append(
                {
                    "model": model,
                    "signal": signal_name,
                    "ticker_day_alert_rate": ticker_day_alert_rate,
                    "system_alert_date_rate": system_alert_rate,
                    "mean_alert_fraction": float(
                        daily_complete["alert_fraction"].mean()
                    ),
                    "n_ticker_days": int(len(m)),
                    "n_complete_dates": int(
                        len(daily_complete)
                    ),
                }
            )

            emit(
                f"  {signal_name:14s} | "
                f"ticker-day alerts={ticker_day_alert_rate:.2%} | "
                f"system alert dates={system_alert_rate:.2%}"
            )

    signal_df = pd.concat(
        signals,
        ignore_index=True,
    )

    signal_path = (
        OUT_DIR
        / "RS05H_standard_backtest_signals_H10.csv.gz"
    )
    signal_df.to_csv(
        signal_path,
        index=False,
        compression="gzip",
    )

    event_results_df = (
        pd.concat(
            all_event_results,
            ignore_index=True,
        )
        if all_event_results
        else pd.DataFrame()
    )

    false_runs_df = (
        pd.concat(
            all_false_runs,
            ignore_index=True,
        )
        if all_false_runs
        else pd.DataFrame()
    )

    burden_df = pd.DataFrame(all_burden_rows)

    events_df = (
        pd.concat(
            all_event_tables,
            ignore_index=True,
        )
        if all_event_tables
        else pd.DataFrame()
    )

    summary_df = summarize_events(event_results_df)

    event_path = (
        OUT_DIR
        / "RS05H_standard_backtest_event_results_H10.csv"
    )
    false_path = (
        OUT_DIR
        / "RS05H_standard_backtest_false_alarm_runs_H10.csv"
    )
    burden_path = (
        OUT_DIR
        / "RS05H_standard_backtest_burden_H10.csv"
    )
    summary_path = (
        OUT_DIR
        / "RS05H_standard_backtest_summary_H10.csv"
    )
    events_def_path = (
        OUT_DIR
        / "RS05H_standard_backtest_failure_events_H10.csv"
    )

    event_results_df.to_csv(
        event_path,
        index=False,
    )
    false_runs_df.to_csv(
        false_path,
        index=False,
    )
    burden_df.to_csv(
        burden_path,
        index=False,
    )
    summary_df.to_csv(
        summary_path,
        index=False,
    )
    events_df.to_csv(
        events_def_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Console results
    # -------------------------------------------------------------------------

    emit()
    emit("=" * 112)
    emit("MATCHED-ALERT EVENT SUMMARY")
    emit("=" * 112)

    if summary_df.empty:
        emit("No event results.")
    else:
        emit(
            summary_df.sort_values(
                ["model", "signal"]
            ).round(6).to_string(index=False)
        )

    emit()
    emit("=" * 112)
    emit("REALIZED ALERT BURDEN")
    emit("=" * 112)

    emit(
        burden_df.sort_values(
            ["model", "signal"]
        ).round(6).to_string(index=False)
    )

    emit()
    emit("=" * 112)
    emit("OUTPUTS")
    emit("=" * 112)
    emit(score_path)
    emit(signal_path)
    emit(event_path)
    emit(false_path)
    emit(burden_path)
    emit(summary_path)
    emit(events_def_path)

    emit()
    emit("IMPORTANT:")
    emit(
        "The traffic-light baseline uses alpha-adjusted 5% VaR cutoffs, "
        "not the literal Basel 99% VaR 0-4 / 5-9 / >=10 thresholds."
    )
    emit(
        "Kupiec and Christoffersen use historical breaches only. "
        "DQ is an intentionally conservative lagged-hit DQ-style diagnostic."
    )
    emit(
        "Outcome labels are used only for event evaluation and never for "
        "constructing a test-period alert."
    )

    emit()
    emit("=" * 112)
    emit("RS05H COMPLETE")
    emit("=" * 112)


if __name__ == "__main__":
    main()
