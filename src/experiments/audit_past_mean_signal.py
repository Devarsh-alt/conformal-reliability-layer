from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")


# =============================================================================
# AUDIT:
# IS PAST MEAN PINBALL GENUINELY PREDICTIVE,
# OR IS IT SPECIFICALLY TIED TO THE RELATIVE TARGET?
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "monitoring"

PANEL_PATH = (
    DATA_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_PATH = (
    DATA_DIR
    / "audit_past_mean_signal_results.csv"
)


MODELS = [
    "qrf",
    "ewma",
    "garch",
    "hs",
]

HORIZONS = [5, 10, 20]

TICKERS = [
    "AAPL", "AMZN", "BA", "BAC", "CAT", "CVX",
    "DIS", "GME", "GOOGL", "GS", "JNJ", "JPM",
    "KO", "MCD", "MSFT", "NFLX", "NVDA", "PFE",
    "PG", "TSLA", "UNH", "V", "WMT", "XOM",
]


# =============================================================================
# HELPERS
# =============================================================================

def safe_auc(y, score):

    y = np.asarray(y)
    score = np.asarray(score, dtype=float)

    mask = (
        pd.notna(y)
        &
        np.isfinite(score)
    )

    y = y[mask]
    score = score[mask]

    if len(y) == 0:
        return np.nan

    if len(np.unique(y)) < 2:
        return np.nan

    return float(
        roc_auc_score(y, score)
    )


def macro_auc(
    df,
    target_col,
    score_col,
):

    values = []

    for ticker, g in df.groupby(
        "ticker",
        sort=True,
    ):

        g = g[
            g[target_col].notna()
            &
            g[score_col].notna()
        ]

        if (
            len(g) < 20
            or g[target_col].nunique() < 2
        ):
            continue

        auc = safe_auc(
            g[target_col],
            g[score_col],
        )

        if np.isfinite(auc):
            values.append(auc)

    if not values:
        return np.nan

    return float(
        np.mean(values)
    )


def chronological_split(
    df,
):

    df = (
        df
        .sort_values(
            [
                "ticker",
                "forecast_date",
            ]
        )
        .copy()
    )

    labels = []

    for ticker, g in df.groupby(
        "ticker",
        sort=False,
    ):

        n = len(g)

        train_end = int(
            np.floor(
                0.60 * n
            )
        )

        val_end = int(
            np.floor(
                0.80 * n
            )
        )

        x = np.array(
            ["test"] * n,
            dtype=object,
        )

        x[:train_end] = "train"
        x[train_end:val_end] = "validation"

        labels.extend(
            x.tolist()
        )

    df["split"] = labels

    return df


# =============================================================================
# LOAD
# =============================================================================

print("=" * 100)
print("PAST-MEAN PINBALL TARGET AUDIT")
print("=" * 100)

if not PANEL_PATH.exists():
    raise FileNotFoundError(PANEL_PATH)

df = pd.read_csv(
    PANEL_PATH
)

df["forecast_date"] = pd.to_datetime(
    df["forecast_date"]
)

df["ticker"] = (
    df["ticker"]
    .astype(str)
    .str.upper()
    .str.strip()
)

df = df[
    df["ticker"].isin(TICKERS)
].copy()

df = chronological_split(df)

print(
    f"Rows: {len(df):,}"
)

print(
    df["split"].value_counts()
    .sort_index()
    .to_string()
)


# =============================================================================
# AUDITS
# =============================================================================

results = []


for model in MODELS:

    print(
        "\n" + "-" * 100
    )

    print(
        f"MODEL: {model.upper()}"
    )

    print(
        "-" * 100
    )

    signal = (
        f"{model}_past_mean_pinball"
    )

    for horizon in HORIZONS:

        print(
            f"\n{horizon}D"
        )

        target_primary = (
            f"{model}_deterioration_{horizon}d"
        )

        target_severe = (
            f"{model}_severe_deterioration_{horizon}d"
        )

        target_fixed = (
            f"{model}_deterioration_fixed_1p5_{horizon}d"
        )

        future_loss = (
            f"{model}_future_pinball_{horizon}d"
        )

        threshold75 = (
            f"{model}_causal_threshold_75_{horizon}d"
        )

        threshold90 = (
            f"{model}_causal_threshold_90_{horizon}d"
        )

        # ---------------------------------------------------------------------
        # IMPORTANT:
        #
        # The negative direction was chosen on validation in the previous
        # experiment. We reproduce that here using validation only.
        # ---------------------------------------------------------------------

        validation = df[
            df["split"] == "validation"
        ].copy()

        val = validation[
            validation[target_primary].notna()
            &
            validation[signal].notna()
        ]

        auc_positive = safe_auc(
            val[target_primary],
            val[signal],
        )

        auc_negative = safe_auc(
            val[target_primary],
            -val[signal],
        )

        direction = (
            1.0
            if auc_positive >= auc_negative
            else -1.0
        )

        # ---------------------------------------------------------------------
        # ALL DATA SPLITS
        # ---------------------------------------------------------------------

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            part = df[
                df["split"] == split_name
            ].copy()

            part = part[
                part[signal].notna()
            ].copy()

            score = (
                direction
                * part[signal]
            )

            # -----------------------------------------------------------------
            # PRIMARY RELATIVE TARGET
            # -----------------------------------------------------------------

            primary_auc = safe_auc(
                part[target_primary],
                score,
            )

            # -----------------------------------------------------------------
            # SEVERE RELATIVE TARGET
            # -----------------------------------------------------------------

            severe_auc = safe_auc(
                part[target_severe],
                score,
            )

            # -----------------------------------------------------------------
            # FIXED 1.5x TARGET
            # -----------------------------------------------------------------

            fixed_auc = safe_auc(
                part[target_fixed],
                score,
            )

            # -----------------------------------------------------------------
            # FUTURE LOSS AS A CONTINUOUS OUTCOME
            #
            # We use Spearman correlation rather than pretending future loss
            # is binary.
            # -----------------------------------------------------------------

            valid_loss = part[
                part[future_loss].notna()
                &
                np.isfinite(
                    score.to_numpy()
                )
            ].copy()

            spearman = (
                valid_loss[
                    [
                        future_loss
                    ]
                ]
                .join(
                    pd.Series(
                        score.to_numpy(),
                        index=score.index,
                        name="score",
                    ),
                    how="inner",
                )
                .corr(
                    method="spearman"
                )
                .iloc[0, 1]
                if len(valid_loss) > 5
                else np.nan
            )

            # -----------------------------------------------------------------
            # SIMPLE LINEAR CORRELATION WITH FUTURE LOSS
            # -----------------------------------------------------------------

            pearson = (
                valid_loss[
                    future_loss
                ]
                .corr(
                    pd.Series(
                        score.loc[
                            valid_loss.index
                        ],
                        index=valid_loss.index,
                    ),
                    method="pearson",
                )
                if len(valid_loss) > 5
                else np.nan
            )

            results.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "split":
                        split_name,

                    "direction":
                        direction,

                    "n":
                        len(part),

                    "primary_relative_auc":
                        primary_auc,

                    "severe_relative_auc":
                        severe_auc,

                    "fixed_1p5_auc":
                        fixed_auc,

                    "future_loss_spearman":
                        spearman,

                    "future_loss_pearson":
                        pearson,

                    "mean_future_loss":
                        (
                            part[
                                future_loss
                            ].mean()
                        ),

                    "mean_past_loss":
                        (
                            part[
                                signal
                            ].mean()
                        ),

                    "mean_threshold75":
                        (
                            part[
                                threshold75
                            ].mean()
                        ),

                    "mean_threshold90":
                        (
                            part[
                                threshold90
                            ].mean()
                        ),

                    "positive_primary_rate":
                        (
                            part[
                                target_primary
                            ].mean()
                        ),

                    "positive_severe_rate":
                        (
                            part[
                                target_severe
                            ].mean()
                        ),

                    "positive_fixed_rate":
                        (
                            part[
                                target_fixed
                            ].mean()
                        ),
                }
            )

        # ---------------------------------------------------------------------
        # WITHIN-TICKER SUBPERIOD TEST AUDIT
        # ---------------------------------------------------------------------

        test = df[
            df["split"] == "test"
        ].copy()

        test = test[
            test[signal].notna()
            &
            test[target_primary].notna()
        ].copy()

        midpoint = (
            test["forecast_date"]
            .median()
        )

        for period_name, mask in [

            (
                "early_test",
                test[
                    "forecast_date"
                ]
                <= midpoint,
            ),

            (
                "late_test",
                test[
                    "forecast_date"
                ]
                > midpoint,
            ),
        ]:

            sub = test[
                mask
            ].copy()

            score = (
                direction
                * sub[signal]
            )

            auc = safe_auc(
                sub[target_primary],
                score,
            )

            macro = macro_auc(
                pd.DataFrame(
                    {
                        "ticker":
                            sub["ticker"],

                        "target":
                            sub[target_primary],

                        "score":
                            score,
                    }
                ),
                "target",
                "score",
            )

            results.append(
                {
                    "model":
                        model.upper(),

                    "horizon":
                        horizon,

                    "split":
                        period_name,

                    "direction":
                        direction,

                    "n":
                        len(sub),

                    "primary_relative_auc":
                        auc,

                    "severe_relative_auc":
                        np.nan,

                    "fixed_1p5_auc":
                        np.nan,

                    "future_loss_spearman":
                        np.nan,

                    "future_loss_pearson":
                        np.nan,

                    "mean_future_loss":
                        sub[
                            future_loss
                        ].mean(),

                    "mean_past_loss":
                        sub[
                            signal
                        ].mean(),

                    "mean_threshold75":
                        sub[
                            threshold75
                        ].mean(),

                    "mean_threshold90":
                        sub[
                            threshold90
                        ].mean(),

                    "positive_primary_rate":
                        sub[
                            target_primary
                        ].mean(),

                    "positive_severe_rate":
                        np.nan,

                    "positive_fixed_rate":
                        np.nan,

                    "macro_primary_auc":
                        macro,
                }
            )


# =============================================================================
# SAVE
# =============================================================================

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    OUTPUT_PATH,
    index=False,
)


# =============================================================================
# PRINT CORE TABLE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "CORE AUDIT RESULTS"
)

print(
    "=" * 100
)

core = results_df[
    results_df["split"].isin(
        [
            "train",
            "validation",
            "test",
        ]
    )
].copy()

print(
    core[
        [
            "model",
            "horizon",
            "split",
            "primary_relative_auc",
            "severe_relative_auc",
            "fixed_1p5_auc",
            "future_loss_spearman",
            "future_loss_pearson",
            "positive_primary_rate",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# SUBPERIOD TABLE
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "TEST SUBPERIOD AUDIT"
)

print(
    "=" * 100
)

sub = results_df[
    results_df["split"].isin(
        [
            "early_test",
            "late_test",
        ]
    )
]

print(
    sub[
        [
            "model",
            "horizon",
            "split",
            "n",
            "primary_relative_auc",
            "macro_primary_auc",
            "mean_future_loss",
            "mean_past_loss",
            "positive_primary_rate",
        ]
    ].to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# =============================================================================
# FINAL
# =============================================================================

print(
    "\n" + "=" * 100
)

print(
    "AUDIT COMPLETE"
)

print(
    "=" * 100
)

print(
    f"\nSaved:"
)

print(
    OUTPUT_PATH
)