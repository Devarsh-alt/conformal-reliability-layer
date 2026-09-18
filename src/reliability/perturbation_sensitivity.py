from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FINAL_DATA_PATH = (
    ROOT
    / "data"
    / "reliability"
    / "final_candidate_dataset.csv"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "reliability"
    / "perturbation_sensitivity"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from features import get_all_features


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "GS", "BAC", "V",
    "XOM", "CVX",
    "JNJ", "UNH", "PFE",
    "PG", "KO", "MCD", "WMT",
    "CAT", "BA",
    "DIS", "NFLX",
    "TSLA", "GME",
]

FEATURE_COLS = [
    "Log_Return",
    "return_lag1",
    "return_lag5",
    "return_lag10",
    "vol_10d",
    "vol_20d",
    "volume_ratio",
    "VIX",
]

INITIAL_TRAIN_DAYS = 252 * 5
TEST_CHUNK_DAYS = 126

CHANGE_WINDOW = 60

N_PERTURBATIONS = 30
PERTURBATION_SCALE = 0.5

PERCENTILE_WINDOW = 60

COV_JITTER = 1e-10

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def build_target(df):
    df = df.copy()

    df["target"] = df["Log_Return"].shift(-1)

    return df.dropna(
        subset=["target"]
    )


def fit_base_model(train_df):
    model = lgb.LGBMRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    model.fit(
        train_df[FEATURE_COLS],
        train_df["target"],
    )

    return model


def causal_percentile(
    series,
    window=60,
):
    """
    Current observation is ranked only against the previous
    `window` observations.
    """

    values = series.to_numpy(
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan,
        dtype=float,
    )

    for i in range(len(values)):

        start = max(
            0,
            i - window,
        )

        historical = values[
            start:i
        ]

        historical = historical[
            np.isfinite(historical)
        ]

        if len(historical) < window:
            continue

        result[i] = np.mean(
            historical <= values[i]
        )

    return pd.Series(
        result,
        index=series.index,
    )


def generate_fold_ps(
    model,
    test_df,
    feature_changes,
    positions,
    rng,
):
    """
    Generate PS for an entire 126-day test fold.

    The fitted model is kept fixed.

    All perturbations for the entire fold are scored in ONE
    batched LightGBM prediction call.
    """

    n_dates = len(test_df)
    n_features = len(FEATURE_COLS)

    all_perturbed = []
    date_ids = []

    valid_dates = []

    for local_idx, (date, row) in enumerate(
        test_df.iterrows()
    ):

        position = positions[date]

        hist_start = max(
            1,
            position - CHANGE_WINDOW,
        )

        historical_changes = (
            feature_changes.iloc[
                hist_start:position
            ]
            .dropna(
                subset=FEATURE_COLS
            )
        )

        if len(historical_changes) < CHANGE_WINDOW:
            continue

        covariance = (
            historical_changes[
                FEATURE_COLS
            ]
            .cov()
            .to_numpy(
                dtype=float
            )
        )

        covariance = (
            covariance
            + COV_JITTER
            * np.eye(n_features)
        )

        x = row[
            FEATURE_COLS
        ].to_numpy(
            dtype=float
        )

        noise = rng.multivariate_normal(
            mean=np.zeros(n_features),
            cov=covariance,
            size=N_PERTURBATIONS,
            check_valid="warn",
        )

        noise *= PERTURBATION_SCALE

        perturbed = (
            x.reshape(1, -1)
            + noise
        )

        all_perturbed.append(
            perturbed
        )

        date_ids.extend(
            [date] * N_PERTURBATIONS
        )

        valid_dates.append(date)

    if not all_perturbed:

        return pd.DataFrame(
            columns=[
                "prediction_date",
                "ps_raw",
            ]
        )

    # --------------------------------------------------------
    # BIG BATCH PREDICTION
    # --------------------------------------------------------

    matrix = np.vstack(
        all_perturbed
    )

    matrix_df = pd.DataFrame(
        matrix,
        columns=FEATURE_COLS,
    )

    predictions = model.predict(
        matrix_df
    )

    temp = pd.DataFrame(
        {
            "prediction_date": date_ids,
            "prediction": predictions,
        }
    )

    ps = (
        temp.groupby(
            "prediction_date"
        )["prediction"]
        .std(
            ddof=1
        )
        .reset_index()
        .rename(
            columns={
                "prediction":
                    "ps_raw"
            }
        )
    )

    return ps


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("PERTURBATION SENSITIVITY - OPTIMIZED")
print("=" * 80)

final_df = pd.read_csv(
    FINAL_DATA_PATH
)

final_df["prediction_date"] = pd.to_datetime(
    final_df["prediction_date"]
)

final_df["outcome_date"] = pd.to_datetime(
    final_df["outcome_date"]
)

final_df = final_df.sort_values(
    [
        "ticker",
        "prediction_date",
    ]
).reset_index(
    drop=True
)

features = get_all_features()

print(
    f"Final dataset rows: {len(final_df):,}"
)

print(
    f"Feature datasets: {len(features)}"
)


# ============================================================
# COMPUTE PS
# ============================================================

all_ps = []

for ticker_idx, ticker in enumerate(
    TICKERS,
    start=1,
):

    print("\n" + "=" * 80)
    print(
        f"[{ticker_idx}/{len(TICKERS)}] "
        f"TICKER: {ticker}"
    )
    print("=" * 80)

    if ticker not in features:
        print(
            "Missing features. Skipping."
        )
        continue

    ticker_features = features[
        ticker
    ].copy()

    ticker_features.index = pd.to_datetime(
        ticker_features.index
    )

    ticker_features = ticker_features.sort_index()

    df = build_target(
        ticker_features
    )

    if len(df) <= INITIAL_TRAIN_DAYS:
        print(
            "Insufficient history. Skipping."
        )
        continue

    # Keep dates corresponding to the final
    # reliability dataset.
    target_dates = set(
        final_df.loc[
            final_df["ticker"] == ticker,
            "prediction_date",
        ]
    )

    max_target_date = max(
        target_dates
    )

    df = df[
        df.index <= max_target_date
    ].copy()

    if len(df) <= INITIAL_TRAIN_DAYS:
        print(
            "Insufficient usable history. Skipping."
        )
        continue

    # --------------------------------------------------------
    # Historical feature changes
    # --------------------------------------------------------

    feature_changes = (
        df[FEATURE_COLS]
        .diff()
    )

    positions = {
        date: i
        for i, date in enumerate(
            df.index
        )
    }

    fold_results = []

    train_end = INITIAL_TRAIN_DAYS
    fold_id = 0

    while train_end < len(df):

        test_end = min(
            train_end + TEST_CHUNK_DAYS,
            len(df),
        )

        train_df = df.iloc[
            :train_end
        ]

        test_df = df.iloc[
            train_end:test_end
        ]

        print(
            f"Fold {fold_id:02d} | "
            f"train={len(train_df):,} "
            f"test={len(test_df):,}",
            flush=True,
        )

        # ----------------------------------------------------
        # Fit exactly one model for this fold.
        # ----------------------------------------------------

        model = fit_base_model(
            train_df
        )

        # Deterministic seed.
        seed = (
            RANDOM_STATE
            + ticker_idx * 1000
            + fold_id
        )

        rng = np.random.default_rng(
            seed
        )

        # ----------------------------------------------------
        # Batched PS
        # ----------------------------------------------------

        ps = generate_fold_ps(
            model=model,
            test_df=test_df,
            feature_changes=feature_changes,
            positions=positions,
            rng=rng,
        )

        if len(ps) > 0:

            ps["fold_id"] = fold_id
            ps["ticker"] = ticker

            fold_results.append(
                ps
            )

            print(
                f"  PS generated: "
                f"{len(ps):,}",
                flush=True,
            )

        fold_id += 1
        train_end = test_end

    if not fold_results:
        print(
            "No results for ticker."
        )
        continue

    ticker_result = pd.concat(
        fold_results,
        ignore_index=True,
    )

    ticker_result = ticker_result.sort_values(
        "prediction_date"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Causal percentile normalization
    # --------------------------------------------------------

    ticker_result[
        "ps_pct"
    ] = causal_percentile(
        ticker_result["ps_raw"],
        window=PERCENTILE_WINDOW,
    )

    print(
        f"Total PS rows: "
        f"{len(ticker_result):,}"
    )

    print(
        f"Mean PS raw: "
        f"{ticker_result['ps_raw'].mean():.8f}"
    )

    all_ps.append(
        ticker_result
    )


# ============================================================
# COMBINE
# ============================================================

if not all_ps:
    raise RuntimeError(
        "No PS results were generated."
    )

ps_df = pd.concat(
    all_ps,
    ignore_index=True,
)

ps_df = ps_df.sort_values(
    [
        "ticker",
        "prediction_date",
    ]
).reset_index(
    drop=True
)


# ============================================================
# SAVE RAW SIGNAL
# ============================================================

signal_path = (
    OUTPUT_DIR
    / "perturbation_sensitivity.csv"
)

ps_df.to_csv(
    signal_path,
    index=False,
)


# ============================================================
# MERGE
# ============================================================

merged = final_df.merge(
    ps_df[
        [
            "ticker",
            "prediction_date",
            "ps_raw",
            "ps_pct",
        ]
    ],
    on=[
        "ticker",
        "prediction_date",
    ],
    how="left",
    validate="one_to_one",
)

dataset_path = (
    OUTPUT_DIR
    / "candidate_dataset_with_ps.csv"
)

merged.to_csv(
    dataset_path,
    index=False,
)


# ============================================================
# DIAGNOSTICS
# ============================================================

diag = merged.dropna(
    subset=[
        "ps_pct",
        "disagreement_pct",
        "miscovered",
    ]
).copy()

print("\n")
print("=" * 80)
print("PS DIAGNOSTICS")
print("=" * 80)

print(
    f"Final rows: {len(merged):,}"
)

print(
    f"Usable PS rows: {len(diag):,}"
)

print(
    f"Missing PS rows: "
    f"{merged['ps_pct'].isna().sum():,}"
)


# ============================================================
# PS vs D
# ============================================================

rho_d, p_d = spearmanr(
    diag["ps_pct"],
    diag["disagreement_pct"],
)

print("\nPS vs D")
print(
    f"  Spearman rho = {rho_d:.6f}"
)

print(
    f"  p-value      = {p_d:.6e}"
)


# ============================================================
# PS VS MISCOVERAGE
# ============================================================

if diag["miscovered"].nunique() == 2:

    auc_ps = roc_auc_score(
        diag["miscovered"],
        diag["ps_pct"],
    )

    rho_y, p_y = spearmanr(
        diag["ps_pct"],
        diag["miscovered"],
    )

    print("\nPS -> Miscoverage")

    print(
        f"  AUC          = {auc_ps:.6f}"
    )

    print(
        f"  Spearman rho = {rho_y:.6f}"
    )

    print(
        f"  p-value      = {p_y:.6e}"
    )


# ============================================================
# PS DECILES
# ============================================================

diag["ps_decile"] = pd.qcut(
    diag["ps_pct"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1

deciles = (
    diag.groupby(
        "ps_decile"
    )
    .agg(
        n=(
            "miscovered",
            "size",
        ),
        mean_ps=(
            "ps_pct",
            "mean",
        ),
        actual_miscoverage=(
            "miscovered",
            "mean",
        ),
        mean_disagreement=(
            "disagreement_pct",
            "mean",
        ),
    )
    .reset_index()
)

deciles.to_csv(
    OUTPUT_DIR / "ps_deciles.csv",
    index=False,
)

print("\n")
print("=" * 80)
print("PS DECILES")
print("=" * 80)

print(
    deciles.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# PS WITHIN D DECILES
# ============================================================

diag["d_decile"] = pd.qcut(
    diag["disagreement_pct"],
    q=10,
    labels=False,
    duplicates="drop",
) + 1

within_rows = []

for d_decile, group in diag.groupby(
    "d_decile"
):

    if (
        len(group) < 100
        or group["miscovered"].nunique() < 2
    ):
        continue

    auc = roc_auc_score(
        group["miscovered"],
        group["ps_pct"],
    )

    within_rows.append(
        {
            "d_decile": d_decile,
            "n": len(group),
            "ps_auc": auc,
            "actual_miscoverage":
                group["miscovered"].mean(),
        }
    )

within_d = pd.DataFrame(
    within_rows
)

within_d.to_csv(
    OUTPUT_DIR
    / "ps_within_d_deciles.csv",
    index=False,
)

print("\n")
print("=" * 80)
print("PS WITHIN D DECILES")
print("=" * 80)

print(
    within_d.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# YEARLY AUC
# ============================================================

diag["year"] = (
    diag["prediction_date"]
    .dt.year
)

yearly_rows = []

for year, group in diag.groupby(
    "year"
):

    if group["miscovered"].nunique() < 2:
        continue

    auc = roc_auc_score(
        group["miscovered"],
        group["ps_pct"],
    )

    yearly_rows.append(
        {
            "year": year,
            "n": len(group),
            "ps_auc": auc,
            "miscoverage":
                group["miscovered"].mean(),
        }
    )

yearly = pd.DataFrame(
    yearly_rows
)

yearly.to_csv(
    OUTPUT_DIR
    / "ps_yearly_auc.csv",
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "n_rows": len(diag),
    "ps_d_spearman": rho_d,
    "ps_d_pvalue": p_d,
}

if diag["miscovered"].nunique() == 2:

    summary.update(
        {
            "ps_auc": auc_ps,
            "ps_miscoverage_spearman":
                rho_y,
            "ps_miscoverage_pvalue":
                p_y,
        }
    )

pd.DataFrame(
    [summary]
).to_csv(
    OUTPUT_DIR
    / "ps_summary.csv",
    index=False,
)


# ============================================================
# FINAL PRINT
# ============================================================

print("\n")
print("=" * 80)
print("YEARLY PS AUC")
print("=" * 80)

print(
    yearly.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print("\n")
print("=" * 80)
print("SAVED")
print("=" * 80)

print(signal_path)
print(dataset_path)
print(
    OUTPUT_DIR
)