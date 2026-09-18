from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    brier_score_loss,
    roc_auc_score,
)


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
    / "robustness"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_PATH = (
    OUTPUT_DIR
    / "ps_robustness_features.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "ps_robustness_summary.csv"
)

YEARLY_PATH = (
    OUTPUT_DIR
    / "ps_robustness_yearly.csv"
)

SINGLE_AUC_PATH = (
    OUTPUT_DIR
    / "ps_single_signal_auc.csv"
)


# ============================================================
# IMPORT PROJECT FEATURES
# ============================================================

sys.path.insert(
    0,
    str(ROOT / "src"),
)

from features import get_all_features


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "JPM",
    "GS",
    "BAC",
    "V",
    "XOM",
    "CVX",
    "JNJ",
    "UNH",
    "PFE",
    "PG",
    "KO",
    "MCD",
    "WMT",
    "CAT",
    "BA",
    "DIS",
    "NFLX",
    "TSLA",
    "GME",
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

# Robustness grid
WINDOWS = [40, 60, 120]
SCALES = [0.25, 0.50, 1.00]

# Generate max number once, then use nested subsets
N_MAX = 50
N_OPTIONS = [20, 30, 50]

RANDOM_STATE = 42
COV_JITTER = 1e-10

TEST_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
    2026,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def build_target(df):
    """
    Build one-step-ahead target.

    Prediction at t:
        features available at t

Target:
        next-day log return
    """

    df = df.copy()

    df["target"] = df["Log_Return"].shift(-1)

    df = df.dropna(
        subset=["target"]
    )

    return df


def fit_base_model(train_df):
    """
    Same LightGBM specification used in the PS experiment.
    """

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
    Causal empirical percentile.

    Current observation is compared only with
    observations strictly before it.
    """

    values = series.to_numpy(
        dtype=float
    )

    result = np.full(
        len(values),
        np.nan,
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


def build_meta_model():
    """
    Reliability meta-model.
    """

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def generate_ps_for_fold(
    model,
    test_df,
    feature_changes,
    positions,
    rng,
):
    """
    Generate all PS robustness variants for one
    walk-forward fold.

    For every prediction:
      - estimate covariance from previous observations
      - generate one perturbation cloud
      - reuse the same cloud for different scales
      - use nested subsets for 20/30/50 perturbations
    """

    records = []

    for date, row in test_df.iterrows():

        position = positions.get(date)

        if position is None:
            continue

        # Need sufficient history for the largest
        # covariance window.
        if position < max(WINDOWS):
            continue

        x = row[
            FEATURE_COLS
        ].to_numpy(
            dtype=float
        )

        # Skip malformed feature rows
        if not np.isfinite(x).all():
            continue

        ps_by_config = {}

        # ----------------------------------------------------
        # Different covariance windows
        # ----------------------------------------------------

        for cov_window in WINDOWS:

            hist_start = max(
                1,
                position - cov_window,
            )

            historical_changes = (
                feature_changes.iloc[
                    hist_start:position
                ]
                .dropna(
                    subset=FEATURE_COLS
                )
            )

            if len(historical_changes) < cov_window:
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

            if not np.isfinite(
                covariance
            ).all():
                continue

            # Numerical stabilization
            covariance = (
                covariance
                + COV_JITTER
                * np.eye(
                    len(FEATURE_COLS)
                )
            )

            # ------------------------------------------------
            # ONE common perturbation cloud
            # ------------------------------------------------

            try:

                noise = rng.multivariate_normal(
                    mean=np.zeros(
                        len(FEATURE_COLS)
                    ),
                    cov=covariance,
                    size=N_MAX,
                )

            except np.linalg.LinAlgError:

                # Fallback to diagonal covariance
                diagonal = np.diag(
                    np.diag(covariance)
                )

                noise = rng.multivariate_normal(
                    mean=np.zeros(
                        len(FEATURE_COLS)
                    ),
                    cov=diagonal,
                    size=N_MAX,
                )

            # ------------------------------------------------
            # Different perturbation scales
            # ------------------------------------------------

            for scale in SCALES:

                perturbed = (
                    x.reshape(1, -1)
                    + scale * noise
                )

                perturbed_df = pd.DataFrame(
                    perturbed,
                    columns=FEATURE_COLS,
                )

                predictions = model.predict(
                    perturbed_df
                )

                # --------------------------------------------
                # Nested perturbation counts
                # --------------------------------------------

                for n_perturb in N_OPTIONS:

                    selected_predictions = (
                        predictions[
                            :n_perturb
                        ]
                    )

                    ps = np.std(
                        selected_predictions,
                        ddof=1,
                    )

                    key = (
                        f"w{cov_window}"
                        f"_s{scale:.2f}"
                        f"_n{n_perturb}"
                    )

                    ps_by_config[key] = ps

        if ps_by_config:

            records.append(
                {
                    "prediction_date": date,
                    **ps_by_config,
                }
            )

    return pd.DataFrame(
        records
    )


def parse_config(config):
    """
    Parse:
        w60_s0.50_n30
    """

    parts = config.split("_")

    window = int(
        parts[0][1:]
    )

    scale = float(
        parts[1][1:]
    )

    n_perturbations = int(
        parts[2][1:]
    )

    return (
        window,
        scale,
        n_perturbations,
    )


# ============================================================
# START
# ============================================================

print("=" * 80)
print("PS ROBUSTNESS EXPERIMENT")
print("=" * 80)


# ============================================================
# LOAD FINAL RELIABILITY DATASET
# ============================================================

final_df = pd.read_csv(
    FINAL_DATA_PATH
)

# CRITICAL FIX:
# Both dates must be converted before comparing them
# against Timestamp objects.

final_df["prediction_date"] = pd.to_datetime(
    final_df["prediction_date"],
    errors="coerce",
)

final_df["outcome_date"] = pd.to_datetime(
    final_df["outcome_date"],
    errors="coerce",
)

final_df = final_df.dropna(
    subset=[
        "prediction_date",
        "outcome_date",
    ]
)

final_df = final_df.sort_values(
    [
        "ticker",
        "prediction_date",
    ]
).reset_index(
    drop=True
)

print(
    f"Final dataset rows: "
    f"{len(final_df):,}"
)


# ============================================================
# STEP 1
# LOAD EXISTING EXPENSIVE PS OUTPUT
# OR GENERATE IT IF MISSING
# ============================================================

if RAW_PATH.exists():

    print("\n")
    print("=" * 80)
    print("EXISTING PS ROBUSTNESS FEATURES FOUND")
    print("=" * 80)

    print(
        f"Loading:\n{RAW_PATH}"
    )

    ps_all = pd.read_csv(
        RAW_PATH
    )

    ps_all["prediction_date"] = pd.to_datetime(
        ps_all["prediction_date"],
        errors="coerce",
    )

    ps_all = ps_all.dropna(
        subset=["prediction_date"]
    )

    print(
        f"Loaded rows: "
        f"{len(ps_all):,}"
    )

    print(
        "Skipping expensive PS generation."
    )

else:

    print("\n")
    print("=" * 80)
    print("PS ROBUSTNESS FEATURES NOT FOUND")
    print("=" * 80)

    print(
        "Running expensive PS generation..."
    )

    features = get_all_features()

    all_outputs = []

    for ticker_idx, ticker in enumerate(
        TICKERS,
        start=1,
    ):

        print("\n" + "=" * 80)
        print(
            f"[{ticker_idx}/{len(TICKERS)}] {ticker}"
        )
        print("=" * 80)

        if ticker not in features:

            print(
                "Missing features. Skipping."
            )

            continue

        # ----------------------------------------------------
        # Prepare ticker data
        # ----------------------------------------------------

        df = build_target(
            features[ticker].copy()
        )

        df.index = pd.to_datetime(
            df.index
        )

        df = df.sort_index()

        # ----------------------------------------------------
        # Align with final dataset date range
        # ----------------------------------------------------

        ticker_final = final_df.loc[
            final_df["ticker"] == ticker
        ]

        if ticker_final.empty:

            print(
                "Ticker missing from final dataset."
            )

            continue

        max_date = ticker_final[
            "prediction_date"
        ].max()

        df = df[
            df.index <= max_date
        ].copy()

        # ----------------------------------------------------
        # Feature differences
        # ----------------------------------------------------

        feature_changes = (
            df[
                FEATURE_COLS
            ]
            .diff()
        )

        # ----------------------------------------------------
        # Fast date -> integer lookup
        # ----------------------------------------------------

        positions = {
            date: i
            for i, date in enumerate(
                df.index
            )
        }

        # ----------------------------------------------------
        # Walk-forward folds
        # ----------------------------------------------------

        train_end = INITIAL_TRAIN_DAYS
        fold_id = 0

        ticker_results = []

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

            # ------------------------------------------------
            # Fit exactly one base model per fold
            # ------------------------------------------------

            model = fit_base_model(
                train_df
            )

            seed = (
                RANDOM_STATE
                + ticker_idx * 10000
                + fold_id
            )

            rng = np.random.default_rng(
                seed
            )

            # ------------------------------------------------
            # Generate robustness PS
            # ------------------------------------------------

            ps = generate_ps_for_fold(
                model=model,
                test_df=test_df,
                feature_changes=feature_changes,
                positions=positions,
                rng=rng,
            )

            if len(ps) > 0:

                ps["ticker"] = ticker
                ps["fold_id"] = fold_id

                ticker_results.append(
                    ps
                )

            fold_id += 1
            train_end = test_end

        # ----------------------------------------------------
        # Save ticker results
        # ----------------------------------------------------

        if not ticker_results:

            print(
                "No PS results for ticker."
            )

            continue

        ticker_df = pd.concat(
            ticker_results,
            ignore_index=True,
        )

        ticker_df = ticker_df.sort_values(
            "prediction_date"
        ).reset_index(
            drop=True
        )

        # ----------------------------------------------------
        # Detect PS configurations
        # ----------------------------------------------------

        config_cols = [
            c
            for c in ticker_df.columns
            if c.startswith("w")
            and "_s" in c
            and "_n" in c
        ]

        # ----------------------------------------------------
        # Causal percentile normalization
        # ----------------------------------------------------

        for col in config_cols:

            pct_col = (
                col
                + "_pct"
            )

            ticker_df[
                pct_col
            ] = causal_percentile(
                ticker_df[col],
                window=60,
            )

        all_outputs.append(
            ticker_df
        )

    # ========================================================
    # COMBINE ALL TICKERS
    # ========================================================

    if not all_outputs:

        raise RuntimeError(
            "PS generation produced no results."
        )

    ps_all = pd.concat(
        all_outputs,
        ignore_index=True,
    )

    print("\n")
    print(
        f"PS robustness rows: "
        f"{len(ps_all):,}"
    )

    # ========================================================
    # SAVE EXPENSIVE OUTPUT
    # ========================================================

    ps_all.to_csv(
        RAW_PATH,
        index=False,
    )

    print(
        f"Saved expensive PS output:\n{RAW_PATH}"
    )


# ============================================================
# STEP 2
# MERGE PS WITH RELIABILITY DATASET
# ============================================================

print("\n")
print("=" * 80)
print("MERGING PS FEATURES")
print("=" * 80)

# Make absolutely sure dates are timestamps
ps_all["prediction_date"] = pd.to_datetime(
    ps_all["prediction_date"],
    errors="coerce",
)

ps_all = ps_all.dropna(
    subset=["prediction_date"]
)

# Remove accidental duplicate keys
ps_all = ps_all.drop_duplicates(
    subset=[
        "ticker",
        "prediction_date",
    ],
    keep="first",
)

df = final_df.merge(
    ps_all,
    on=[
        "ticker",
        "prediction_date",
    ],
    how="inner",
    validate="one_to_one",
)

print(
    f"Merged rows: "
    f"{len(df):,}"
)


# ============================================================
# STEP 3
# IDENTIFY CONFIGURATIONS
# ============================================================

configs = [
    c
    for c in ps_all.columns
    if c.startswith("w")
    and "_s" in c
    and "_n" in c
    and not c.endswith("_pct")
]

if len(configs) == 0:

    raise RuntimeError(
        "No PS configuration columns found."
    )

print(
    f"Number of PS configurations: "
    f"{len(configs)}"
)


# ============================================================
# STEP 4
# SINGLE-SIGNAL AUC
# ============================================================

print("\n")
print("=" * 80)
print("PS SINGLE-SIGNAL ROBUSTNESS")
print("=" * 80)

single_results = []

for col in configs:

    pct_col = (
        col
        + "_pct"
    )

    if pct_col not in df.columns:

        continue

    subset = df.dropna(
        subset=[
            pct_col,
            "miscovered",
        ]
    ).copy()

    if len(subset) == 0:
        continue

    if subset["miscovered"].nunique() < 2:
        continue

    auc = roc_auc_score(
        subset["miscovered"].astype(int),
        subset[pct_col],
    )

    single_results.append(
        {
            "config": col,
            "auc": auc,
            "n": len(subset),
        }
    )

single_results = pd.DataFrame(
    single_results
)

if not single_results.empty:

    print(
        single_results.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

else:

    print(
        "No valid single-signal results."
    )


# ============================================================
# STEP 5
# FIXED OUT-OF-SAMPLE D VS D+PS
# ============================================================

print("\n")
print("=" * 80)
print("D -> D+PS ROBUSTNESS")
print("=" * 80)

oos_results = []

for config in configs:

    ps_col = (
        config
        + "_pct"
    )

    if ps_col not in df.columns:

        continue

    print(
        f"\nConfiguration: {config}"
    )

    for test_year in TEST_YEARS:

        test_start = pd.Timestamp(
            f"{test_year}-01-01"
        )

        test_end = pd.Timestamp(
            f"{test_year + 1}-01-01"
        )

        # ----------------------------------------------------
        # Causal temporal split
        #
        # Training outcomes must have arrived BEFORE
        # test period begins.
        # ----------------------------------------------------

        train = df[
            (df["outcome_date"] < test_start)
            &
            (df["prediction_date"] < test_start)
        ].copy()

        test = df[
            (df["prediction_date"] >= test_start)
            &
            (df["prediction_date"] < test_end)
        ].copy()

        # ----------------------------------------------------
        # D
        # ----------------------------------------------------

        train_d = train.dropna(
            subset=[
                "disagreement_pct",
                "miscovered",
            ]
        ).copy()

        test_d = test.dropna(
            subset=[
                "disagreement_pct",
                "miscovered",
            ]
        ).copy()

        if len(train_d) == 0 or len(test_d) == 0:
            continue

        if train_d["miscovered"].nunique() < 2:
            continue

        model_d = build_meta_model()

        model_d.fit(
            train_d[
                ["disagreement_pct"]
            ],
            train_d[
                "miscovered"
            ].astype(int),
        )

        p_d = model_d.predict_proba(
            test_d[
                ["disagreement_pct"]
            ]
        )[:, 1]

        brier_d = brier_score_loss(
            test_d["miscovered"].astype(int),
            p_d,
        )

        # ----------------------------------------------------
        # D + PS
        # ----------------------------------------------------

        train_dps = train.dropna(
            subset=[
                "disagreement_pct",
                ps_col,
                "miscovered",
            ]
        ).copy()

        test_dps = test.dropna(
            subset=[
                "disagreement_pct",
                ps_col,
                "miscovered",
            ]
        ).copy()

        if (
            len(train_dps) == 0
            or len(test_dps) == 0
        ):
            continue

        if train_dps["miscovered"].nunique() < 2:
            continue

        model_dps = build_meta_model()

        model_dps.fit(
            train_dps[
                [
                    "disagreement_pct",
                    ps_col,
                ]
            ],
            train_dps[
                "miscovered"
            ].astype(int),
        )

        p_dps = model_dps.predict_proba(
            test_dps[
                [
                    "disagreement_pct",
                    ps_col,
                ]
            ]
        )[:, 1]

        brier_dps = brier_score_loss(
            test_dps[
                "miscovered"
            ].astype(int),
            p_dps,
        )

        # ----------------------------------------------------
        # AUC D
        # ----------------------------------------------------

        auc_d = roc_auc_score(
            test_d["miscovered"].astype(int),
            p_d,
        )

        # ----------------------------------------------------
        # AUC D + PS
        # ----------------------------------------------------

        auc_dps = roc_auc_score(
            test_dps[
                "miscovered"
            ].astype(int),
            p_dps,
        )

        # ----------------------------------------------------
        # Store
        # ----------------------------------------------------

        oos_results.append(
            {
                "config": config,
                "test_year": test_year,
                "brier_D": brier_d,
                "brier_D_PS": brier_dps,
                "improvement_D_to_DPS":
                    brier_d - brier_dps,
                "auc_D": auc_d,
                "auc_D_PS": auc_dps,
                "n_test": len(test_dps),
            }
        )

        print(
            f"  {test_year}: "
            f"D={brier_d:.6f} | "
            f"D+PS={brier_dps:.6f} | "
            f"improvement="
            f"{brier_d - brier_dps:+.6f}"
        )


oos_results = pd.DataFrame(
    oos_results
)


# ============================================================
# STEP 6
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("ROBUSTNESS SUMMARY")
print("=" * 80)

if oos_results.empty:

    raise RuntimeError(
        "No OOS robustness results were generated."
    )


summary = (
    oos_results
    .groupby("config")
    .agg(
        mean_brier_D=(
            "brier_D",
            "mean",
        ),
        mean_brier_D_PS=(
            "brier_D_PS",
            "mean",
        ),
        mean_improvement=(
            "improvement_D_to_DPS",
            "mean",
        ),
        median_improvement=(
            "improvement_D_to_DPS",
            "median",
        ),
        positive_years=(
            "improvement_D_to_DPS",
            lambda x: (
                x > 0
            ).sum()
        ),
        n_years=(
            "test_year",
            "count",
        ),
        mean_auc_D=(
            "auc_D",
            "mean",
        ),
        mean_auc_D_PS=(
            "auc_D_PS",
            "mean",
        ),
    )
    .reset_index()
)


summary["positive_fraction"] = (
    summary["positive_years"]
    / summary["n_years"]
)


# ============================================================
# MERGE SINGLE-SIGNAL AUC
# ============================================================

if not single_results.empty:

    summary = summary.merge(
        single_results[
            [
                "config",
                "auc",
            ]
        ],
        on="config",
        how="left",
    )

else:

    summary["auc"] = np.nan


# ============================================================
# PARSE CONFIG
# ============================================================

parsed = summary[
    "config"
].apply(
    lambda x: pd.Series(
        parse_config(x)
    )
)

parsed.columns = [
    "cov_window",
    "scale",
    "n_perturbations",
]

summary = pd.concat(
    [
        summary.drop(
            columns=["config"]
        ),
        parsed,
    ],
    axis=1,
)


summary = summary.sort_values(
    [
        "cov_window",
        "scale",
        "n_perturbations",
    ]
).reset_index(
    drop=True
)


# ============================================================
# DISPLAY SINGLE PS RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("SINGLE PS AUC")
print("=" * 100)

print(
    summary[
        [
            "cov_window",
            "scale",
            "n_perturbations",
            "auc",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# ============================================================
# DISPLAY INCREMENTAL D -> D+PS RESULTS
# ============================================================

print("\n")
print("=" * 100)
print("D -> D+PS")
print("=" * 100)

print(
    summary[
        [
            "cov_window",
            "scale",
            "n_perturbations",
            "mean_improvement",
            "median_improvement",
            "positive_years",
            "n_years",
            "positive_fraction",
            "mean_auc_D",
            "mean_auc_D_PS",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}",
    )
)


# ============================================================
# STEP 7
# IDENTIFY ORIGINAL PRIMARY CONFIGURATION
# ============================================================

primary_mask = (
    (summary["cov_window"] == 60)
    &
    (summary["scale"] == 0.50)
    &
    (summary["n_perturbations"] == 30)
)

primary = summary[
    primary_mask
]

print("\n")
print("=" * 80)
print("PRIMARY CONFIGURATION: w60_s0.50_n30")
print("=" * 80)

if primary.empty:

    print(
        "Primary configuration not found."
    )

else:

    print(
        primary.to_string(
            index=False,
            float_format=lambda x: f"{x:.8f}",
        )
    )


# ============================================================
# STEP 8
# SAVE RESULTS
# ============================================================

summary.to_csv(
    SUMMARY_PATH,
    index=False,
)

oos_results.to_csv(
    YEARLY_PATH,
    index=False,
)

single_results.to_csv(
    SINGLE_AUC_PATH,
    index=False,
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n")
print("=" * 80)
print("DONE")
print("=" * 80)

print(
    f"Robustness features:\n{RAW_PATH}"
)

print(
    f"Summary:\n{SUMMARY_PATH}"
)

print(
    f"Yearly results:\n{YEARLY_PATH}"
)

print(
    f"Single-signal AUC:\n{SINGLE_AUC_PATH}"
)

print("\nNo expensive PS regeneration will be required")
print("on subsequent runs as long as the raw CSV exists.")