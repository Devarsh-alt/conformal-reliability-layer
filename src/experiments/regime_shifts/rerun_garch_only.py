from pathlib import Path
import shutil
import logging

import numpy as np
import pandas as pd

import long_history_var as L


# =============================================================================
# RS01 — GARCH-ONLY CORRECTION / REGENERATION
#
# PURPOSE
# -------
# Correct ONLY the GARCH recursion in the existing RS01 panel.
#
# DO NOT recompute:
#   - Historical Simulation (HS)
#   - EWMA
#   - Quantile Random Forest (QRF)
#
# EXISTING RS01 PANEL
# -------------------
# RS01_long_history_var_panel.csv
#
# The existing panel is the common QRF-valid sample:
#     68,952 rows
#     24 tickers
#     2,873 rows per ticker
#
# The full GARCH reconstruction naturally contains:
#     2,893 rows per ticker
#
# Therefore:
#     20 extra rows × 24 tickers = 480 extra GARCH keys
#
# Those 480 rows occur before the common QRF sample begins and are expected.
#
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(
    r"D:\reliability_layer"
)

PANEL_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_long_history_var_panel.csv"
)

BACKUP_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_panel_PRE_GARCH_FIX.csv"
)

LOG_PATH = (
    ROOT
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_garch_correction.log"
)


# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(
            LOG_PATH,
            mode="w",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

log = logging.getLogger(__name__)


# =============================================================================
# CORRECTED GARCH BLOCK
# =============================================================================

def fit_garch_block_corrected(
    returns_by_date: pd.Series,
    dates: pd.DatetimeIndex,
    positions: np.ndarray,
    ticker: str,
    fit_number: int,
):
    """
    Corrected GARCH(1,1)-Student-t block.

    Parameters are fitted once at the beginning of the block and then frozen
    over REFIT_EVERY forecast dates.

    Correct timing:

        At forecast date t:

            sigma²(t+1)
                = omega
                + alpha * epsilon(t)^2
                + beta  * sigma²(t)

        The resulting sigma(t+1) is used for the VaR forecast of return t+1.

    The original implementation incorrectly reused the previous forecast
    date's return inside the block. This version uses the CURRENT forecast
    date's return.
    """

    if len(positions) == 0:
        return []

    # =========================================================================
    # FIRST FORECAST DATE
    # =========================================================================

    first_position = positions[0]

    first_forecast_date = dates[
        first_position
    ]


    # =========================================================================
    # CAUSAL TRAINING WINDOW
    #
    # Information through first_forecast_date is available.
    # =========================================================================

    training_returns = (
        returns_by_date.loc[
            returns_by_date.index
            <= first_forecast_date
        ]
        .dropna()
    )

    if len(training_returns) < L.WINDOW:
        return []

    training_window = (
        training_returns
        .iloc[-L.WINDOW:]
        .to_numpy(
            dtype=float
        )
    )


    # =========================================================================
    # SCALE RETURNS
    # =========================================================================

    scaled_returns = (
        training_window
        * 100.0
    )


    # =========================================================================
    # FIT GARCH(1,1)-STUDENT-t
    # =========================================================================

    model = L.arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=L.GARCH_P,
        q=L.GARCH_Q,
        dist="t",
        rescale=False,
    )

    try:

        fitted = model.fit(
            disp="off"
        )

    except Exception as exc:

        log.error(
            f"{ticker}: GARCH fit failure "
            f"at fit {fit_number}: {exc}"
        )

        return []


    # =========================================================================
    # GET PARAMETERS
    # =========================================================================

    params = fitted.params

    mu = float(
        params.get(
            "mu",
            0.0,
        )
    )

    omega = float(
        params.get(
            "omega",
            1e-8,
        )
    )

    alpha1 = float(
        params.get(
            "alpha[1]",
            0.0,
        )
    )

    beta1 = float(
        params.get(
            "beta[1]",
            0.0,
        )
    )

    nu = float(
        params.get(
            "nu",
            8.0,
        )
    )


    # =========================================================================
    # LAST FITTED CONDITIONAL VARIANCE
    # =========================================================================

    conditional_volatility = np.asarray(
        fitted.conditional_volatility,
        dtype=float,
    )

    sigma2_t = float(
        conditional_volatility[-1] ** 2
    )


    # =========================================================================
    # FIRST ONE-STEP-AHEAD VARIANCE
    #
    # Training sample ends at t.
    # Use epsilon(t) to obtain variance for t+1.
    # =========================================================================

    last_return_scaled = (
        training_window[-1]
        * 100.0
    )

    epsilon_t = (
        last_return_scaled
        - mu
    )

    current_variance = (
        omega
        + alpha1
        * epsilon_t
        * epsilon_t
        + beta1
        * sigma2_t
    )

    current_variance = max(
        current_variance,
        1e-12,
    )


    # =========================================================================
    # STANDARDIZED STUDENT-t 5% QUANTILE
    # =========================================================================

    raw_quantile = L.student_t.ppf(
        L.ALPHA,
        df=nu,
    )

    if nu > 2:

        standardized_quantile = (
            raw_quantile
            * np.sqrt(
                (nu - 2.0)
                / nu
            )
        )

    else:

        standardized_quantile = (
            raw_quantile
        )


    # =========================================================================
    # FORECAST BLOCK
    # =========================================================================

    rows = []

    for j, position in enumerate(
        positions
    ):

        forecast_date = dates[
            position
        ]


        # =====================================================================
        # CRITICAL CORRECTION
        #
        # For forecast made at t:
        #
        #     use return observed at t
        #
        # to obtain:
        #
        #     sigma²(t+1)
        #
        # NOT the previous forecast-date return.
        # =====================================================================

        if j > 0:

            current_return = (
                returns_by_date.get(
                    forecast_date,
                    np.nan,
                )
            )

            if np.isfinite(
                current_return
            ):

                current_scaled = (
                    current_return
                    * 100.0
                )

                current_epsilon = (
                    current_scaled
                    - mu
                )

                current_variance = (
                    omega
                    + alpha1
                    * current_epsilon
                    * current_epsilon
                    + beta1
                    * current_variance
                )

                current_variance = max(
                    current_variance,
                    1e-12,
                )


        # =====================================================================
        # RETURN-SCALE CONDITIONAL STANDARD DEVIATION
        # =====================================================================

        sigma = (
            np.sqrt(
                current_variance
            )
            / 100.0
        )


        # =====================================================================
        # GARCH-t 5% VaR
        # =====================================================================

        garch_quantile = (
            mu / 100.0
            + standardized_quantile
            * sigma
        )


        # =====================================================================
        # OUTPUT
        # =====================================================================

        rows.append(
            {
                "ticker": ticker,

                "forecast_date": forecast_date,

                "garch_var": float(
                    garch_quantile
                ),

                "garch_mu": float(
                    mu / 100.0
                ),

                "garch_sigma": float(
                    sigma
                ),

                "garch_df": float(
                    nu
                ),

                "garch_fit_number": int(
                    fit_number
                ),
            }
        )

    return rows


# =============================================================================
# START
# =============================================================================

log.info(
    "=" * 100
)

log.info(
    "RS01 — GARCH-ONLY CORRECTION / REGENERATION"
)

log.info(
    "=" * 100
)

log.info("")

log.info(
    "HS, EWMA and QRF will NOT be recomputed."
)

log.info(
    "Only GARCH columns will be replaced."
)


# =============================================================================
# CHECK EXISTING PANEL
# =============================================================================

if not PANEL_PATH.exists():

    raise FileNotFoundError(
        f"RS01 panel not found:\n{PANEL_PATH}"
    )

panel = pd.read_csv(
    PANEL_PATH
)

panel["forecast_date"] = pd.to_datetime(
    panel["forecast_date"],
    errors="coerce",
)

panel["realized_date"] = pd.to_datetime(
    panel["realized_date"],
    errors="coerce",
)

log.info("")

log.info(
    f"Existing panel rows: "
    f"{len(panel):,}"
)

log.info(
    f"Tickers: "
    f"{panel['ticker'].nunique()}"
)


# =============================================================================
# BACKUP ORIGINAL PANEL
# =============================================================================

if not BACKUP_PATH.exists():

    shutil.copy2(
        PANEL_PATH,
        BACKUP_PATH,
    )

    log.info("")

    log.info(
        f"Backup created:\n{BACKUP_PATH}"
    )

else:

    log.info("")

    log.info(
        "Backup already exists; leaving it untouched."
    )

    log.info(
        f"Backup:\n{BACKUP_PATH}"
    )


# =============================================================================
# REQUIRED EXISTING PANEL COLUMNS
# =============================================================================

required_panel_columns = [
    "ticker",
    "forecast_date",
    "realized_date",
    "actual_return",

    # HS
    "hs_var",
    "hs_breach",
    "hs_pinball_loss",

    # EWMA
    "ewma_var",
    "ewma_breach",
    "ewma_pinball_loss",

    # QRF
    "qrf_var",
    "qrf_tree_mean",
    "qrf_tree_std",
    "qrf_n_trees",
    "qrf_breach",
    "qrf_pinball_loss",
]

missing = [
    column
    for column in required_panel_columns
    if column not in panel.columns
]

if missing:

    raise RuntimeError(
        "Existing RS01 panel is missing columns:\n"
        + "\n".join(missing)
    )


# =============================================================================
# LOAD RAW DATA USING ORIGINAL RS01 FUNCTIONS
# =============================================================================

log.info("")

log.info(
    "=" * 100
)

log.info(
    "LOADING RAW DATA"
)

log.info(
    "=" * 100
)

stocks, vix_df = L.load_raw_data()


# =============================================================================
# GENERATE CORRECTED GARCH
# =============================================================================

all_garch_rows = []


for ticker in L.TICKERS:

    log.info("")

    log.info(
        "=" * 100
    )

    log.info(
        f"PROCESSING GARCH — {ticker}"
    )

    log.info(
        "=" * 100
    )


    # =========================================================================
    # SAME FEATURE BUILDING AS ORIGINAL RS01
    # =========================================================================

    df = L.build_features(
        stocks[ticker],
        vix_df,
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
        .copy()
    )


    # =========================================================================
    # SAME USABLE SAMPLE AS ORIGINAL RS01
    # =========================================================================

    usable = df[
        df["log_return"].notna()
        & df["next_return"].notna()
        & df["realized_date"].notna()
    ].copy()

    usable = (
        usable
        .sort_values("date")
        .reset_index(drop=True)
    )


    # =========================================================================
    # SAME FORECAST POSITIONS AS ORIGINAL RS01
    # =========================================================================

    possible_positions = np.arange(
        L.WINDOW - 1,
        len(usable),
    )

    if len(possible_positions) == 0:

        raise RuntimeError(
            f"{ticker}: no valid GARCH forecast positions."
        )


    dates = pd.DatetimeIndex(
        usable["date"]
    )


    returns_by_date = (
        usable
        .set_index("date")["log_return"]
        .sort_index()
    )


    # =========================================================================
    # CONTIGUITY CHECK
    #
    # realized_date[t] must equal forecast_date[t+1].
    # =========================================================================

    if len(usable) > 1:

        contiguity = (
            usable["realized_date"]
            .iloc[:-1]
            .to_numpy()
            ==
            usable["date"]
            .iloc[1:]
            .to_numpy()
        )

        if not contiguity.all():

            bad_positions = np.where(
                ~contiguity
            )[0]

            raise RuntimeError(
                f"{ticker}: non-contiguous "
                f"forecast/realized dates at positions "
                f"{bad_positions[:10].tolist()}"
            )


    # =========================================================================
    # SAME 20-DAY REFIT BLOCKS AS ORIGINAL RS01
    # =========================================================================

    blocks = [
        possible_positions[
            start:start + L.REFIT_EVERY
        ]

        for start in range(
            0,
            len(possible_positions),
            L.REFIT_EVERY,
        )
    ]

    total_blocks = len(blocks)

    log.info(
        f"{ticker}: forecast positions = "
        f"{len(possible_positions):,}"
    )

    log.info(
        f"{ticker}: GARCH refit blocks = "
        f"{total_blocks}"
    )


    ticker_garch_rows = []


    # =========================================================================
    # REFIT BLOCK LOOP
    # =========================================================================

    for block_number, positions in enumerate(
        blocks,
        start=1,
    ):

        first_date = dates[
            positions[0]
        ]

        last_date = dates[
            positions[-1]
        ]

        print(
            f"{ticker}: "
            f"FIT {block_number:03d}/"
            f"{total_blocks:03d} | "
            f"{first_date.date()} "
            f"to "
            f"{last_date.date()}",
            flush=True,
        )


        block_rows = fit_garch_block_corrected(
            returns_by_date=returns_by_date,
            dates=dates,
            positions=positions,
            ticker=ticker,
            fit_number=block_number,
        )


        ticker_garch_rows.extend(
            block_rows
        )


    log.info(
        f"{ticker}: corrected GARCH forecasts = "
        f"{len(ticker_garch_rows):,}"
    )


    all_garch_rows.extend(
        ticker_garch_rows
    )


# =============================================================================
# BUILD GARCH DATAFRAME
# =============================================================================

garch_df = pd.DataFrame(
    all_garch_rows
)

if garch_df.empty:

    raise RuntimeError(
        "No corrected GARCH forecasts were generated."
    )

garch_df["forecast_date"] = pd.to_datetime(
    garch_df["forecast_date"],
    errors="coerce",
)


# =============================================================================
# CORRECTED GARCH COVERAGE VALIDATION
# =============================================================================

log.info("")

log.info(
    "=" * 100
)

log.info(
    "CORRECTED GARCH FORECAST VALIDATION"
)

log.info(
    "=" * 100
)

log.info(
    f"Corrected GARCH rows: "
    f"{len(garch_df):,}"
)

log.info(
    f"Tickers: "
    f"{garch_df['ticker'].nunique()}"
)


rows_per_ticker = (
    garch_df
    .groupby("ticker")
    .size()
    .sort_index()
)

log.info("")

log.info(
    "Rows per ticker:"
)

log.info(
    rows_per_ticker.to_string()
)

if rows_per_ticker.nunique() != 1:

    raise RuntimeError(
        "Corrected GARCH row counts differ across tickers."
    )


# =============================================================================
# BUILD MULTI-INDEX KEYS
# =============================================================================

panel_index = pd.MultiIndex.from_frame(
    panel[
        [
            "ticker",
            "forecast_date",
        ]
    ]
)

garch_index = pd.MultiIndex.from_frame(
    garch_df[
        [
            "ticker",
            "forecast_date",
        ]
    ]
)


# =============================================================================
# KEY COVERAGE CHECK
#
# Existing panel = common QRF-valid sample.
#
# GARCH has 480 extra early keys:
#
#     20 dates × 24 tickers = 480
# =============================================================================

missing_garch_keys = (
    panel_index
    .difference(
        garch_index
    )
)

extra_garch_keys = (
    garch_index
    .difference(
        panel_index
    )

)

log.info("")

log.info(
    f"Existing-panel keys missing corrected GARCH: "
    f"{len(missing_garch_keys)}"
)

log.info(
    f"Corrected GARCH keys absent from existing panel: "
    f"{len(extra_garch_keys)}"
)


if len(missing_garch_keys) != 0:

    raise RuntimeError(
        "Corrected GARCH is missing keys required "
        "by the existing RS01 panel."
    )


expected_extra_keys = (
    len(L.TICKERS)
    * 20
)

if len(extra_garch_keys) != expected_extra_keys:

    raise RuntimeError(
        f"Unexpected number of extra corrected GARCH keys: "
        f"{len(extra_garch_keys)}; "
        f"expected {expected_extra_keys}."
    )

log.info(
    f"Extra corrected GARCH keys outside common RS01 sample: "
    f"{len(extra_garch_keys)} "
    f"(expected {expected_extra_keys})"
)


# =============================================================================
# ALIGN CORRECTED GARCH TO EXISTING COMMON PANEL
# =============================================================================

garch_indexed = (
    garch_df
    .set_index(
        [
            "ticker",
            "forecast_date",
        ]
    )
)

garch_aligned = (
    garch_indexed
    .reindex(
        panel_index
    )
    .reset_index()
)


# =============================================================================
# ACTUAL RETURNS
# =============================================================================

actual_return = (
    panel["actual_return"]
    .to_numpy(
        dtype=float
    )
)


# =============================================================================
# CORRECTED GARCH VAR
# =============================================================================

corrected_var = (
    garch_aligned["garch_var"]
    .to_numpy(
        dtype=float
    )
)


# =============================================================================
# CORRECTED BREACH
# =============================================================================

corrected_breach = np.where(
    np.isfinite(corrected_var)
    &
    np.isfinite(actual_return),

    (
        actual_return
        < corrected_var
    ).astype(float),

    np.nan,
)


# =============================================================================
# CORRECTED PINBALL LOSS
# =============================================================================

corrected_loss = np.full(
    len(panel),
    np.nan,
)

valid = (
    np.isfinite(corrected_var)
    &
    np.isfinite(actual_return)
)

error = (
    actual_return[valid]
    - corrected_var[valid]
)

corrected_loss[valid] = np.where(
    error >= 0,

    L.ALPHA
    * error,

    (L.ALPHA - 1.0)
    * error,
)


# =============================================================================
# STORE OLD GARCH FOR COMPARISON
# =============================================================================

old_garch_var = (
    panel["garch_var"]
    .to_numpy(
        dtype=float
    )
)

old_garch_breach = (
    panel["garch_breach"]
    .to_numpy(
        dtype=float
    )
)

old_garch_loss = (
    panel["garch_pinball_loss"]
    .to_numpy(
        dtype=float
    )
)

old_valid = (
    np.isfinite(old_garch_var)
    &
    np.isfinite(old_garch_breach)
    &
    np.isfinite(old_garch_loss)
)


# =============================================================================
# OLD VS NEW FORECAST DIFFERENCES
# =============================================================================

var_difference = (
    corrected_var
    - old_garch_var
)

finite_var_difference = (
    np.isfinite(
        var_difference
    )
)

log.info("")

log.info(
    "=" * 100
)

log.info(
    "OLD VS CORRECTED GARCH FORECASTS"
)

log.info(
    "=" * 100
)

log.info(
    f"Forecasts compared: "
    f"{finite_var_difference.sum():,}"
)

log.info(
    f"Mean absolute VaR change: "
    f"{np.nanmean(np.abs(var_difference)):.10f}"
)

log.info(
    f"Maximum absolute VaR change: "
    f"{np.nanmax(np.abs(var_difference)):.10f}"
)

log.info(
    f"Mean signed VaR change: "
    f"{np.nanmean(var_difference):.10f}"
)


# =============================================================================
# REPLACE ONLY GARCH COLUMNS
# =============================================================================

panel["garch_var"] = (
    corrected_var
)

panel["garch_mu"] = (
    garch_aligned["garch_mu"]
    .to_numpy(
        dtype=float
    )
)

panel["garch_sigma"] = (
    garch_aligned["garch_sigma"]
    .to_numpy(
        dtype=float
    )
)

panel["garch_df"] = (
    garch_aligned["garch_df"]
    .to_numpy(
        dtype=float
    )
)

panel["garch_fit_number"] = (
    garch_aligned["garch_fit_number"]
    .to_numpy(
        dtype=float
    )
)

panel["garch_breach"] = (
    corrected_breach
)

panel["garch_pinball_loss"] = (
    corrected_loss
)


# =============================================================================
# CORRECTED GARCH METRICS
# =============================================================================

valid_garch = (
    np.isfinite(
        panel["garch_var"]
    )
    &
    np.isfinite(
        panel["garch_breach"]
    )
    &
    np.isfinite(
        panel["garch_pinball_loss"]
    )
)

log.info("")

log.info(
    "=" * 100
)

log.info(
    "CORRECTED GARCH METRICS"
)

log.info(
    "=" * 100
)

log.info(
    f"Valid GARCH rows: "
    f"{valid_garch.sum():,}"
)

log.info(
    f"GARCH breach rate: "
    f"{panel.loc[valid_garch, 'garch_breach'].mean():.6f}"
)

log.info(
    f"Mean |GARCH VaR|: "
    f"{panel.loc[valid_garch, 'garch_var'].abs().mean():.6f}"
)

log.info(
    f"Mean GARCH pinball loss: "
    f"{panel.loc[valid_garch, 'garch_pinball_loss'].mean():.6f}"
)


# =============================================================================
# PRE-FIX GARCH METRICS
# =============================================================================

log.info("")

log.info(
    "-" * 100
)

log.info(
    "PRE-FIX GARCH"
)

log.info(
    "-" * 100
)

log.info(
    f"Valid rows: "
    f"{old_valid.sum():,}"
)

log.info(
    f"Breach rate: "
    f"{np.nanmean(old_garch_breach):.6f}"
)

log.info(
    f"Mean |GARCH VaR|: "
    f"{np.nanmean(np.abs(old_garch_var)):.6f}"
)

log.info(
    f"Mean pinball loss: "
    f"{np.nanmean(old_garch_loss):.6f}"
)


# =============================================================================
# DIRECT OLD VS NEW COMPARISON
# =============================================================================

new_valid = (
    np.isfinite(corrected_var)
    &
    np.isfinite(corrected_breach)
    &
    np.isfinite(corrected_loss)
)

common_valid = (
    old_valid
    &
    new_valid
)

log.info("")

log.info(
    "-" * 100
)

log.info(
    "DIRECT OLD VS NEW COMPARISON"
)

log.info(
    "-" * 100
)

log.info(
    f"Common valid rows: "
    f"{common_valid.sum():,}"
)

if common_valid.any():

    old_breach_rate_common = (
        old_garch_breach[
            common_valid
        ].mean()
    )

    new_breach_rate_common = (
        corrected_breach[
            common_valid
        ].mean()
    )

    old_loss_common = (
        old_garch_loss[
            common_valid
        ].mean()
    )

    new_loss_common = (
        corrected_loss[
            common_valid
        ].mean()
    )

    old_abs_var_common = (
        np.abs(
            old_garch_var[
                common_valid
            ]
        ).mean()
    )

    new_abs_var_common = (
        np.abs(
            corrected_var[
                common_valid
            ]
        ).mean()
    )

    log.info(
        f"Old breach rate: "
        f"{old_breach_rate_common:.6f}"
    )

    log.info(
        f"New breach rate: "
        f"{new_breach_rate_common:.6f}"
    )

    log.info(
        f"Old mean pinball: "
        f"{old_loss_common:.6f}"
    )

    log.info(
        f"New mean pinball: "
        f"{new_loss_common:.6f}"
    )

    log.info(
        f"Old mean |VaR|: "
        f"{old_abs_var_common:.6f}"
    )

    log.info(
        f"New mean |VaR|: "
        f"{new_abs_var_common:.6f}"
    )


# =============================================================================
# FINAL PANEL INTEGRITY
# =============================================================================

log.info("")

log.info(
    "=" * 100
)

log.info(
    "FINAL PANEL INTEGRITY CHECK"
)

log.info(
    "=" * 100
)

log.info(
    f"Final panel rows: "
    f"{len(panel):,}"
)


# =============================================================================
# DUPLICATE CHECK
# =============================================================================

duplicate_keys = (
    panel
    .duplicated(
        subset=[
            "ticker",
            "forecast_date",
        ]
    )
    .sum()
)

log.info(
    f"Duplicate ticker + forecast_date rows: "
    f"{duplicate_keys}"
)

if duplicate_keys != 0:

    raise RuntimeError(
        "Duplicate ticker + forecast_date rows detected."
    )


# =============================================================================
# LOAD BACKUP FOR NON-GARCH COMPARISON
# =============================================================================

before = pd.read_csv(
    BACKUP_PATH
)

before["forecast_date"] = pd.to_datetime(
    before["forecast_date"],
    errors="coerce",
)

before["realized_date"] = pd.to_datetime(
    before["realized_date"],
    errors="coerce",
)


# =============================================================================
# SORT BOTH PANELS
# =============================================================================

before_sorted = (
    before
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(
        drop=True
    )
)

after_sorted = (
    panel
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# CHECK ROW COUNT
# =============================================================================

if len(before_sorted) != len(after_sorted):

    raise RuntimeError(
        "Backup and final panel have different row counts."
    )


# =============================================================================
# VERIFY NON-GARCH COLUMNS WERE NOT CHANGED
#
# Datetime columns are explicitly normalized before comparison.
# =============================================================================

non_garch_columns = [
    column
    for column in before_sorted.columns
    if not column.startswith(
        "garch_"
    )
    and column in after_sorted.columns
]


non_garch_changed = []


for column in non_garch_columns:

    before_series = (
        before_sorted[column]
    )

    after_series = (
        after_sorted[column]
    )


    # -------------------------------------------------------------------------
    # Datetime columns
    # -------------------------------------------------------------------------

    if (
        pd.api.types.is_datetime64_any_dtype(
            before_series
        )
        or
        pd.api.types.is_datetime64_any_dtype(
            after_series
        )
        or
        column in {
            "forecast_date",
            "realized_date",
        }
    ):

        before_dt = pd.to_datetime(
            before_series,
            errors="coerce",
        )

        after_dt = pd.to_datetime(
            after_series,
            errors="coerce",
        )

        same = (
            before_dt.equals(
                after_dt
            )
        )


    # -------------------------------------------------------------------------
    # Numeric columns
    # -------------------------------------------------------------------------

    elif (
        pd.api.types.is_numeric_dtype(
            before_series
        )
    ):

        same = np.allclose(
            before_series.to_numpy(
                dtype=float
            ),
            after_series.to_numpy(
                dtype=float
            ),
            equal_nan=True,
        )


    # -------------------------------------------------------------------------
    # Other columns
    # -------------------------------------------------------------------------

    else:

        same = (
            before_series
            .astype(str)
            .equals(
                after_series.astype(str)
            )
        )


    if not same:

        non_garch_changed.append(
            column
        )


log.info("")

log.info(
    f"Non-GARCH columns changed: "
    f"{non_garch_changed}"
)


if non_garch_changed:

    raise RuntimeError(
        "Unexpected non-GARCH columns changed:\n"
        + "\n".join(
            non_garch_changed
        )
    )

log.info(
    "HS / EWMA / QRF / timing columns preserved: OK"
)


# =============================================================================
# COMMON PANEL SIZE CHECK
# =============================================================================

if len(panel) != 68952:

    log.warning(
        f"Expected common RS01 panel size = 68,952; "
        f"observed = {len(panel):,}"
    )

else:

    log.info(
        "Common RS01 panel size = 68,952: OK"
    )


# =============================================================================
# MODEL COVERAGE CHECK
# =============================================================================

model_columns = {

    "HS": [
        "hs_var",
        "hs_breach",
        "hs_pinball_loss",
    ],

    "EWMA": [
        "ewma_var",
        "ewma_breach",
        "ewma_pinball_loss",
    ],

    "GARCH-t": [
        "garch_var",
        "garch_breach",
        "garch_pinball_loss",
    ],

    "QRF": [
        "qrf_var",
        "qrf_breach",
        "qrf_pinball_loss",
    ],
}


log.info("")

log.info(
    "MODEL COVERAGE:"
)


for model_name, columns in model_columns.items():

    model_valid = (
        panel[columns]
        .notna()
        .all(
            axis=1
        )
    )

    log.info(
        f"{model_name:8s}: "
        f"{model_valid.sum():,} / "
        f"{len(panel):,}"
    )

    if model_valid.sum() != len(panel):

        raise RuntimeError(
            f"{model_name} does not have complete "
            f"coverage on the common RS01 panel."
        )


# =============================================================================
# GARCH KEY UNIQUENESS
# =============================================================================

garch_duplicate_keys = (
    garch_df
    .duplicated(
        subset=[
            "ticker",
            "forecast_date",
        ]
    )
    .sum()
)

log.info("")

log.info(
    f"Duplicate corrected GARCH keys: "
    f"{garch_duplicate_keys}"
)

if garch_duplicate_keys != 0:

    raise RuntimeError(
        "Duplicate corrected GARCH keys detected."
    )


# =============================================================================
# SAVE CORRECTED PANEL
# =============================================================================

panel.to_csv(
    PANEL_PATH,
    index=False,
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

log.info("")

log.info(
    "=" * 100
)

log.info(
    "OUTPUT SAVED"
)

log.info(
    "=" * 100
)

log.info(
    f"Corrected RS01 panel:\n"
    f"{PANEL_PATH}"
)

log.info(
    f"Pre-fix backup:\n"
    f"{BACKUP_PATH}"
)

log.info(
    f"Correction log:\n"
    f"{LOG_PATH}"
)

log.info("")

log.info(
    "=" * 100
)

log.info(
    "GARCH-ONLY RS01 CORRECTION COMPLETE"
)

log.info(
    "=" * 100
)