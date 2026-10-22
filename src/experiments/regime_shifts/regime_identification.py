from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# RS03 — DATA-DRIVEN REGIME IDENTIFICATION
#
# Goal:
# Identify NORMAL / TRANSITION / STRESS / RECOVERY market regimes using
# information available at time t only.
#
# Core market variables:
#   1. VIX level
#   2. Cross-sectional median 20-day realized volatility
#   3. 10-day changes in VIX and realized volatility
#
# No future VaR outcomes are used to define regimes.
# =============================================================================


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path(r"D:\reliability_layer")

RAW_DIR = ROOT / "data" / "raw"
RAW_ZIP = ROOT / "data" / "raw.zip"

RS01_PANEL_PATH = (
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


REGIME_DAILY_PATH = (
    OUTPUT_DIR / "RS03_regime_daily.csv"
)

REGIME_SUMMARY_PATH = (
    OUTPUT_DIR / "RS03_regime_summary.csv"
)

REGIME_EVENTS_PATH = (
    OUTPUT_DIR / "RS03_regime_events.csv"
)

RS01_REGIME_PANEL_PATH = (
    OUTPUT_DIR / "RS03_RS01_regime_panel.csv"
)

LOG_PATH = (
    OUTPUT_DIR / "RS03_regime_identification.log"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

RV_WINDOW = 20
CHANGE_WINDOW = 10

# Expanding quantile thresholds.
# IMPORTANT:
# These are calculated from information available strictly BEFORE date t.
LEVEL_LOW_Q = 0.75
LEVEL_STRESS_Q = 0.90

CHANGE_STRESS_Q = 0.90

# Minimum number of prior observations required before thresholds are used.
MIN_HISTORY = 252

# Minimum consecutive stress observations used to characterize an event.
MIN_STRESS_DAYS = 3


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
# TICKERS
# =============================================================================

EXPECTED_TICKERS = [
    "AAPL",
    "AMZN",
    "BA",
    "BAC",
    "CAT",
    "CVX",
    "DIS",
    "GME",
    "GOOGL",
    "GS",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MSFT",
    "NFLX",
    "NVDA",
    "PFE",
    "PG",
    "TSLA",
    "UNH",
    "V",
    "WMT",
    "XOM",
]


# =============================================================================
# HELPERS
# =============================================================================

def normalize_columns(df):
    """
    Normalize column names to lowercase.
    """

    df = df.copy()

    df.columns = [
        str(c).strip().lower()
        for c in df.columns
    ]

    return df


def find_column(columns, candidates):
    """
    Find the first matching candidate column.
    """

    columns_lower = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:
        if candidate.lower() in columns_lower:
            return columns_lower[candidate.lower()]

    return None


def load_raw_market_data():
    """
    Load raw daily market data.

    Supports both formats:

    1. One ticker per CSV:
         AAPL.csv
         MSFT.csv
         ...
         VIX.csv

       where the CSV contains Date / Close / OHLCV columns but no ticker
       column. In this case the ticker is inferred from the filename.

    2. Combined CSVs containing a ticker/symbol column.

    COIN is explicitly excluded from the final 24-ticker universe.
    """

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw directory not found:\n{RAW_DIR}\n\n"
            f"Expected raw data at:\n{RAW_ZIP}"
        )

    csv_files = sorted(
        RAW_DIR.rglob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found under:\n{RAW_DIR}"
        )

    log.info(
        f"Found {len(csv_files)} CSV files under raw directory."
    )

    ticker_frames = []
    vix_frames = []

    for path in csv_files:

        try:
            df = pd.read_csv(path)

        except Exception as exc:
            log.warning(
                f"Could not read {path}: {exc}"
            )
            continue

        df = normalize_columns(df)

        # ---------------------------------------------------------------------
        # Locate date and close columns
        # ---------------------------------------------------------------------

        date_col = find_column(
            df.columns,
            [
                "date",
                "datetime",
                "timestamp",
            ],
        )

        close_col = find_column(
            df.columns,
            [
                "close",
                "adj close",
                "adj_close",
            ],
        )

        if date_col is None or close_col is None:
            log.warning(
                f"Skipping {path.name}: "
                f"date/close column not found."
            )
            continue

        # ---------------------------------------------------------------------
        # Look for an explicit ticker column
        # ---------------------------------------------------------------------

        ticker_col = find_column(
            df.columns,
            [
                "ticker",
                "symbol",
                "stock",
                "name",
            ],
        )

        # ---------------------------------------------------------------------
        # CASE 1 — CSV contains a ticker column
        # ---------------------------------------------------------------------

        if ticker_col is not None:

            ticker_values = (
                df[ticker_col]
                .astype(str)
                .str.upper()
                .str.strip()
            )

            # -------------------------------------------------------------
            # Split VIX from stock data
            # -------------------------------------------------------------

            vix_mask = ticker_values.isin(
                {
                    "VIX",
                    "^VIX",
                    "VIXCLS",
                }
            )

            if vix_mask.any():

                tmp_vix = df.loc[
                    vix_mask,
                    [
                        date_col,
                        close_col,
                    ]
                ].copy()

                tmp_vix.columns = [
                    "date",
                    "vix",
                ]

                tmp_vix["date"] = pd.to_datetime(
                    tmp_vix["date"],
                    errors="coerce",
                )

                tmp_vix["vix"] = pd.to_numeric(
                    tmp_vix["vix"],
                    errors="coerce",
                )

                tmp_vix = tmp_vix.dropna(
                    subset=[
                        "date",
                        "vix",
                    ]
                )

                if not tmp_vix.empty:
                    vix_frames.append(tmp_vix)

            # -------------------------------------------------------------
            # Stock rows
            # -------------------------------------------------------------

            stock_mask = (
                ticker_values.isin(
                    EXPECTED_TICKERS
                )
            )

            if stock_mask.any():

                tmp_stock = df.loc[
                    stock_mask,
                    [
                        date_col,
                        ticker_col,
                        close_col,
                    ]
                ].copy()

                tmp_stock.columns = [
                    "date",
                    "ticker",
                    "close",
                ]

                tmp_stock["date"] = pd.to_datetime(
                    tmp_stock["date"],
                    errors="coerce",
                )

                tmp_stock["ticker"] = (
                    tmp_stock["ticker"]
                    .astype(str)
                    .str.upper()
                    .str.strip()
                )

                tmp_stock["close"] = pd.to_numeric(
                    tmp_stock["close"],
                    errors="coerce",
                )

                tmp_stock = tmp_stock.dropna(
                    subset=[
                        "date",
                        "close",
                    ]
                )

                tmp_stock = tmp_stock[
                    tmp_stock["close"] > 0
                ]

                if not tmp_stock.empty:
                    ticker_frames.append(
                        tmp_stock
                    )

            continue

        # ---------------------------------------------------------------------
        # CASE 2 — One ticker per CSV, infer ticker from filename
        # ---------------------------------------------------------------------

        inferred_ticker = (
            path.stem
            .upper()
            .strip()
        )

        # Handle common VIX filename variants.
        inferred_is_vix = (
            inferred_ticker in {
                "VIX",
                "^VIX",
                "VIXCLS",
            }
            or "VIX" in inferred_ticker
        )

        # -------------------------------------------------------------
        # VIX file
        # -------------------------------------------------------------

        if inferred_is_vix:

            tmp_vix = df[
                [
                    date_col,
                    close_col,
                ]
            ].copy()

            tmp_vix.columns = [
                "date",
                "vix",
            ]

            tmp_vix["date"] = pd.to_datetime(
                tmp_vix["date"],
                errors="coerce",
            )

            tmp_vix["vix"] = pd.to_numeric(
                tmp_vix["vix"],
                errors="coerce",
            )

            tmp_vix = tmp_vix.dropna(
                subset=[
                    "date",
                    "vix",
                ]
            )

            tmp_vix = tmp_vix[
                tmp_vix["vix"] > 0
            ]

            if not tmp_vix.empty:
                vix_frames.append(
                    tmp_vix
                )

            continue

        # -------------------------------------------------------------
        # Explicitly exclude COIN
        # -------------------------------------------------------------

        if inferred_ticker == "COIN":
            log.info(
                "Excluding COIN from final 24-ticker universe."
            )
            continue

        # -------------------------------------------------------------
        # Expected stock ticker
        # -------------------------------------------------------------

        if inferred_ticker not in EXPECTED_TICKERS:

            log.warning(
                f"Skipping {path.name}: "
                f"inferred ticker '{inferred_ticker}' "
                f"is not in the expected 24-ticker universe."
            )

            continue

        tmp_stock = df[
            [
                date_col,
                close_col,
            ]
        ].copy()

        tmp_stock["ticker"] = (
            inferred_ticker
        )

        tmp_stock = tmp_stock[
            [
                date_col,
                "ticker",
                close_col,
            ]
        ]

        tmp_stock.columns = [
            "date",
            "ticker",
            "close",
        ]

        tmp_stock["date"] = pd.to_datetime(
            tmp_stock["date"],
            errors="coerce",
        )

        tmp_stock["close"] = pd.to_numeric(
            tmp_stock["close"],
            errors="coerce",
        )

        tmp_stock = tmp_stock.dropna(
            subset=[
                "date",
                "close",
            ]
        )

        tmp_stock = tmp_stock[
            tmp_stock["close"] > 0
        ]

        if not tmp_stock.empty:
            ticker_frames.append(
                tmp_stock
            )

    # =========================================================================
    # VALIDATE STOCK DATA
    # =========================================================================

    if not ticker_frames:
        raise RuntimeError(
            "No stock CSV files could be loaded.\n\n"
            "The loader checked both explicit ticker columns and "
            "ticker names inferred from filenames."
        )

    stocks = pd.concat(
        ticker_frames,
        ignore_index=True,
    )

    stocks = (
        stocks
        .drop_duplicates(
            subset=[
                "ticker",
                "date",
            ]
        )
        .sort_values(
            [
                "ticker",
                "date",
            ]
        )
        .reset_index(drop=True)
    )

    found_tickers = sorted(
        stocks["ticker"].unique()
    )

    missing_tickers = sorted(
        set(EXPECTED_TICKERS)
        - set(found_tickers)
    )

    extra_tickers = sorted(
        set(found_tickers)
        - set(EXPECTED_TICKERS)
    )

    if missing_tickers:
        raise RuntimeError(
            "Missing required tickers:\n"
            + ", ".join(missing_tickers)
        )

    if extra_tickers:
        log.warning(
            "Unexpected tickers found:\n"
            + ", ".join(extra_tickers)
        )

    # =========================================================================
    # VALIDATE VIX
    # =========================================================================

    if not vix_frames:
        raise RuntimeError(
            "VIX data could not be located."
        )

    vix = pd.concat(
        vix_frames,
        ignore_index=True,
    )

    vix = (
        vix
        .drop_duplicates(
            subset=["date"]
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # =========================================================================
    # RAW DATA CHECK
    # =========================================================================

    log.info("")
    log.info("RAW DATA CHECK")
    log.info("-" * 100)

    log.info(
        f"Stock rows: {len(stocks):,}"
    )

    log.info(
        f"Tickers: {stocks['ticker'].nunique()}"
    )

    log.info(
        f"Stock date range: "
        f"{stocks['date'].min().date()} "
        f"to "
        f"{stocks['date'].max().date()}"
    )

    log.info(
        f"VIX rows: {len(vix):,}"
    )

    log.info(
        f"VIX date range: "
        f"{vix['date'].min().date()} "
        f"to "
        f"{vix['date'].max().date()}"
    )

    # -------------------------------------------------------------------------
    # Per-ticker row counts
    # -------------------------------------------------------------------------

    ticker_counts = (
        stocks
        .groupby("ticker")
        .size()
        .sort_index()
    )

    log.info("")
    log.info("Rows per ticker:")
    log.info("")
    log.info(
        ticker_counts.to_string()
    )

    # -------------------------------------------------------------------------
    # VIX sanity check
    # -------------------------------------------------------------------------

    if vix["vix"].isna().any():
        raise RuntimeError(
            "VIX contains missing values after cleaning."
        )

    if (vix["vix"] <= 0).any():
        raise RuntimeError(
            "VIX contains non-positive values."
        )

    return stocks, vix


def build_market_features(stocks, vix):
    """
    Build strictly observable market-state variables.

    For every date t:

        log_return_i,t
        RV20_i,t
        median_RV20_t
        VIX_t

    RV20 uses returns ending on t, which is observable at the close of t.

    Changes:

        VIX_change_10d_t
        RV20_change_10d_t

    are also observable at t.
    """

    stocks = stocks.copy()

    stocks["log_return"] = (
        stocks
        .groupby("ticker")["close"]
        .transform(
            lambda x: np.log(x / x.shift(1))
        )
    )

    stocks["rv20"] = (
        stocks
        .groupby("ticker")["log_return"]
        .transform(
            lambda x: x.rolling(
                RV_WINDOW,
                min_periods=RV_WINDOW,
            ).std()
            * np.sqrt(252.0)
        )
    )

    market_rv = (
        stocks
        .groupby("date")["rv20"]
        .median()
        .rename("median_rv20")
        .reset_index()
    )

    market = (
        market_rv
        .merge(
            vix[
                [
                    "date",
                    "vix",
                ]
            ],
            on="date",
            how="inner",
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    market["vix_change_10d"] = (
        market["vix"]
        - market["vix"].shift(CHANGE_WINDOW)
    )

    market["rv20_change_10d"] = (
        market["median_rv20"]
        - market["median_rv20"].shift(CHANGE_WINDOW)
    )

    market["vix_pct_change_10d"] = (
        market["vix"]
        / market["vix"].shift(CHANGE_WINDOW)
        - 1.0
    )

    market["rv20_pct_change_10d"] = (
        market["median_rv20"]
        / market["median_rv20"].shift(CHANGE_WINDOW)
        - 1.0
    )

    return market


# =============================================================================
# CAUSAL EXPANDING THRESHOLDS
# =============================================================================

def add_causal_thresholds(market):
    """
    Calculate thresholds using historical information only.

    Example:

        threshold_t
        =
        quantile(
            observations before t
        )

    The shift(1) is essential.
    """

    market = market.copy()

    # -------------------------------------------------------------------------
    # Level thresholds
    # -------------------------------------------------------------------------

    market["vix_q75"] = (
        market["vix"]
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(LEVEL_LOW_Q)
        .shift(1)
    )

    market["vix_q90"] = (
        market["vix"]
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(LEVEL_STRESS_Q)
        .shift(1)
    )

    market["rv20_q75"] = (
        market["median_rv20"]
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(LEVEL_LOW_Q)
        .shift(1)
    )

    market["rv20_q90"] = (
        market["median_rv20"]
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(LEVEL_STRESS_Q)
        .shift(1)
    )


    # -------------------------------------------------------------------------
    # Positive-change stress thresholds
    # -------------------------------------------------------------------------

    positive_vix_change = (
        market["vix_change_10d"]
        .where(
            market["vix_change_10d"] > 0
        )
    )

    positive_rv_change = (
        market["rv20_change_10d"]
        .where(
            market["rv20_change_10d"] > 0
        )
    )

    market["vix_change_q90"] = (
        positive_vix_change
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(CHANGE_STRESS_Q)
        .shift(1)
    )

    market["rv20_change_q90"] = (
        positive_rv_change
        .expanding(
            min_periods=MIN_HISTORY
        )
        .quantile(CHANGE_STRESS_Q)
        .shift(1)
    )

    return market


# =============================================================================
# REGIME SCORING
# =============================================================================

def add_regime_signals(market):
    """
    Construct causal regime indicators.

    LEVEL STRESS:
        VIX >= historical 90th percentile
        OR
        RV20 >= historical 90th percentile

    TRANSITION:
        rapid increase in VIX or RV20

    NORMAL:
        neither stress nor transition

    RECOVERY:
        a previous stress/transition state followed by a meaningful decline
        in market stress conditions.

    The actual final regime is assigned by the state machine below.
    """

    market = market.copy()

    market["vix_stress"] = (
        market["vix"]
        >= market["vix_q90"]
    )

    market["rv20_stress"] = (
        market["median_rv20"]
        >= market["rv20_q90"]
    )

    market["level_stress"] = (
        market["vix_stress"]
        | market["rv20_stress"]
    )

    market["vix_transition"] = (
        market["vix_change_10d"]
        >= market["vix_change_q90"]
    )

    market["rv20_transition"] = (
        market["rv20_change_10d"]
        >= market["rv20_change_q90"]
    )

    market["transition_signal"] = (
        market["vix_transition"]
        | market["rv20_transition"]
    )

    # -------------------------------------------------------------------------
    # Stress intensity score
    #
    # This is NOT a probability.
    #
    # It measures how many observable stress dimensions are simultaneously
    # elevated.
    # -------------------------------------------------------------------------

    market["stress_dimensions"] = (
        market["vix_stress"].astype(int)
        + market["rv20_stress"].astype(int)
        + market["vix_transition"].astype(int)
        + market["rv20_transition"].astype(int)
    )

    return market


def assign_causal_regimes(market):
    """
    Stateful causal regime assignment.

    Important:
    Only current and past observations are used.

    States:
        normal
        transition
        stress
        recovery

    Logic:

        stress:
            current level is extreme

        transition:
            current level is not yet extreme, but volatility is rising sharply

        recovery:
            market was recently in a stress/transition state and stress
            conditions are now declining

        normal:
            none of the above

    Recovery requires:
        - previous state was transition/stress/recovery
        - current level is below stress threshold
        - both VIX and RV20 are not simultaneously rising sharply
        - at least one risk measure has fallen relative to its 10d level
    """

    market = market.copy()

    regimes = []

    previous_state = "normal"

    for _, row in market.iterrows():

        current_level_stress = bool(
            row["level_stress"]
        ) if pd.notna(
            row["level_stress"]
        ) else False

        current_transition = bool(
            row["transition_signal"]
        ) if pd.notna(
            row["transition_signal"]
        ) else False

        vix_falling = (
            pd.notna(row["vix_change_10d"])
            and row["vix_change_10d"] < 0
        )

        rv_falling = (
            pd.notna(row["rv20_change_10d"])
            and row["rv20_change_10d"] < 0
        )

        recovering = (
            previous_state in {
                "transition",
                "stress",
                "recovery",
            }
            and not current_level_stress
            and not current_transition
            and (vix_falling or rv_falling)
        )

        if current_level_stress:

            current_state = "stress"

        elif current_transition:

            current_state = "transition"

        elif recovering:

            current_state = "recovery"

        else:

            current_state = "normal"

        regimes.append(current_state)

        previous_state = current_state

    market["regime"] = regimes

    return market


# =============================================================================
# EVENT EXTRACTION
# =============================================================================

def extract_regime_events(market):
    """
    Extract contiguous transition/stress/recovery periods.

    Each contiguous regime receives an event id.
    """

    market = market.copy()

    market["regime_change"] = (
        market["regime"]
        != market["regime"].shift(1)
    )

    market["regime_episode_id"] = (
        market["regime_change"]
        .cumsum()
    )

    episodes = (
        market
        .groupby(
            [
                "regime_episode_id",
                "regime",
            ],
            as_index=False,
        )
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            n_days=("date", "size"),
            max_vix=("vix", "max"),
            max_rv20=("median_rv20", "max"),
            max_stress_dimensions=(
                "stress_dimensions",
                "max",
            ),
        )
    )

    episodes["duration_days"] = (
        episodes["n_days"]
    )

    # -------------------------------------------------------------------------
    # Keep meaningful stress episodes
    # -------------------------------------------------------------------------

    stress_episodes = episodes[
        (
            episodes["regime"] == "stress"
        )
        & (
            episodes["n_days"]
            >= MIN_STRESS_DAYS
        )
    ].copy()

    stress_episodes["stress_event_id"] = (
        np.arange(len(stress_episodes))
        + 1
    )

    return episodes, stress_episodes


# =============================================================================
# RS01 MERGE
# =============================================================================

def attach_regimes_to_rs01(
    market,
    rs01_panel,
):
    """
    Attach the market regime at forecast date t to every RS01 VaR forecast.

    This means:
        regime_t
        -> VaR forecast made using information through t
        -> realized return at t+1

    This preserves the causal ordering.
    """

    rs01 = rs01_panel.copy()

    rs01["forecast_date"] = pd.to_datetime(
        rs01["forecast_date"]
    )

    regime_columns = [
        "date",
        "regime",
        "vix",
        "median_rv20",
        "vix_change_10d",
        "rv20_change_10d",
        "vix_pct_change_10d",
        "rv20_pct_change_10d",
        "stress_dimensions",
    ]

    rs01 = rs01.merge(
        market[regime_columns],
        left_on="forecast_date",
        right_on="date",
        how="left",
    )

    rs01 = rs01.drop(
        columns=["date"],
        errors="ignore",
    )

    return rs01


# =============================================================================
# MAIN
# =============================================================================

log.info("=" * 100)
log.info("RS03 — DATA-DRIVEN REGIME IDENTIFICATION")
log.info("=" * 100)

log.info("")
log.info("Core principle:")
log.info(
    "Regime at date t is determined only from information available by date t."
)

log.info("")
log.info(
    "Loading raw market data..."
)

stocks, vix = load_raw_market_data()


# =============================================================================
# BUILD FEATURES
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("BUILDING MARKET FEATURES")
log.info("=" * 100)

market = build_market_features(
    stocks,
    vix,
)

log.info(
    f"Market dates: {len(market):,}"
)

log.info(
    f"Market date range: "
    f"{market['date'].min().date()} "
    f"to "
    f"{market['date'].max().date()}"
)


# =============================================================================
# CAUSAL THRESHOLDS
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("BUILDING CAUSAL EXPANDING THRESHOLDS")
log.info("=" * 100)

add_thresholds = add_causal_thresholds(
    market
)

market = add_thresholds


# =============================================================================
# SIGNALS
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("BUILDING REGIME SIGNALS")
log.info("=" * 100)

market = add_regime_signals(
    market
)


# =============================================================================
# REGIME ASSIGNMENT
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("ASSIGNING CAUSAL REGIMES")
log.info("=" * 100)

market = assign_causal_regimes(
    market
)


# Remove dates for which the causal thresholds
# could not yet be established.

market_valid = market[
    market["vix_q90"].notna()
    & market["rv20_q90"].notna()
    & market["vix_change_q90"].notna()
    & market["rv20_change_q90"].notna()
].copy()

log.info(
    f"Dates with valid causal thresholds: "
    f"{len(market_valid):,}"
)


# =============================================================================
# REGIME DISTRIBUTION
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("REGIME DISTRIBUTION")
log.info("=" * 100)

regime_counts = (
    market_valid["regime"]
    .value_counts()
    .reindex(
        [
            "normal",
            "transition",
            "stress",
            "recovery",
        ]
    )
)

regime_share = (
    regime_counts
    / regime_counts.sum()
)

regime_distribution = pd.DataFrame({
    "regime": regime_counts.index,
    "n_days": regime_counts.values,
    "share": regime_share.values,
})

log.info("")
log.info(
    regime_distribution.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# REGIME METRICS
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("REGIME CHARACTERISTICS")
log.info("=" * 100)

regime_summary = (
    market_valid
    .groupby("regime")
    .agg(
        n_days=("date", "size"),
        mean_vix=("vix", "mean"),
        median_vix=("vix", "median"),
        max_vix=("vix", "max"),
        mean_rv20=("median_rv20", "mean"),
        median_rv20=("median_rv20", "median"),
        max_rv20=("median_rv20", "max"),
        mean_vix_change_10d=(
            "vix_change_10d",
            "mean",
        ),
        mean_rv20_change_10d=(
            "rv20_change_10d",
            "mean",
        ),
        mean_stress_dimensions=(
            "stress_dimensions",
            "mean",
        ),
    )
    .reset_index()
)

regime_summary["share"] = (
    regime_summary["n_days"]
    / regime_summary["n_days"].sum()
)

log.info("")
log.info(
    regime_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)


# =============================================================================
# REGIME EPISODES
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("EXTRACTING REGIME EPISODES")
log.info("=" * 100)

episodes, stress_episodes = extract_regime_events(
    market_valid
)

log.info(
    f"Total regime episodes: "
    f"{len(episodes):,}"
)

log.info(
    f"Meaningful stress episodes: "
    f"{len(stress_episodes):,}"
)

if not stress_episodes.empty:

    display_cols = [
        "stress_event_id",
        "start_date",
        "end_date",
        "n_days",
        "max_vix",
        "max_rv20",
        "max_stress_dimensions",
    ]

    log.info("")
    log.info(
        stress_episodes[
            display_cols
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )


# =============================================================================
# MERGE WITH RS01
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("ATTACHING REGIMES TO RS01")
log.info("=" * 100)

if not RS01_PANEL_PATH.exists():

    raise FileNotFoundError(
        f"RS01 panel not found:\n"
        f"{RS01_PANEL_PATH}"
    )

rs01 = pd.read_csv(
    RS01_PANEL_PATH
)

log.info(
    f"RS01 rows: {len(rs01):,}"
)

rs01_regime = attach_regimes_to_rs01(
    market_valid,
    rs01,
)

log.info(
    f"RS01 + regime rows: "
    f"{len(rs01_regime):,}"
)

matched = (
    rs01_regime["regime"]
    .notna()
    .sum()
)

log.info(
    f"RS01 rows with identified regime: "
    f"{matched:,} / {len(rs01_regime):,}"
)


# =============================================================================
# SAVE DAILY REGIME DATA
# =============================================================================

market_valid.to_csv(
    REGIME_DAILY_PATH,
    index=False,
)


# =============================================================================
# SAVE SUMMARY
# =============================================================================

regime_summary.to_csv(
    REGIME_SUMMARY_PATH,
    index=False,
)


# =============================================================================
# SAVE EVENTS
# =============================================================================

episodes.to_csv(
    REGIME_EVENTS_PATH,
    index=False,
)


# =============================================================================
# SAVE RS01 + REGIME PANEL
# =============================================================================

rs01_regime.to_csv(
    RS01_REGIME_PANEL_PATH,
    index=False,
)


# =============================================================================
# FINAL SANITY CHECKS
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("FINAL SANITY CHECKS")
log.info("=" * 100)

# No future-looking threshold check:
# thresholds at t are generated with shift(1),
# therefore there should be no threshold defined on a date using itself.
log.info(
    "Threshold construction: causal expanding + one-day shift"
)

# Check that regime labels are valid.

valid_regimes = {
    "normal",
    "transition",
    "stress",
    "recovery",
}

observed_regimes = set(
    market_valid["regime"].dropna().unique()
)

unexpected = (
    observed_regimes
    - valid_regimes
)

if unexpected:

    raise RuntimeError(
        f"Unexpected regime labels: {unexpected}"
    )

log.info(
    "Regime labels: OK"
)

# Check RS01 row count preservation.

if len(rs01_regime) != len(rs01):

    raise RuntimeError(
        "RS01 row count changed after regime merge."
    )

log.info(
    "RS01 row count preserved: OK"
)


# Check duplicate ticker/date pairs.

duplicates = rs01_regime.duplicated(
    subset=[
        "ticker",
        "forecast_date",
    ]
).sum()

log.info(
    f"Duplicate ticker + forecast_date rows: "
    f"{duplicates}"
)

if duplicates != 0:

    raise RuntimeError(
        "Duplicate ticker + forecast_date rows detected."
    )


# =============================================================================
# OUTPUTS
# =============================================================================

log.info("")
log.info("=" * 100)
log.info("OUTPUTS SAVED")
log.info("=" * 100)

log.info(
    f"Daily regimes:\n"
    f"{REGIME_DAILY_PATH}"
)

log.info(
    f"Regime summary:\n"
    f"{REGIME_SUMMARY_PATH}"
)

log.info(
    f"Regime events:\n"
    f"{REGIME_EVENTS_PATH}"
)

log.info(
    f"RS01 + regimes:\n"
    f"{RS01_REGIME_PANEL_PATH}"
)

log.info(
    f"Log:\n"
    f"{LOG_PATH}"
)

log.info("")
log.info("=" * 100)
log.info("RS03 REGIME IDENTIFICATION COMPLETE")
log.info("=" * 100)