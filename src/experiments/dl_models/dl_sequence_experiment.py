from __future__ import annotations

import warnings
from pathlib import Path
import random

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

# Script is inside:
# D:\reliability_layer\src\experiments\dl_models
#
# parents[3] -> D:\reliability_layer

ROOT = Path(__file__).resolve().parents[3]

DATA_DIR = ROOT / "data" / "monitoring"

PANEL_PATH = (
    DATA_DIR
    / "final_24ticker_common_monitoring_panel.csv"
)

OUTPUT_DIR = DATA_DIR / "deep_learning"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = (
    OUTPUT_DIR / "dl_sequence_summary.csv"
)

PRED_PATH = (
    OUTPUT_DIR / "dl_sequence_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

TRAIN_FRAC = 0.60
VAL_FRAC = 0.20

HORIZONS = [5, 10, 20]

MODELS = [
    "qrf",
    "ewma",
    "garch",
    "hs",
]

TICKERS = [
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


# Sequence configuration
LOOKBACK = 40

# Deep learning architectures
ARCHS = [
    "cnn",
    "gru",
    "tcn",
]

# Three fixed seeds
SEEDS = [
    0,
    1,
    2,
]


# Training configuration
MAX_EPOCHS = 40
PATIENCE = 6
BATCH = 256

LR = 1e-3
WD = 1e-4
DROPOUT = 0.20

HIDDEN = 32


# Logistic benchmark
LOGISTIC_C = [
    0.01,
    0.1,
    1.0,
    10.0,
]


# Bootstrap configuration
BOOT_REPS = 300
BOOT_BLOCK = 20


# ============================================================
# 19 SIMPLE FEATURES
# ============================================================

SIMPLE = [
    "var_abs_current",
    "var_change_1d",
    "var_ratio_to_prior",
    "var_ratio_to_20d_mean",

    "pinball_lag1",
    "pinball_mean_5d_prior",
    "pinball_mean_20d_prior",
    "pinball_mean_60d_prior",
    "pinball_acceleration",

    "breach_lag1",
    "breach_streak_prior",
    "days_since_last_breach",

    "breach_rate_5d_prior",
    "breach_rate_20d_prior",
    "breach_rate_60d_prior",

    "realized_vol_20d",
    "realized_vol_60d",
    "vol_ratio_20_60",

    "past_mean_pinball",
]


# Four cross-model disagreement features
COMPLEX = [
    "qrf_ewma_disagreement",
    "qrf_garch_disagreement",
    "qrf_hs_disagreement",
    "mean_absolute_model_disagreement",
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# AUC
# ============================================================

def safe_auc(y, score):

    y = np.asarray(y)

    score = np.asarray(
        score,
        dtype=float
    )

    mask = (
        np.isfinite(score)
        & ~pd.isna(y)
    )

    y = y[mask]
    score = score[mask]

    if (
        len(y) == 0
        or len(np.unique(y)) != 2
    ):
        return np.nan

    return float(
        roc_auc_score(
            y,
            score
        )
    )


# ============================================================
# CHRONOLOGICAL SPLITS
# ============================================================

def add_splits_and_position(df):

    df = (
        df
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    split = np.empty(
        len(df),
        dtype=object
    )

    pos = np.zeros(
        len(df),
        dtype=int
    )

    from_end = np.zeros(
        len(df),
        dtype=int
    )

    for ticker, idx in df.groupby(
        "ticker",
        sort=False
    ).indices.items():

        n = len(idx)

        train_end = int(
            np.floor(
                TRAIN_FRAC * n
            )
        )

        val_end = int(
            np.floor(
                (TRAIN_FRAC + VAL_FRAC) * n
            )
        )

        labels = np.array(
            ["train"] * train_end
            + ["validation"] * (val_end - train_end)
            + ["test"] * (n - val_end),
            dtype=object
        )

        split[idx] = labels

        pos[idx] = np.arange(n)

        for s in [
            "train",
            "validation",
            "test",
        ]:

            selected = idx[
                labels == s
            ]

            from_end[selected] = (
                np.arange(
                    len(selected)
                )[::-1]
            )

    df["split"] = split
    df["pos"] = pos
    df["from_end"] = from_end

    return df


# ============================================================
# LOGISTIC PIPELINE
# ============================================================

def logistic_pipe(
    num_cols,
    C
):

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),

                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),

                num_cols,
            ),

            (
                "ticker",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            )
                        ),

                        (
                            "encoder",
                            OneHotEncoder(
                                drop="first",
                                handle_unknown="ignore",
                                sparse_output=False,
                            )
                        ),
                    ]
                ),

                ["ticker"],
            ),
        ]
    )

    return Pipeline(
        steps=[
            (
                "preprocess",
                preprocessor
            ),

            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=4000,
                    random_state=SEED,
                )
            ),
        ]
    )


# ============================================================
# 1D CNN
# ============================================================

class CNN1D(nn.Module):

    def __init__(
        self,
        n_features,
        n_tickers,
        hidden=HIDDEN,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.ticker_embedding = nn.Embedding(
            n_tickers,
            4
        )

        self.conv = nn.Sequential(

            nn.Conv1d(
                n_features,
                hidden,
                kernel_size=5,
                padding=2,
            ),

            nn.ReLU(),

            nn.Conv1d(
                hidden,
                hidden,
                kernel_size=3,
                padding=1,
            ),

            nn.ReLU(),
        )

        self.head = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                2 * hidden + 4,
                hidden
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden,
                1
            ),
        )

    def forward(
        self,
        x,
        ticker
    ):

        # x:
        # batch x lookback x features

        z = x.transpose(
            1,
            2
        )

        z = self.conv(z)

        # Average temporal representation
        mean_pool = z.mean(
            dim=-1
        )

        # Maximum temporal representation
        max_pool = z.amax(
            dim=-1
        )

        ticker_embedding = (
            self.ticker_embedding(
                ticker
            )
        )

        combined = torch.cat(
            [
                mean_pool,
                max_pool,
                ticker_embedding,
            ],
            dim=1
        )

        return self.head(
            combined
        ).squeeze(-1)


# ============================================================
# GRU
# ============================================================

class GRUNet(nn.Module):

    def __init__(
        self,
        n_features,
        n_tickers,
        hidden=HIDDEN,
        dropout=DROPOUT,
    ):

        super().__init__()

        self.ticker_embedding = nn.Embedding(
            n_tickers,
            4
        )

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden,
            batch_first=True,
        )

        self.head = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                hidden + 4,
                hidden
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden,
                1
            ),
        )

    def forward(
        self,
        x,
        ticker
    ):

        _, hidden = self.gru(x)

        last_hidden = hidden[-1]

        ticker_embedding = (
            self.ticker_embedding(
                ticker
            )
        )

        combined = torch.cat(
            [
                last_hidden,
                ticker_embedding,
            ],
            dim=1
        )

        return self.head(
            combined
        ).squeeze(-1)


# ============================================================
# TCN BLOCK
# ============================================================

class TCNBlock(nn.Module):

    def __init__(
        self,
        channels_in,
        channels_out,
        dilation,
        dropout,
    ):

        super().__init__()

        self.padding = (
            2 * dilation
        )

        self.conv1 = nn.Conv1d(
            channels_in,
            channels_out,
            kernel_size=3,
            dilation=dilation,
        )

        self.conv2 = nn.Conv1d(
            channels_out,
            channels_out,
            kernel_size=3,
            dilation=dilation,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        if channels_in != channels_out:

            self.skip = nn.Conv1d(
                channels_in,
                channels_out,
                kernel_size=1,
            )

        else:

            self.skip = nn.Identity()

    def forward(self, x):

        # Left-only padding makes this causal.

        y = F.pad(
            x,
            (
                self.padding,
                0
            )
        )

        y = self.conv1(y)

        y = torch.relu(y)

        y = self.dropout(y)

        y = F.pad(
            y,
            (
                self.padding,
                0
            )
        )

        y = self.conv2(y)

        y = torch.relu(y)

        y = self.dropout(y)

        return torch.relu(
            y + self.skip(x)
        )


# ============================================================
# TCN
# ============================================================

class TCN(nn.Module):

    def __init__(
        self,
        n_features,
        n_tickers,
        hidden=HIDDEN,
        dropout=DROPOUT,
        dilations=(1, 2, 4, 8),
    ):

        super().__init__()

        self.ticker_embedding = nn.Embedding(
            n_tickers,
            4
        )

        blocks = []

        channels_in = n_features

        for dilation in dilations:

            blocks.append(
                TCNBlock(
                    channels_in,
                    hidden,
                    dilation,
                    dropout,
                )
            )

            channels_in = hidden

        self.tcn = nn.Sequential(
            *blocks
        )

        self.head = nn.Sequential(

            nn.Dropout(dropout),

            nn.Linear(
                hidden + 4,
                hidden
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden,
                1
            ),
        )

    def forward(
        self,
        x,
        ticker
    ):

        z = x.transpose(
            1,
            2
        )

        z = self.tcn(z)

        # Last causal time point
        z = z[:, :, -1]

        ticker_embedding = (
            self.ticker_embedding(
                ticker
            )
        )

        combined = torch.cat(
            [
                z,
                ticker_embedding,
            ],
            dim=1
        )

        return self.head(
            combined
        ).squeeze(-1)


# ============================================================
# NETWORK DICTIONARY
# ============================================================

NETS = {
    "cnn": CNN1D,
    "gru": GRUNet,
    "tcn": TCN,
}


# ============================================================
# SEQUENCE EXTRACTION
# ============================================================

def gather_windows(
    Xall,
    indices,
    lookback,
):

    offsets = torch.arange(
        -lookback + 1,
        1,
        device=Xall.device
    )

    return Xall[
        indices[:, None]
        + offsets[None, :]
    ]


# ============================================================
# PREDICTION
# ============================================================

@torch.no_grad()
def predict(
    net,
    Xall,
    ticker_codes,
    indices,
    lookback,
):

    net.eval()

    outputs = []

    for start in range(
        0,
        len(indices),
        1024
    ):

        batch_indices = indices[
            start : start + 1024
        ]

        X_batch = gather_windows(
            Xall,
            batch_indices,
            lookback
        )

        ticker_batch = (
            ticker_codes[
                batch_indices
            ]
        )

        logits = net(
            X_batch,
            ticker_batch
        )

        probabilities = (
            torch.sigmoid(
                logits
            )
            .cpu()
            .numpy()
        )

        outputs.append(
            probabilities
        )

    if not outputs:

        return np.empty(
            0,
            dtype=float
        )

    return np.concatenate(
        outputs
    )


# ============================================================
# TRAIN NETWORK
# ============================================================

def fit_net(
    architecture,
    seed,
    Xall,
    ticker_codes,
    y_all,
    train_indices,
    validation_indices,
    n_tickers,
    lookback,
):

    set_seed(seed)

    net = NETS[
        architecture
    ](
        Xall.shape[1],
        n_tickers,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    loss_function = (
        nn.BCEWithLogitsLoss()
    )

    best_auc = -np.inf
    best_state = None
    bad_epochs = 0

    y_validation = y_all[
        validation_indices
        .cpu()
        .numpy()
    ]

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        net.train()

        permutation = train_indices[
            torch.randperm(
                len(train_indices),
                device=DEVICE
            )
        ]

        total_loss = 0.0
        total_count = 0

        for start in range(
            0,
            len(permutation),
            BATCH
        ):

            batch_indices = (
                permutation[
                    start : start + BATCH
                ]
            )

            X_batch = gather_windows(
                Xall,
                batch_indices,
                lookback
            )

            ticker_batch = (
                ticker_codes[
                    batch_indices
                ]
            )

            y_batch = torch.as_tensor(
                y_all[
                    batch_indices
                    .cpu()
                    .numpy()
                ],
                dtype=torch.float32,
                device=DEVICE,
            )

            optimizer.zero_grad()

            logits = net(
                X_batch,
                ticker_batch
            )

            loss = loss_function(
                logits,
                y_batch
            )

            loss.backward()

            nn.utils.clip_grad_norm_(
                net.parameters(),
                1.0
            )

            optimizer.step()

            batch_size = len(
                y_batch
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_count += batch_size

        validation_probability = predict(
            net,
            Xall,
            ticker_codes,
            validation_indices,
            lookback
        )

        validation_auc = safe_auc(
            y_validation,
            validation_probability
        )

        if (
            np.isfinite(validation_auc)
            and validation_auc > best_auc
        ):

            best_auc = validation_auc

            best_state = {
                key: value.detach().clone()
                for key, value
                in net.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

        if bad_epochs >= PATIENCE:

            break

    if best_state is None:

        raise RuntimeError(
            f"No valid best state for "
            f"{architecture}, seed={seed}"
        )

    net.load_state_dict(
        best_state
    )

    return net, best_auc


# ============================================================
# COMMON-DATE BLOCK BOOTSTRAP
# ============================================================

def date_block_bootstrap(
    evaluation_df,
    baseline_col,
    model_cols,
    reps=BOOT_REPS,
    block=BOOT_BLOCK,
    seed=SEED,
):

    """
    Paired moving-block bootstrap over common forecast dates.

    Each bootstrap sample selects consecutive blocks of dates.
    All tickers observed on those dates are kept together.

    This preserves cross-sectional dependence caused by
    common market conditions.
    """

    rng = np.random.default_rng(
        seed
    )

    df = (
        evaluation_df
        .sort_values(
            "forecast_date"
        )
        .reset_index(drop=True)
    )

    dates = (
        df["forecast_date"]
        .drop_duplicates()
        .sort_values()
        .to_numpy()
    )

    date_to_positions = {
        date: np.where(
            df["forecast_date"].to_numpy()
            == date
        )[0]
        for date in dates
    }

    differences = {
        col: []
        for col in model_cols
    }

    n_dates = len(dates)

    blocks_needed = int(
        np.ceil(
            n_dates / block
        )
    )

    for _ in range(reps):

        sampled_dates = []

        for _ in range(
            blocks_needed
        ):

            start = rng.integers(
                0,
                n_dates
            )

            block_dates = [
                dates[
                    (start + j) % n_dates
                ]
                for j in range(block)
            ]

            sampled_dates.extend(
                block_dates
            )

        sampled_dates = sampled_dates[
            :n_dates
        ]

        positions = np.concatenate(
            [
                date_to_positions[d]
                for d in sampled_dates
            ]
        )

        y = df.loc[
            positions,
            "y"
        ].to_numpy()

        if (
            len(np.unique(y))
            < 2
        ):
            continue

        baseline_score = df.loc[
            positions,
            baseline_col
        ].to_numpy()

        baseline_auc = safe_auc(
            y,
            baseline_score
        )

        if not np.isfinite(
            baseline_auc
        ):
            continue

        for col in model_cols:

            model_score = df.loc[
                positions,
                col
            ].to_numpy()

            model_auc = safe_auc(
                y,
                model_score
            )

            if np.isfinite(
                model_auc
            ):

                differences[col].append(
                    model_auc
                    - baseline_auc
                )

    results = {}

    for col in model_cols:

        values = np.asarray(
            differences[col],
            dtype=float
        )

        if len(values) == 0:

            results[col] = (
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            )

            continue

        mean_delta = float(
            values.mean()
        )

        ci_low = float(
            np.percentile(
                values,
                2.5
            )
        )

        ci_high = float(
            np.percentile(
                values,
                97.5
            )
        )

        # Two-sided bootstrap sign p-value
        p = max(
            2
            * min(
                np.mean(values <= 0),
                np.mean(values >= 0)
            ),
            1.0
            / (len(values) + 1)
        )

        results[col] = (
            mean_delta,
            ci_low,
            ci_high,
            float(p),
        )

    return results


# ============================================================
# HOLM CORRECTION
# ============================================================

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan
    )

    valid = np.isfinite(
        p_values
    )

    if not valid.any():

        return adjusted

    valid_indices = np.where(
        valid
    )[0]

    order = valid_indices[
        np.argsort(
            p_values[valid]
        )
    ]

    running_max = 0.0

    total = len(order)

    for rank, idx in enumerate(
        order
    ):

        corrected = (
            total - rank
        ) * p_values[idx]

        running_max = max(
            running_max,
            corrected
        )

        adjusted[idx] = min(
            1.0,
            running_max
        )

    return adjusted


# ============================================================
# PER-TICKER AUC
# ============================================================

def per_ticker_metrics(
    evaluation_df,
    score_col,
):

    rows = []

    for ticker, group in (
        evaluation_df
        .groupby("ticker")
    ):

        auc = safe_auc(
            group["y"],
            group[score_col]
        )

        rows.append(
            {
                "ticker": ticker,
                "auc": auc,
                "n": len(group),
                "positive_rate":
                    group["y"].mean(),
            }
        )

    result = pd.DataFrame(
        rows
    )

    valid = result["auc"].dropna()

    return {
        "macro_auc":
            float(valid.mean())
            if len(valid)
            else np.nan,

        "median_auc":
            float(valid.median())
            if len(valid)
            else np.nan,

        "fraction_gt_0_50":
            float(
                (valid > 0.50).mean()
            )
            if len(valid)
            else np.nan,
    }


# ============================================================
# NON-OVERLAPPING AUC
# ============================================================

def nonoverlapping_auc(
    df,
    horizon,
    score_col,
):

    temp = (
        df
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .copy()
    )

    # One observation every h days within each ticker.
    keep = (
        temp
        .groupby("ticker")
        .cumcount()
        % horizon
        == 0
    )

    temp = temp.loc[
        keep
    ]

    return safe_auc(
        temp["y"],
        temp[score_col]
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("DEEP LEARNING SEQUENCE EXPERIMENT")
    print("=" * 100)

    print(
        f"\nROOT:\n{ROOT}"
    )

    print(
        f"\nPANEL:\n{PANEL_PATH}"
    )

    print(
        f"\nDEVICE:\n{DEVICE}"
    )

    print(
        f"\nLookback: {LOOKBACK} days"
    )

    print(
        f"Architectures: {ARCHS}"
    )

    print(
        f"Seeds: {SEEDS}"
    )

    print(
        f"Horizons: {HORIZONS}"
    )

    print(
        f"Models: {MODELS}"
    )

    # ========================================================
    # LOAD
    # ========================================================

    if not PANEL_PATH.exists():

        raise FileNotFoundError(
            f"\nPanel not found:\n"
            f"{PANEL_PATH}"
        )

    df = pd.read_csv(
        PANEL_PATH
    )

    df["forecast_date"] = pd.to_datetime(
        df["forecast_date"],
        errors="coerce"
    )

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    df = df[
        df["ticker"].isin(
            TICKERS
        )
    ].copy()

    df = (
        df
        .sort_values(
            ["ticker", "forecast_date"]
        )
        .reset_index(drop=True)
    )

    df = add_splits_and_position(
        df
    )

    print(
        f"\nPanel rows: {len(df):,}"
    )

    print(
        f"Tickers: "
        f"{df['ticker'].nunique()}"
    )

    print(
        sorted(
            df["ticker"].unique()
        )
    )

    # ========================================================
    # TICKER CODES
    # ========================================================

    ticker_names = sorted(
        df["ticker"].unique()
    )

    ticker_map = {
        ticker: i
        for i, ticker
        in enumerate(ticker_names)
    }

    ticker_numpy = (
        df["ticker"]
        .map(ticker_map)
        .to_numpy()
    )

    ticker_tensor = torch.as_tensor(
        ticker_numpy,
        dtype=torch.long,
        device=DEVICE
    )

    n_tickers = len(
        ticker_map
    )

    # ========================================================
    # OUTPUT STORAGE
    # ========================================================

    summaries = []

    prediction_frames = []

    # ========================================================
    # MODEL LOOP
    # ========================================================

    for monitored_model in MODELS:

        print("\n")
        print("=" * 100)
        print(
            f"MONITORED MODEL: "
            f"{monitored_model.upper()}"
        )
        print("=" * 100)

        features = [
            f"{monitored_model}_{name}"
            for name in SIMPLE
        ] + COMPLEX

        missing = [
            col
            for col in features
            if col not in df.columns
        ]

        if missing:

            raise RuntimeError(
                f"\nMissing columns for "
                f"{monitored_model}:\n"
                + "\n".join(missing)
            )

        print(
            f"\nFeatures: {len(features)}"
        )

        # ====================================================
        # TRAIN-ONLY IMPUTATION + STANDARDIZATION
        # ====================================================

        train_rows = (
            df["split"] == "train"
        )

        raw_X = df[
            features
        ].to_numpy(
            dtype=float
        )

        train_X = raw_X[
            train_rows.to_numpy()
        ]

        medians = np.nanmedian(
            train_X,
            axis=0
        )

        raw_X = np.where(
            np.isfinite(raw_X),
            raw_X,
            medians
        )

        train_mask = (
            train_rows
            .to_numpy()
        )

        means = (
            raw_X[train_mask]
            .mean(axis=0)
        )

        stds = (
            raw_X[train_mask]
            .std(axis=0)
            + 1e-8
        )

        scaled_X = (
            raw_X - means
        ) / stds

        # Prevent extreme numerical values.
        scaled_X = np.clip(
            scaled_X,
            -5,
            5
        ).astype(
            np.float32
        )

        Xall = torch.as_tensor(
            scaled_X,
            dtype=torch.float32,
            device=DEVICE
        )

        # ====================================================
        # HORIZON LOOP
        # ====================================================

        for horizon in HORIZONS:

            target_column = (
                f"{monitored_model}"
                f"_deterioration_"
                f"{horizon}d"
            )

            if target_column not in df.columns:

                raise RuntimeError(
                    f"Missing target:\n"
                    f"{target_column}"
                )

            y_all = df[
                target_column
            ].to_numpy(
                dtype=float
            )

            valid_target = np.isfinite(
                y_all
            )

            enough_history = (
                df["pos"].to_numpy()
                >= LOOKBACK - 1
            )

            valid = (
                valid_target
                & enough_history
            )

            split_array = (
                df["split"]
                .to_numpy()
            )

            from_end = (
                df["from_end"]
                .to_numpy()
            )

            # =================================================
            # PURGE TRAIN / VALIDATION
            # =================================================
            #
            # Last h observations of train and validation are
            # excluded because their forward target window
            # reaches into the next split.
            # =================================================

            train_mask = (
                valid
                & (split_array == "train")
                & (from_end >= horizon)
            )

            validation_mask = (
                valid
                & (
                    split_array
                    == "validation"
                )
                & (from_end >= horizon)
            )

            test_mask = (
                valid
                & (split_array == "test")
            )

            train_indices = np.where(
                train_mask
            )[0]

            validation_indices = np.where(
                validation_mask
            )[0]

            test_indices = np.where(
                test_mask
            )[0]

            print("\n")
            print("-" * 100)

            print(
                f"{monitored_model.upper()} "
                f"{horizon}D"
            )

            print(
                f"Train: "
                f"{len(train_indices):,}"
            )

            print(
                f"Validation: "
                f"{len(validation_indices):,}"
            )

            print(
                f"Test: "
                f"{len(test_indices):,}"
            )

            print(
                f"Test positive rate: "
                f"{y_all[test_indices].mean():.4%}"
            )

            # =================================================
            # LOGISTIC BASELINE
            # =================================================

            print(
                "\n[1] Logistic baseline"
            )

            best_C = None
            best_validation_auc = -np.inf

            for C in LOGISTIC_C:

                model = logistic_pipe(
                    features,
                    C
                )

                train_data = df.loc[
                    train_indices,
                    features + ["ticker"]
                ]

                validation_data = df.loc[
                    validation_indices,
                    features + ["ticker"]
                ]

                model.fit(
                    train_data,
                    y_all[
                        train_indices
                    ].astype(int)
                )

                validation_probability = (
                    model
                    .predict_proba(
                        validation_data
                    )[:, 1]
                )

                validation_auc = safe_auc(
                    y_all[
                        validation_indices
                    ],
                    validation_probability
                )

                print(
                    f"  C={C:<5} "
                    f"validation AUC="
                    f"{validation_auc:.4f}"
                )

                if (
                    np.isfinite(
                        validation_auc
                    )
                    and validation_auc
                    > best_validation_auc
                ):

                    best_validation_auc = (
                        validation_auc
                    )

                    best_C = C

            # Refit final logistic using train only.
            logistic_model = logistic_pipe(
                features,
                best_C
            )

            logistic_model.fit(
                df.loc[
                    train_indices,
                    features + ["ticker"]
                ],
                y_all[
                    train_indices
                ].astype(int)
            )

            logistic_test_probability = (
                logistic_model
                .predict_proba(
                    df.loc[
                        test_indices,
                        features + ["ticker"]
                    ]
                )[:, 1]
            )

            # =================================================
            # EVALUATION FRAME
            # =================================================

            evaluation = df.loc[
                test_indices,
                [
                    "ticker",
                    "forecast_date",
                ]
            ].copy()

            evaluation["y"] = (
                y_all[
                    test_indices
                ].astype(int)
            )

            evaluation["logistic"] = (
                logistic_test_probability
            )

            # =================================================
            # DEEP NETWORKS
            # =================================================

            validation_auc_by_arch = {}

            print(
                "\n[2] Deep-learning models"
            )

            train_tensor_indices = torch.as_tensor(
                train_indices,
                dtype=torch.long,
                device=DEVICE
            )

            validation_tensor_indices = (
                torch.as_tensor(
                    validation_indices,
                    dtype=torch.long,
                    device=DEVICE
                )
            )

            test_tensor_indices = torch.as_tensor(
                test_indices,
                dtype=torch.long,
                device=DEVICE
            )

            for architecture in ARCHS:

                print(
                    f"\n  Architecture: "
                    f"{architecture.upper()}"
                )

                seed_test_predictions = []
                seed_validation_aucs = []

                for seed_offset in SEEDS:

                    actual_seed = (
                        SEED
                        + seed_offset
                    )

                    print(
                        f"    Seed "
                        f"{actual_seed}..."
                    )

                    net, validation_auc = (
                        fit_net(
                            architecture,
                            actual_seed,
                            Xall,
                            ticker_tensor,
                            y_all,
                            train_tensor_indices,
                            validation_tensor_indices,
                            n_tickers,
                            LOOKBACK,
                        )
                    )

                    test_probability = (
                        predict(
                            net,
                            Xall,
                            ticker_tensor,
                            test_tensor_indices,
                            LOOKBACK,
                        )
                    )

                    seed_test_predictions.append(
                        test_probability
                    )

                    seed_validation_aucs.append(
                        validation_auc
                    )

                    print(
                        f"      validation AUC="
                        f"{validation_auc:.4f}"
                    )

                    del net

                    if DEVICE.type == "cuda":

                        torch.cuda.empty_cache()

                # Average the three seeds.
                architecture_test_probability = (
                    np.mean(
                        seed_test_predictions,
                        axis=0
                    )
                )

                architecture_validation_auc = float(
                    np.mean(
                        seed_validation_aucs
                    )
                )

                evaluation[
                    architecture
                ] = architecture_test_probability

                validation_auc_by_arch[
                    architecture
                ] = architecture_validation_auc

                print(
                    f"    Mean validation AUC="
                    f"{architecture_validation_auc:.4f}"
                )

            # =================================================
            # METRICS
            # =================================================

            model_scores = [
                "logistic"
            ] + ARCHS

            print(
                "\n[3] Test results"
            )

            for scorer in model_scores:

                pooled_auc = safe_auc(
                    evaluation["y"],
                    evaluation[scorer]
                )

                nonoverlap = (
                    evaluation
                    .sort_values(
                        [
                            "ticker",
                            "forecast_date"
                        ]
                    )
                    .copy()
                )

                keep = (
                    nonoverlap
                    .groupby("ticker")
                    .cumcount()
                    % horizon
                    == 0
                )

                nonoverlap = (
                    nonoverlap
                    .loc[keep]
                )

                nonoverlap_auc = safe_auc(
                    nonoverlap["y"],
                    nonoverlap[scorer]
                )

                per_ticker = (
                    per_ticker_metrics(
                        evaluation,
                        scorer
                    )
                )

                if scorer == "logistic":

                    validation_value = (
                        best_validation_auc
                    )

                else:

                    validation_value = (
                        validation_auc_by_arch[
                            scorer
                        ]
                    )

                summary_row = {
                    "model":
                        monitored_model.upper(),

                    "horizon":
                        horizon,

                    "scorer":
                        scorer,

                    "test_auc":
                        pooled_auc,

                    "nonoverlap_auc":
                        nonoverlap_auc,

                    "macro_ticker_auc":
                        per_ticker[
                            "macro_auc"
                        ],

                    "median_ticker_auc":
                        per_ticker[
                            "median_auc"
                        ],

                    "fraction_ticker_auc_gt_0_50":
                        per_ticker[
                            "fraction_gt_0_50"
                        ],

                    "val_auc":
                        validation_value,

                    "n_test":
                        len(evaluation),
                }

                summaries.append(
                    summary_row
                )

            print(
                pd.DataFrame(
                    summaries[-4:]
                )[
                    [
                        "scorer",
                        "test_auc",
                        "nonoverlap_auc",
                        "macro_ticker_auc",
                    ]
                ].to_string(
                    index=False,
                    float_format="%.4f"
                )
            )

            # =================================================
            # BOOTSTRAP
            # =================================================

            print(
                "\n[4] Date-block bootstrap"
            )

            bootstrap = date_block_bootstrap(
                evaluation,
                baseline_col="logistic",
                model_cols=ARCHS,
                reps=BOOT_REPS,
                block=BOOT_BLOCK,
                seed=SEED,
            )

            # Add bootstrap results to the relevant rows.
            for row in summaries:

                if (
                    row["model"]
                    != monitored_model.upper()
                    or row["horizon"]
                    != horizon
                    or row["scorer"]
                    not in ARCHS
                ):

                    continue

                delta, ci_low, ci_high, p = (
                    bootstrap[
                        row["scorer"]
                    ]
                )

                row[
                    "diff_vs_logistic"
                ] = delta

                row[
                    "ci_lo"
                ] = ci_low

                row[
                    "ci_hi"
                ] = ci_high

                row[
                    "p_boot"
                ] = p

            # =================================================
            # SAVE TEST PREDICTIONS
            # =================================================

            prediction_output = (
                evaluation.copy()
            )

            prediction_output[
                "monitored_model"
            ] = monitored_model.upper()

            prediction_output[
                "horizon"
            ] = horizon

            prediction_frames.append(
                prediction_output
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary_df = pd.DataFrame(
        summaries
    )

    # Holm correction across ALL 36
    # architecture x monitored-model x horizon comparisons.
    network_mask = (
        summary_df["scorer"]
        != "logistic"
    )

    summary_df.loc[
        network_mask,
        "p_holm"
    ] = holm_adjust(
        summary_df.loc[
            network_mask,
            "p_boot"
        ].to_numpy()
    )

    summary_df[
        "holm_sig_0p05"
    ] = (
        summary_df[
            "p_holm"
        ]
        < 0.05
    )

    # ========================================================
    # SAVE
    # ========================================================

    summary_df.to_csv(
        SUMMARY_PATH,
        index=False
    )

    predictions_df = pd.concat(
        prediction_frames,
        ignore_index=True
    )

    predictions_df.to_csv(
        PRED_PATH,
        index=False
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        "FINAL DL vs LOGISTIC SUMMARY"
    )
    print("=" * 100)

    display_columns = [
        "model",
        "horizon",
        "scorer",
        "test_auc",
        "nonoverlap_auc",
        "macro_ticker_auc",
        "diff_vs_logistic",
        "ci_lo",
        "ci_hi",
        "p_boot",
        "p_holm",
    ]

    print(
        summary_df[
            [
                c
                for c in display_columns
                if c in summary_df.columns
            ]
        ].to_string(
            index=False,
            float_format="%.4f"
        )
    )

    significant = summary_df[
        network_mask
        & summary_df[
            "holm_sig_0p05"
        ].fillna(False)
    ]

    print(
        "\nHolm-significant DL "
        f"comparisons: "
        f"{len(significant)} "
        f"of "
        f"{int(network_mask.sum())}"
    )

    # Positive significant gains only
    positive_significant = significant[
        significant[
            "diff_vs_logistic"
        ] > 0
    ]

    print(
        "Holm-significant positive "
        f"DL gains: "
        f"{len(positive_significant)}"
    )

    print(
        "\nSaved summary:"
    )

    print(
        SUMMARY_PATH
    )

    print(
        "\nSaved predictions:"
    )

    print(
        PRED_PATH
    )

    print(
        "\nExperiment complete."
    )


if __name__ == "__main__":

    main()