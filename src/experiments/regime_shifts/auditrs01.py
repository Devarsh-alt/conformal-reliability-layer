from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\reliability_layer")

PANEL_PATH = (
    BASE
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_long_history_var_panel.csv"
)

OUT_PATH = (
    BASE
    / "data"
    / "monitoring"
    / "regime_shifts"
    / "RS01_qrf_audit.csv"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 90)
print("RS01 QRF AUDIT")
print("=" * 90)

print(f"\nLoading:")
print(PANEL_PATH)

df = pd.read_csv(PANEL_PATH)

print("\nColumns:")
for c in df.columns:
    print(f"  {c}")

print(f"\nTotal rows: {len(df):,}")


# ============================================================
# BASIC PANEL STRUCTURE
# ============================================================

print("\n" + "=" * 90)
print("PANEL STRUCTURE")
print("=" * 90)

print(f"Unique tickers: {df['ticker'].nunique()}")

print("\nRows per ticker:")

ticker_counts = (
    df.groupby("ticker")
      .size()
      .sort_index()
)

print(ticker_counts.to_string())

print("\nExpected rows per ticker: 2893")
print("Expected total rows:      69432")

print("\nDifference from expected:")

print(
    (ticker_counts - 2893)
    .to_string()
)


# ============================================================
# DATE COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("DATE COVERAGE")
print("=" * 90)

df["forecast_date"] = pd.to_datetime(df["forecast_date"])
df["realized_date"] = pd.to_datetime(df["realized_date"])

coverage = (
    df.groupby("ticker")
      .agg(
          rows=("forecast_date", "size"),
          unique_forecast_dates=("forecast_date", "nunique"),
          forecast_start=("forecast_date", "min"),
          forecast_end=("forecast_date", "max"),
          realized_start=("realized_date", "min"),
          realized_end=("realized_date", "max"),
      )
      .sort_index()
)

print(coverage.to_string())


# ============================================================
# DUPLICATES
# ============================================================

print("\n" + "=" * 90)
print("DUPLICATE CHECK")
print("=" * 90)

duplicate_rows = df.duplicated(
    subset=["ticker", "forecast_date"]
).sum()

print(
    f"Duplicate ticker + forecast_date rows: "
    f"{duplicate_rows}"
)


# ============================================================
# NULL CHECK
# ============================================================

print("\n" + "=" * 90)
print("NULL CHECK")
print("=" * 90)

important_cols = [
    "ticker",
    "forecast_date",
    "realized_date",
    "actual_return",
    "hs_var",
    "hs_breach",
    "hs_pinball_loss",
    "ewma_var",
    "ewma_breach",
    "ewma_pinball_loss",
    "garch_var",
    "garch_breach",
    "garch_pinball_loss",
    "qrf_var",
    "qrf_breach",
    "qrf_pinball_loss",
]

null_counts = df[important_cols].isna().sum()

print(null_counts.to_string())


# ============================================================
# MODEL COVERAGE
# ============================================================

print("\n" + "=" * 90)
print("MODEL COVERAGE")
print("=" * 90)

models = {
    "HS": "hs_var",
    "EWMA": "ewma_var",
    "GARCH-t": "garch_var",
    "QRF": "qrf_var",
}

for model, col in models.items():

    valid = df[col].notna()

    print(
        f"{model:8s}: "
        f"{valid.sum():,} valid / "
        f"{len(df):,} total "
        f"({valid.mean() * 100:.3f}%)"
    )


# ============================================================
# QRF ROW COVERAGE BY TICKER
# ============================================================

print("\n" + "=" * 90)
print("QRF VALID ROWS PER TICKER")
print("=" * 90)

qrf_coverage = (
    df.groupby("ticker")
      .agg(
          total_rows=("ticker", "size"),
          qrf_valid=("qrf_var", "count"),
          qrf_missing=("qrf_var", lambda x: x.isna().sum()),
      )
)

qrf_coverage["qrf_valid_pct"] = (
    qrf_coverage["qrf_valid"]
    / qrf_coverage["total_rows"]
    * 100
)

print(qrf_coverage.to_string())


# ============================================================
# EXACT QRF MISSING DATES
# ============================================================

print("\n" + "=" * 90)
print("EXACT QRF MISSING-DATE ANALYSIS")
print("=" * 90)

missing_records = []

for ticker in sorted(df["ticker"].unique()):

    ticker_df = df[df["ticker"] == ticker].copy()

    missing = ticker_df[
        ticker_df["qrf_var"].isna()
    ]

    print(
        f"\n{ticker}: "
        f"{len(missing)} missing QRF forecasts"
    )

    if len(missing) > 0:

        dates = (
            missing["forecast_date"]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        print("  Missing dates:")
        print(" ", dates)

        for _, row in missing.iterrows():

            missing_records.append({
                "ticker": ticker,
                "forecast_date": row["forecast_date"],
                "realized_date": row["realized_date"],
            })


# ============================================================
# QRF FIT NUMBER ANALYSIS
# ============================================================

print("\n" + "=" * 90)
print("QRF FIT NUMBER ANALYSIS")
print("=" * 90)

if "qrf_fit_number" in df.columns:

    print(
        df["qrf_fit_number"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )

    print("\nQRF fit number by ticker:")

    fit_counts = (
        df.groupby(["ticker", "qrf_fit_number"])
          .size()
          .unstack(fill_value=0)
    )

    print(fit_counts.to_string())


# ============================================================
# QRF TREE COUNT
# ============================================================

print("\n" + "=" * 90)
print("QRF TREE COUNT")
print("=" * 90)

if "qrf_n_trees" in df.columns:

    print(
        df["qrf_n_trees"]
        .value_counts(dropna=False)
        .sort_index()
        .to_string()
    )


# ============================================================
# QRF FORECAST DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("QRF FORECAST DISTRIBUTION")
print("=" * 90)

qrf_valid = df[df["qrf_var"].notna()].copy()

print(
    qrf_valid["qrf_var"]
    .describe(
        percentiles=[
            0.01,
            0.05,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
        ]
    )
)


# ============================================================
# QRF TREE DISTRIBUTION
# ============================================================

print("\n" + "=" * 90)
print("QRF TREE PREDICTION DISTRIBUTION")
print("=" * 90)

for col in ["qrf_tree_mean", "qrf_tree_std"]:

    if col in df.columns:

        print(f"\n{col}:")

        print(
            qrf_valid[col]
            .describe(
                percentiles=[
                    0.01,
                    0.05,
                    0.25,
                    0.50,
                    0.75,
                    0.95,
                    0.99,
                ]
            )
        )


# ============================================================
# QRF BY TICKER
# ============================================================

print("\n" + "=" * 90)
print("QRF FORECAST STATISTICS BY TICKER")
print("=" * 90)

qrf_ticker_stats = (
    qrf_valid.groupby("ticker")
             .agg(
                 rows=("qrf_var", "count"),
                 mean_qrf_var=("qrf_var", "mean"),
                 median_qrf_var=("qrf_var", "median"),
                 std_qrf_var=("qrf_var", "std"),
                 min_qrf_var=("qrf_var", "min"),
                 max_qrf_var=("qrf_var", "max"),
             )
             .sort_values("mean_qrf_var")
)

print(qrf_ticker_stats.to_string())


# ============================================================
# BREACH RATES
# ============================================================

print("\n" + "=" * 90)
print("BREACH RATES")
print("=" * 90)

breach_columns = {
    "HS": "hs_breach",
    "EWMA": "ewma_breach",
    "GARCH-t": "garch_breach",
    "QRF": "qrf_breach",
}

for model, col in breach_columns.items():

    valid = df[col].notna()

    print(
        f"{model:8s}: "
        f"{df.loc[valid, col].mean():.6f} "
        f"({df.loc[valid, col].mean() * 100:.3f}%) "
        f"over {valid.sum():,} observations"
    )


# ============================================================
# BREACH RATE BY TICKER
# ============================================================

print("\n" + "=" * 90)
print("BREACH RATE BY TICKER")
print("=" * 90)

breach_table = []

for ticker, g in df.groupby("ticker"):

    row = {"ticker": ticker}

    for model, col in breach_columns.items():

        valid = g[col].notna()

        row[f"{model}_rows"] = valid.sum()

        if valid.sum() > 0:
            row[f"{model}_breach_rate"] = (
                g.loc[valid, col].mean()
            )
        else:
            row[f"{model}_breach_rate"] = np.nan

    breach_table.append(row)

breach_df = pd.DataFrame(breach_table)

print(
    breach_df
    .sort_values("QRF_breach_rate", ascending=False)
    .to_string(index=False)
)


# ============================================================
# PINBALL LOSS
# ============================================================

print("\n" + "=" * 90)
print("PINBALL LOSS")
print("=" * 90)

pinball_columns = {
    "HS": "hs_pinball_loss",
    "EWMA": "ewma_pinball_loss",
    "GARCH-t": "garch_pinball_loss",
    "QRF": "qrf_pinball_loss",
}

for model, col in pinball_columns.items():

    valid = df[col].notna()

    print(
        f"{model:8s}: "
        f"mean={df.loc[valid, col].mean():.6f}, "
        f"median={df.loc[valid, col].median():.6f}, "
        f"n={valid.sum():,}"
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "=" * 90)
print("MODEL COMPARISON")
print("=" * 90)

comparison_rows = []

for model, var_col in models.items():

    valid = df[var_col].notna()

    breach_col = breach_columns[model]
    loss_col = pinball_columns[model]

    comparison_rows.append({
        "model": model,
        "rows": valid.sum(),
        "coverage_pct": valid.mean() * 100,
        "breach_rate": df.loc[valid, breach_col].mean(),
        "mean_pinball_loss": df.loc[valid, loss_col].mean(),
        "median_pinball_loss": df.loc[valid, loss_col].median(),
        "mean_abs_var": df.loc[valid, var_col].abs().mean(),
        "mean_var": df.loc[valid, var_col].mean(),
    })

comparison = pd.DataFrame(comparison_rows)

print(
    comparison
    .to_string(index=False)
)


# ============================================================
# SAVE TICKER AUDIT
# ============================================================

audit_df = qrf_coverage.reset_index()

audit_df = audit_df.merge(
    qrf_ticker_stats.reset_index(),
    on="ticker",
    how="left"
)

audit_df = audit_df.merge(
    breach_df,
    on="ticker",
    how="left"
)

audit_df.to_csv(OUT_PATH, index=False)

print("\n" + "=" * 90)
print("AUDIT SAVED")
print("=" * 90)

print(OUT_PATH)

print("\n" + "=" * 90)
print("QRF AUDIT COMPLETE")
print("=" * 90)