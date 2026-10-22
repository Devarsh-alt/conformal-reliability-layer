from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "QRF_25ticker_reliability_panel.csv"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "monitoring"
    / "deterioration_target_diagnostics.csv"
)

TARGETS = {
    5: "future_pinball_ratio_5d",
    10: "future_pinball_ratio_10d",
    20: "future_pinball_ratio_20d",
}


def main():

    print("=" * 90)
    print("DETERIORATION TARGET DIAGNOSTICS")
    print("=" * 90)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required = ["ticker"] + list(TARGETS.values())

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print(f"\nLoaded: {INPUT_FILE}")
    print(f"Rows: {len(df):,}")
    print(f"Tickers: {df['ticker'].nunique()}")

    # ========================================================
    # OVERALL DISTRIBUTIONS
    # ========================================================

    print("\n" + "=" * 90)
    print("OVERALL TARGET DISTRIBUTIONS")
    print("=" * 90)

    summary_rows = []

    for h, col in TARGETS.items():

        x = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        q = x.quantile(
            [0.50, 0.75, 0.90, 0.95, 0.99]
        )

        print(f"\nHORIZON = {h} DAYS")
        print("-" * 50)

        print(f"Valid observations : {len(x):,}")
        print(f"Mean               : {x.mean():.4f}")
        print(f"Median             : {x.median():.4f}")
        print(f"Std                : {x.std():.4f}")
        print(f"Min                : {x.min():.4f}")
        print(f"Max                : {x.max():.4f}")
        print(f"P50                : {q.loc[0.50]:.4f}")
        print(f"P75                : {q.loc[0.75]:.4f}")
        print(f"P90                : {q.loc[0.90]:.4f}")
        print(f"P95                : {q.loc[0.95]:.4f}")
        print(f"P99                : {q.loc[0.99]:.4f}")

        summary_rows.append({
            "horizon": h,
            "n": len(x),
            "mean": x.mean(),
            "median": x.median(),
            "std": x.std(),
            "min": x.min(),
            "max": x.max(),
            "p50": q.loc[0.50],
            "p75": q.loc[0.75],
            "p90": q.loc[0.90],
            "p95": q.loc[0.95],
            "p99": q.loc[0.99],
        })

    # ========================================================
    # TARGET CORRELATIONS
    # ========================================================

    print("\n" + "=" * 90)
    print("TARGET CORRELATIONS")
    print("=" * 90)

    corr = df[
        list(TARGETS.values())
    ].corr()

    corr.index = [
        "5d",
        "10d",
        "20d"
    ]

    corr.columns = [
        "5d",
        "10d",
        "20d"
    ]

    print(corr.round(4).to_string())

    # ========================================================
    # HIGH-DETERIORATION RATES
    # ========================================================

    print("\n" + "=" * 90)
    print("HIGH-DETERIORATION RATES")
    print("=" * 90)

    thresholds = [1.0, 1.25, 1.50, 2.0]

    threshold_rows = []

    for h, col in TARGETS.items():

        x = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        print(f"\n{h}-day target")

        for threshold in thresholds:

            valid = x.dropna()

            rate = (
                (valid >= threshold)
                .mean()
            )

            count = (
                valid >= threshold
            ).sum()

            print(
                f"  ratio >= {threshold:>4.2f}: "
                f"{count:>5}/{len(valid):<5} "
                f"({rate:6.2%})"
            )

            threshold_rows.append({
                "horizon": h,
                "threshold": threshold,
                "n_valid": len(valid),
                "n_high": int(count),
                "high_rate": float(rate),
            })

    # ========================================================
    # PER-TICKER DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 90)
    print("PER-TICKER TARGET SUMMARY")
    print("=" * 90)

    ticker_rows = []

    for ticker, g in df.groupby("ticker"):

        row = {
            "ticker": ticker,
            "n_rows": len(g),
        }

        print(f"\n{ticker}")

        for h, col in TARGETS.items():

            x = pd.to_numeric(
                g[col],
                errors="coerce"
            ).dropna()

            if len(x) == 0:
                continue

            p75 = x.quantile(0.75)
            p90 = x.quantile(0.90)

            row[f"h{h}_n"] = len(x)
            row[f"h{h}_mean"] = x.mean()
            row[f"h{h}_median"] = x.median()
            row[f"h{h}_p75"] = p75
            row[f"h{h}_p90"] = p90

            row[f"h{h}_high_1.5_rate"] = (
                x >= 1.5
            ).mean()

            print(
                f"  {h:2d}d: "
                f"median={x.median():.3f}, "
                f"p75={p75:.3f}, "
                f"p90={p90:.3f}, "
                f">=1.5={((x >= 1.5).mean()):.2%}"
            )

        ticker_rows.append(row)

    ticker_summary = pd.DataFrame(
        ticker_rows
    )

    # ========================================================
    # EXTREME OBSERVATIONS
    # ========================================================

    print("\n" + "=" * 90)
    print("EXTREME TARGET OBSERVATIONS")
    print("=" * 90)

    for h, col in TARGETS.items():

        tmp = df[
            [
                "ticker",
                "forecast_date",
                col
            ]
        ].copy()

        tmp[col] = pd.to_numeric(
            tmp[col],
            errors="coerce"
        )

        tmp = (
            tmp
            .dropna()
            .sort_values(col, ascending=False)
            .head(10)
        )

        print(f"\nTop 10 {h}-day deterioration observations:")
        print(
            tmp.to_string(index=False)
        )

    # ========================================================
    # SUMMARY OUTPUT
    # ========================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    ticker_summary_file = (
        ROOT
        / "data"
        / "monitoring"
        / "deterioration_target_by_ticker.csv"
    )

    threshold_file = (
        ROOT
        / "data"
        / "monitoring"
        / "deterioration_target_thresholds.csv"
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    ticker_summary.to_csv(
        ticker_summary_file,
        index=False
    )

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        threshold_file,
        index=False
    )

    print("\n" + "=" * 90)
    print("SAVED")
    print("=" * 90)

    print(OUTPUT_FILE)
    print(ticker_summary_file)
    print(threshold_file)

    print("\nTarget diagnostic complete.")


if __name__ == "__main__":
    main()