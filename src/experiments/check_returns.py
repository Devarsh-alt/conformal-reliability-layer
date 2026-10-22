from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"

files = sorted(DATA_DIR.glob("*.csv"))

print("=" * 80)
print("PROCESSED RETURN DATA CHECK")
print("=" * 80)

print("Number of files:", len(files))
print("Files:", [f.stem for f in files])

for path in files:
    df = pd.read_csv(path)

    print("\n" + "-" * 80)
    print(path.name)
    print("-" * 80)

    print("Shape:", df.shape)

    # Parse date
    df["Date"] = pd.to_datetime(df["Date"])

    # Basic return statistics
    r = df["Log_Return"].dropna()

    print("Date range:", df["Date"].min().date(), "to", df["Date"].max().date())
    print("Return observations:", len(r))
    print("Missing returns:", df["Log_Return"].isna().sum())

    print("Mean:", r.mean())
    print("Std:", r.std())
    print("Min:", r.min())
    print("Max:", r.max())

    print("Quantiles:")
    print(r.quantile([0.01, 0.05, 0.50, 0.95, 0.99]).to_string())

    # Check for duplicate dates
    print("Duplicate dates:", df["Date"].duplicated().sum())

    # Check whether returns are plausible
    print("Infinite returns:", np.isinf(r).sum())