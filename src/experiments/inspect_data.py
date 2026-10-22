from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

files = [
    ROOT / "data" / "training_panel.csv",
    ROOT / "data" / "signals" / "all_tickers_signals.csv",
    ROOT / "data" / "signals" / "all_tickers_signals_normalized.csv",
    ROOT / "data" / "reliability" / "reliability_dataset.csv",
]

for path in files:
    print("\n" + "=" * 80)
    print(path)
    print("=" * 80)

    if not path.exists():
        print("NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)
    print("\nColumns:")
    for c in df.columns:
        print(" ", c)

    print("\nFirst 5 rows:")
    print(df.head().to_string())

    print("\nData types:")
    print(df.dtypes)