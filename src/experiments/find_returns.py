from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# Search every CSV in data/ for columns that might contain returns/prices
csv_files = list((ROOT / "data").rglob("*.csv"))

print(f"Found {len(csv_files)} CSV files\n")

for path in csv_files:
    try:
        df = pd.read_csv(path, nrows=5)
    except Exception as e:
        print(f"\nFAILED: {path}")
        print(e)
        continue

    cols = [str(c).lower() for c in df.columns]

    interesting = [
        c for c in cols
        if any(x in c for x in [
            "return",
            "ret",
            "price",
            "close",
            "actual",
            "ticker",
            "date"
        ])
    ]

    if interesting:
        print("=" * 80)
        print(path.relative_to(ROOT))
        print("Columns:")
        print(list(df.columns))
        print("Potentially relevant:")
        print(interesting)