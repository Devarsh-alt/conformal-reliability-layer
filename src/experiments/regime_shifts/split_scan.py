from __future__ import annotations
from pathlib import Path
import math
import re
import sys
import zipfile
from io import TextIOWrapper

import numpy as np
import pandas as pd


RAW_ZIP = Path(r"D:\reliability_layer\data\raw.zip")

TICKERS = [
    "AAPL", "AMZN", "BA", "BAC", "CAT", "CVX", "DIS", "GME",
    "GOOGL", "GS", "JNJ", "JPM", "KO", "MCD", "MSFT", "NFLX",
    "NVDA", "PFE", "PG", "TSLA", "UNH", "V", "WMT", "XOM",
]

# User-specified known split dates.
KNOWN_SPLITS = {
    "AAPL": ["2020-08-31"],
    "AMZN": ["2022-06-06"],
    "GOOGL": ["2022-07-18"],
    "TSLA": ["2020-08-31", "2022-08-25"],
    "NVDA": ["2021-07-20", "2024-06-10"],
}

RETURN_THRESHOLD = 0.40  # |log return| above this is flagged for manual review.
SPLIT_RETURN_WARN = 0.40


def normalize_name(name: str) -> str:
    """Normalize filename/header text for loose matching."""
    return re.sub(r"[^A-Za-z0-9]+", "", name).upper()


def ticker_candidates(member_names: list[str], ticker: str) -> list[str]:
    """
    Return CSV members whose filename appears to refer to the ticker.
    We intentionally print all candidates rather than silently choosing one.
    """
    out = []
    t = ticker.upper()
    for member in member_names:
        if not member.lower().endswith(".csv"):
            continue
        stem = Path(member).stem.upper()
        # Token-ish matching: exact stem, or ticker next to separators/brackets.
        if (
            stem == t
            or re.search(rf"(^|[^A-Z0-9]){re.escape(t)}([^A-Z0-9]|$)", stem)
            or normalize_name(stem) == normalize_name(t)
        ):
            out.append(member)
    return out


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_actual = {normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in norm_to_actual:
            return norm_to_actual[key]
    return None


def read_csv_member(zf: zipfile.ZipFile, member: str) -> tuple[pd.DataFrame, str, str | None, str | None]:
    with zf.open(member) as raw:
        # utf-8-sig handles common CSV exports with a BOM.
        text = TextIOWrapper(raw, encoding="utf-8-sig", errors="replace")
        df = pd.read_csv(text)

    date_col = find_column(df, ["Date", "Datetime", "Timestamp"])
    close_col = find_column(df, ["Close", "Adj Close", "Adjusted Close"])
    adj_col = find_column(df, ["Adj Close", "Adjusted Close"])

    if date_col is None:
        raise ValueError(f"{member}: no Date/Datetime/Timestamp column found")
    if close_col is None:
        raise ValueError(f"{member}: no Close/Adj Close column found")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col)
    return df, date_col, close_col, adj_col


def log_returns(df: pd.DataFrame, date_col: str, price_col: str) -> pd.Series:
    px = pd.to_numeric(df[price_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    px = px.where(px > 0)
    return np.log(px).diff().set_axis(df[date_col])


def main() -> int:
    if not RAW_ZIP.exists():
        print(f"ERROR: raw zip not found: {RAW_ZIP}")
        print("Update RAW_ZIP at the top of this script if your raw.zip is elsewhere.")
        return 2

    print("=" * 90)
    print("RAW DATA SPLIT / FILE-MATCHING AUDIT")
    print("=" * 90)
    print(f"ZIP: {RAW_ZIP}")
    print()

    with zipfile.ZipFile(RAW_ZIP, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        csv_members = [m for m in members if m.lower().endswith(".csv")]

        print(f"Total members: {len(members)}")
        print(f"CSV members:   {len(csv_members)}")
        print()
        print("ALL ZIP MEMBERS:")
        for m in members:
            print(f"  {m}")
        print()

        print("=" * 90)
        print("TICKER -> FILE MATCHES")
        print("=" * 90)

        matches: dict[str, list[str]] = {}
        for ticker in TICKERS:
            cand = ticker_candidates(csv_members, ticker)
            matches[ticker] = cand
            status = "OK" if len(cand) == 1 else ("MISSING" if not cand else "MULTIPLE")
            print(f"{ticker:6s} {status:9s} {cand}")

        matched_unique = sum(len(v) >= 1 for v in matches.values())
        unique_one_to_one = sum(len(v) == 1 for v in matches.values())
        print()
        print(f"Tickers matched at least one CSV: {matched_unique}/24")
        print(f"Tickers matched exactly one CSV:   {unique_one_to_one}/24")

        if matched_unique < 24:
            print("FAIL: fewer than 24 tickers were matched. Do not trust downstream results yet.")
            return 3

        if unique_one_to_one < 24:
            print("FAIL: one or more tickers have multiple candidate CSVs.")
            print("Resolve the filename ambiguity before using the audit as a clean PASS.")
            return 4

        print()
        print("=" * 90)
        print("CSV SCHEMA + DATE COVERAGE + LARGE LOG-RETURN SCAN")
        print("=" * 90)

        summaries = []
        failed_members = []

        for ticker in TICKERS:
            member = matches[ticker][0]
            try:
                df, date_col, close_col, adj_col = read_csv_member(zf, member)

                # Scan the actual column selected by the loader.
                lr = log_returns(df, date_col, close_col)
                large = pd.DataFrame({
                    "date": df[date_col].values,
                    "log_return": lr.to_numpy(),
                })
                large = large.loc[large["log_return"].abs() > RETURN_THRESHOLD].copy()

                print(
                    f"{ticker:6s} | {member} | "
                    f"rows={len(df):4d} | "
                    f"range={df[date_col].min().date()} to {df[date_col].max().date()} | "
                    f"price_col={close_col!r} | adj_close={'YES' if adj_col else 'NO'} | "
                    f"|lr|>{RETURN_THRESHOLD:.2f}: {len(large)}"
                )

                if len(large):
                    preview = large.sort_values("date").head(12)
                    for _, r in preview.iterrows():
                        print(f"        large move {r['date'].date()}  log_return={r['log_return']:.6f}")

                summaries.append({
                    "ticker": ticker,
                    "member": member,
                    "rows": len(df),
                    "start": df[date_col].min(),
                    "end": df[date_col].max(),
                    "price_col": close_col,
                    "has_adj_close": bool(adj_col),
                    "large_move_count": len(large),
                })
            except Exception as exc:
                failed_members.append((ticker, member, repr(exc)))
                print(f"{ticker:6s} | ERROR | {exc}")

        if failed_members:
            print()
            print("FAIL: one or more CSV files could not be parsed.")
            for item in failed_members:
                print("  ", item)
            return 5

        print()
        print("=" * 90)
        print("KNOWN SPLIT-DATE CHECK")
        print("=" * 90)
        print("Rule: a split date should not show an obviously artificial ~-1.x or +1.x log return.")
        print()

        split_failures = []

        for ticker, dates in KNOWN_SPLITS.items():
            member = matches[ticker][0]
            df, date_col, close_col, adj_col = read_csv_member(zf, member)
            df = df.sort_values(date_col).reset_index(drop=True)

            px = pd.to_numeric(df[close_col], errors="coerce")
            px = px.where(px > 0)
            lr = np.log(px).diff()
            date_series = df[date_col]

            print(f"{ticker}: file={member} | price={close_col} | adj_close={'YES' if adj_col else 'NO'}")

            for split_date in dates:
                target = pd.Timestamp(split_date)
                idx = np.flatnonzero(date_series.to_numpy() == target.to_datetime64())

                if len(idx) == 0:
                    print(f"  {split_date}: NOT FOUND")
                    split_failures.append((ticker, split_date, "date_not_found"))
                    continue

                i = int(idx[0])
                cur_px = float(px.iloc[i]) if pd.notna(px.iloc[i]) else math.nan
                prev_px = float(px.iloc[i - 1]) if i > 0 and pd.notna(px.iloc[i - 1]) else math.nan
                ret = float(lr.iloc[i]) if pd.notna(lr.iloc[i]) else math.nan

                print(
                    f"  {split_date}: prev_price={prev_px:.6f} "
                    f"price={cur_px:.6f} log_return={ret:.6f}"
                )

                if math.isfinite(ret) and abs(ret) > SPLIT_RETURN_WARN:
                    split_failures.append((ticker, split_date, ret))

        print()
        print("=" * 90)
        print("FINAL AUDIT")
        print("=" * 90)

        all_rows = pd.DataFrame(summaries)
        print(f"24/24 ticker files parsed: {'PASS' if len(all_rows) == 24 else 'FAIL'}")
        print(f"All ticker histories present: {'PASS' if (all_rows['rows'] > 0).all() else 'FAIL'}")

        if split_failures:
            print("Known split-date check: FAIL / REVIEW")
            for item in split_failures:
                print("  ", item)
            print()
            print("ACTION: inspect the affected series before rerunning RS01-RS04.")
            return 6

        print("Known split-date check: PASS")
        print()
        print("IMPORTANT:")
        print("- This script verifies file coverage and flags suspicious returns.")
        print("- It does NOT silently rewrite prices or alter raw.zip.")
        print("- If all 24 files pass, the next step is to confirm which price column RS01 actually uses.")
        print("- Only after that confirmation should you decide whether RS01-RS04 need a rebuild.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
