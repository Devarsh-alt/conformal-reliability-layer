import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "..", "data", "raw")

START_DATE = "2011-01-01"
END_DATE = "2026-06-30"

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "JPM", "GS", "BAC", "V",
    "XOM", "CVX",
    "JNJ", "UNH", "PFE",
    "PG", "KO", "MCD", "WMT",
    "CAT", "BA",
    "DIS", "NFLX",
    "TSLA", "GME", "COIN"
]