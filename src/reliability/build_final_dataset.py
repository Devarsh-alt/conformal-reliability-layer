import os
import sys
import pandas as pd


# -------------------------------------------------------------------
# IMPORT PATH
# -------------------------------------------------------------------

SRC_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
    )
)

sys.path.insert(0, SRC_DIR)

from config import BASE_DIR


# -------------------------------------------------------------------
# PATHS
# -------------------------------------------------------------------

DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "data",
    "reliability",
)

CANDIDATE_PATH = os.path.join(
    DATA_DIR,
    "reliability_dataset_candidates.csv",
)

NOVELTY_PATH = os.path.join(
    DATA_DIR,
    "novelty_diagnostics",
    "feature_novelty_panel.csv",
)

OUTPUT_PATH = os.path.join(
    DATA_DIR,
    "final_candidate_dataset.csv",
)


# -------------------------------------------------------------------
# FINAL CANDIDATE SIGNALS
# -------------------------------------------------------------------

FINAL_SIGNALS = {
    "D": "disagreement_pct",
    "O": "feature_novelty_pct",
    "M": "ewma_miscoverage_pct",
    "W": "interval_width_pct",
}


# -------------------------------------------------------------------
# LOAD
# -------------------------------------------------------------------

def load_candidate_dataset():

    if not os.path.exists(CANDIDATE_PATH):
        raise FileNotFoundError(
            f"Candidate dataset not found:\n"
            f"{CANDIDATE_PATH}\n\n"
            "Run candidate_signals.py first."
        )

    return pd.read_csv(
        CANDIDATE_PATH,
        parse_dates=[
            "prediction_date",
            "outcome_date",
        ],
    )


def load_novelty():

    if not os.path.exists(NOVELTY_PATH):
        raise FileNotFoundError(
            f"Novelty panel not found:\n"
            f"{NOVELTY_PATH}\n\n"
            "Run novelty_diagnostics.py first."
        )

    return pd.read_csv(
        NOVELTY_PATH,
        parse_dates=[
            "prediction_date",
        ],
    )


# -------------------------------------------------------------------
# BUILD
# -------------------------------------------------------------------

def build_final_dataset():

    print(
        "Loading candidate reliability dataset..."
    )

    candidates = load_candidate_dataset()

    print(
        f"Candidate rows: {len(candidates)}"
    )

    print(
        "Loading feature novelty panel..."
    )

    novelty = load_novelty()

    print(
        f"Novelty rows: {len(novelty)}"
    )

    novelty = novelty[
        [
            "prediction_date",
            "ticker",
            "feature_novelty",
            "feature_novelty_pct",
        ]
    ].copy()

    # ---------------------------------------------------------------
    # Merge O onto the candidate dataset.
    # ---------------------------------------------------------------

    merged = candidates.merge(
        novelty,
        on=[
            "prediction_date",
            "ticker",
        ],
        how="inner",
        validate="one_to_one",
    )

    merged = (
        merged
        .sort_values(
            [
                "ticker",
                "prediction_date",
            ]
        )
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Validate required signals.
    # ---------------------------------------------------------------

    required = [
        "disagreement_pct",
        "feature_novelty_pct",
        "ewma_miscoverage_pct",
        "interval_width_pct",
        "miscovered",
        "abs_error",
        "prediction_date",
        "outcome_date",
        "ticker",
    ]

    missing = [
        col
        for col in required
        if col not in merged.columns
    ]

    if missing:
        raise ValueError(
            f"Final dataset missing columns: {missing}"
        )

    # ---------------------------------------------------------------
    # Rename O to make the four-signal notation explicit.
    # ---------------------------------------------------------------

    merged = merged.rename(
        columns={
            "feature_novelty_pct":
                "novelty_pct",
        }
    )

    signal_cols = [
        "disagreement_pct",
        "novelty_pct",
        "ewma_miscoverage_pct",
        "interval_width_pct",
    ]

    # ---------------------------------------------------------------
    # Only complete observations can enter the multivariate
    # reliability experiments.
    # ---------------------------------------------------------------

    before = len(merged)

    merged = merged.dropna(
        subset=signal_cols
    ).reset_index(
        drop=True
    )

    print(
        f"\nDropped {before - len(merged)} "
        f"rows with unavailable candidate signals."
    )

    # ---------------------------------------------------------------
    # Range checks.
    # ---------------------------------------------------------------

    for col in signal_cols:

        if not merged[
            col
        ].between(
            0,
            1,
        ).all():

            raise AssertionError(
                f"{col} contains values outside [0,1]."
            )

    # ---------------------------------------------------------------
    # Temporal checks.
    # ---------------------------------------------------------------

    if not merged[
        "prediction_date"
    ].is_monotonic_increasing:

        # This is expected globally because we sort by ticker first,
        # so perform the actual validation per ticker instead.
        pass

    for ticker, group in merged.groupby(
        "ticker"
    ):

        if not group[
            "prediction_date"
        ].is_monotonic_increasing:

            raise AssertionError(
                f"{ticker}: prediction dates "
                "are not chronological."
            )

        if not (
            group[
                "prediction_date"
            ]
            <
            group[
                "outcome_date"
            ]
        ).all():

            raise AssertionError(
                f"{ticker}: invalid prediction/outcome chronology."
            )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    merged.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL CANDIDATE DATASET"
    )

    print(
        "========================================"
    )

    print(
        f"Rows: {len(merged)}"
    )

    print(
        f"Tickers: "
        f"{merged['ticker'].nunique()}"
    )

    print(
        f"Date range: "
        f"{merged['prediction_date'].min().date()} "
        f"-> "
        f"{merged['prediction_date'].max().date()}"
    )

    print(
        "\n--- Four candidate signals ---"
    )

    for name, col in FINAL_SIGNALS.items():

        actual_col = (
            "novelty_pct"
            if name == "O"
            else col
        )

        print(
            f"{name}: {actual_col}"
        )

    print(
        "\n--- Signal statistics ---"
    )

    print(
        merged[
            [
                "disagreement_pct",
                "novelty_pct",
                "ewma_miscoverage_pct",
                "interval_width_pct",
            ]
        ]
        .describe()
        .to_string()
    )

    print(
        "\n--- Target ---"
    )

    print(
        f"Miscoverage rate: "
        f"{merged['miscovered'].mean():.6f}"
    )

    print(
        f"Coverage rate: "
        f"{merged['covered'].mean():.6f}"
    )

    print(
        f"\nSaved to:\n{OUTPUT_PATH}"
    )

    return merged


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------

if __name__ == "__main__":

    build_final_dataset()