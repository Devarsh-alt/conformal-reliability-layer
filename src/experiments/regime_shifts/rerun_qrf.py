import shutil
import numpy as np
import pandas as pd

import long_history_var as L


# ============================================================
# BACK UP ORIGINAL PANEL
# ============================================================

backup_path = L.OUTPUT_PANEL.with_name(
    "RS01_panel_OLD_QRF.csv"
)

if not backup_path.exists():
    shutil.copy(
        L.OUTPUT_PANEL,
        backup_path,
    )

    print(
        f"Old panel backed up to:\n{backup_path}",
        flush=True,
    )
else:
    print(
        f"Backup already exists; leaving it untouched:\n{backup_path}",
        flush=True,
    )


# ============================================================
# LOAD EXISTING RS01 PANEL
# ============================================================

panel = pd.read_csv(
    L.OUTPUT_PANEL,
    parse_dates=[
        "forecast_date",
        "realized_date",
    ],
)

print(
    f"\nExisting panel: {len(panel):,} rows",
    flush=True,
)


# ============================================================
# LOAD RAW DATA
# ============================================================

stocks, vix = L.load_raw_data()


# ============================================================
# RECOMPUTE ONLY QRF
# ============================================================

new_rows = []

for ticker in L.TICKERS:

    print(
        f"\nStarting {ticker}...",
        flush=True,
    )

    df = L.build_features(
        stocks[ticker],
        vix,
    )

    usable = (
        df[
            df["log_return"].notna()
            & df["next_return"].notna()
            & df["realized_date"].notna()
        ]
        .sort_values("date")
        .reset_index(drop=True)
    )

    positions = np.arange(
        L.WINDOW - 1,
        len(usable),
    )

    blocks = [
        positions[
            start:
            start + L.REFIT_EVERY
        ]
        for start in range(
            0,
            len(positions),
            L.REFIT_EVERY,
        )
    ]

    for fit_number, block_positions in enumerate(
        blocks,
        start=1,
    ):

        block_rows = L.fit_qrf_block(
            usable,
            block_positions,
            ticker,
            fit_number,
        )

        new_rows.extend(
            block_rows
        )

    print(
        f"{ticker} done",
        flush=True,
    )


# ============================================================
# CREATE NEW QRF DATAFRAME
# ============================================================

qrf = pd.DataFrame(
    new_rows
)

print(
    f"\nGenerated {len(qrf):,} QRF forecasts.",
    flush=True,
)


# ============================================================
# REMOVE OLD QRF COLUMNS
# ============================================================

old_qrf_cols = [
    c
    for c in panel.columns
    if c.startswith("qrf_")
]

panel = panel.drop(
    columns=old_qrf_cols
)


# ============================================================
# MERGE CORRECTED QRF
# ============================================================

panel = panel.merge(
    qrf,
    on=[
        "ticker",
        "forecast_date",
    ],
    how="left",
    validate="one_to_one",
)


# ============================================================
# RECOMPUTE QRF BREACH + PINBALL LOSS
# ============================================================

err = (
    panel["actual_return"]
    - panel["qrf_var"]
)

panel["qrf_breach"] = np.where(
    panel["qrf_var"].notna(),
    (
        panel["actual_return"]
        < panel["qrf_var"]
    ).astype(float),
    np.nan,
)

panel["qrf_pinball_loss"] = np.where(
    panel["qrf_var"].notna(),
    np.where(
        err >= 0,
        L.ALPHA * err,
        (L.ALPHA - 1.0) * err,
    ),
    np.nan,
)


# ============================================================
# QUICK CHECK BEFORE MODIFYING PANEL
# ============================================================

valid = panel[
    panel["qrf_var"].notna()
].copy()

print("\n" + "=" * 70)
print("CORRECTED QRF QUICK CHECK")
print("=" * 70)

print(
    f"Valid QRF rows: "
    f"{len(valid):,}"
)

print(
    f"QRF breach rate: "
    f"{valid['qrf_breach'].mean():.4%}"
)

print(
    f"Mean |QRF VaR|: "
    f"{valid['qrf_var'].abs().mean():.6f}"
)

print(
    f"Mean QRF pinball loss: "
    f"{valid['qrf_pinball_loss'].mean():.6f}"
)


# ============================================================
# COMMON EVALUATION SAMPLE
#
# Removes observations for which corrected QRF is unavailable.
# HS/EWMA/GARCH values themselves are NOT recomputed.
# ============================================================

panel = (
    panel[
        panel["qrf_var"].notna()
    ]
    .sort_values(
        [
            "ticker",
            "forecast_date",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# FINAL SAMPLE CHECK
# ============================================================

print("\nRows per ticker:")

print(
    panel
    .groupby("ticker")
    .size()
)

print(
    f"\nFinal panel rows: "
    f"{len(panel):,}"
)


# ============================================================
# SAVE
# ============================================================

panel.to_csv(
    L.OUTPUT_PANEL,
    index=False,
)

print(
    "\nCorrected RS01 panel saved to:"
)

print(
    L.OUTPUT_PANEL
)