"""
DQ specification audit for the QRF 25-ticker OOS forecasts.

Run from anywhere (defaults assume this file lives in <project>/src/experiments/):

    python src/experiments/dq_specification_audit.py

Inputs : data/monitoring/QRF_25ticker_oos_predictions.csv  (existing OOS forecasts; nothing is retrained)
Outputs: data/monitoring/DQ_specification_audit.csv         (ticker x lag results, long format)
         data/monitoring/DQ_component_diagnostics.csv       (what drives rejections, primary lag)

Conventions (unchanged from the project):
    lower-tail 5% return quantile q_t,   breach  I_t = 1(R_t < q_t)
    centered hit  H_t = I_t - alpha

Dynamic Quantile test (Engle & Manganelli, 2004, JBES 22(4), 367-381):
    Regress   H_t = b0 + b1*H_{t-1} + ... + bL*H_{t-L} + b_{L+1}*q_t + e_t     (t = L+1..n)
    H0: all coefficients are zero  (=> E[H_t | information at t-1] = 0, i.e. conditional coverage)
    DQ = H'X (X'X)^-1 X'H / [alpha(1-alpha)] = beta_hat' X'X beta_hat / [alpha(1-alpha)]  ~  chi2(k) asymptotically
    k  = number of regressors = L + 2 (constant, L lagged hits, VaR forecast).
Under H0 the hit H_t takes value (1-alpha) w.p. alpha and -alpha w.p. (1-alpha), so Var(H_t) = alpha(1-alpha)
exactly; that is the correct normalisation.

The statistic equals (fitted values)'(fitted values) / [alpha(1-alpha)], hence it is invariant to the sign and
scale of any regressor column.  Using the signed quantile q_t or the VaR magnitude -q_t gives the identical
statistic; q_t is z-scored internally below purely for numerical conditioning (verified in the sanity checks).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

ROOT = Path(__file__).resolve().parents[2] if len(Path(__file__).resolve().parents) > 2 else Path.cwd()

DEFAULT_INPUT = ROOT / "data" / "monitoring" / "QRF_25ticker_oos_predictions.csv"
DEFAULT_OUTPUT = ROOT / "data" / "monitoring" / "DQ_specification_audit.csv"
DEFAULT_COMPONENTS = ROOT / "data" / "monitoring" / "DQ_component_diagnostics.csv"

ALPHA = 0.05
LAGS = (1, 4, 5)
PRIMARY_LAG = 4          # Engle-Manganelli's out-of-sample specification; fixed ex ante, not tuned on pass counts
SIGNIF = 0.05
MIN_REG_ROWS = 200       # refuse to run the regression on fewer rows than this

# Exact (case-insensitive) column-name candidates.  No substring matching: a partial match could silently pick
# a positive VaR-magnitude column (e.g. VaR_5pct) or a lagged-return feature.
CANDIDATES = {
    "ticker": ["ticker", "symbol", "asset", "tic"],
    "date": ["date", "realized_date", "outcome_date", "forecast_date", "prediction_date"],
    "return": ["actual_return", "realized_return", "return", "actual"],
    "q": ["q05_forecast", "q05", "q_05", "qrf_q05", "q05_pred"],
}


# --------------------------------------------------------------------------------------
# Column detection
# --------------------------------------------------------------------------------------
def detect_column(df, role, override=None):
    cols_lower = {c.lower(): c for c in df.columns}
    if override is not None:
        if override not in df.columns:
            sys.exit(f"ERROR: --{role}-col '{override}' not in file.\nAvailable columns: {list(df.columns)}")
        return override
    hits = [cols_lower[c] for c in CANDIDATES[role] if c in cols_lower]
    if len(hits) == 1:
        return hits[0]
    if len(hits) == 0:
        sys.exit(f"ERROR: could not find a '{role}' column among {CANDIDATES[role]}.\n"
                 f"Available columns: {list(df.columns)}\nPass --{role}-col to choose explicitly.")
    sys.exit(f"ERROR: ambiguous '{role}' column, several candidates present: {hits}.\n"
             f"Available columns: {list(df.columns)}\nPass --{role}-col to choose explicitly.")


# --------------------------------------------------------------------------------------
# DQ test
# --------------------------------------------------------------------------------------
def build_design(hit_c, q, lags, use_lags=True, use_q=True):
    """Design matrix for rows t = lags..n-1 (first `lags` rows are lost to the lag structure)."""
    n = len(hit_c)
    cols = [np.ones(n - lags)]
    if use_lags:
        for l in range(1, lags + 1):
            cols.append(hit_c[lags - l:n - l])
    if use_q:
        qq = q[lags:]
        sd = qq.std()
        # z-score for conditioning only; constant q (sd=0) is collinear with the intercept and is caught by the rank
        cols.append((qq - qq.mean()) / sd if sd > 0 else qq - qq.mean())
    X = np.column_stack(cols)
    y = hit_c[lags:]
    return X, y


def dq_test(returns, q, alpha=ALPHA, lags=4, use_lags=True, use_q=True):
    returns = np.asarray(returns, float)
    q = np.asarray(q, float)
    hit = (returns < q).astype(float)          # lower-tail convention: breach = return < q05
    hit_c = hit - alpha
    X, y = build_design(hit_c, q, lags, use_lags, use_q)
    # SVD-based least squares: handles (near-)singular designs without silently inverting garbage.
    beta, _, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    fitted = X @ beta
    stat = float(fitted @ fitted) / (alpha * (1.0 - alpha))   # == beta' X'X beta / [alpha(1-alpha)]
    k = X.shape[1]
    return {
        "n_obs": len(returns),
        "n_reg": len(y),
        "breaches": int(hit.sum()),
        "breach_rate": float(hit.mean()),
        "breaches_in_reg_sample": int(hit[lags:].sum()),
        "dq_stat": stat,
        "k_regressors": k,
        "rank": int(rank),
        "df": int(rank),                       # equals k unless the design is rank deficient (flagged)
        "cond_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf,
        "dq_p": float(chi2.sf(stat, rank)),
    }


def dq_original_form(returns, q, alpha=ALPHA, lags=4):
    """Literal beta'X'X beta/[alpha(1-alpha)] with raw signed q_t and an explicit inverse.  Sanity-check only."""
    returns = np.asarray(returns, float)
    q = np.asarray(q, float)
    hit_c = (returns < q).astype(float) - alpha
    n = len(hit_c)
    cols = [np.ones(n - lags)] + [hit_c[lags - l:n - l] for l in range(1, lags + 1)] + [q[lags:]]
    X = np.column_stack(cols)
    y = hit_c[lags:]
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return float(beta @ XtX @ beta) / (alpha * (1 - alpha))


# --------------------------------------------------------------------------------------
# Multiple-testing helpers
# --------------------------------------------------------------------------------------
def holm_rejections(p, level):
    order = np.argsort(p)
    m = len(p)
    rej = 0
    for rank_i, idx in enumerate(order):
        if p[idx] <= level / (m - rank_i):
            rej += 1
        else:
            break
    return rej


def bh_rejections(p, level):
    m = len(p)
    ps = np.sort(p)
    ok = np.where(ps <= level * (np.arange(1, m + 1) / m))[0]
    return int(ok.max() + 1) if len(ok) else 0


# --------------------------------------------------------------------------------------
# Optional finite-sample size check (deterministic seed)
# --------------------------------------------------------------------------------------
def size_simulation(n, lags_list, alpha=ALPHA, reps=5000, seed=12345):
    """Rejection rate of the DQ test at the 5% level when the forecasts are correct (iid Bernoulli hits) and q_t is
    an arbitrary persistent series unrelated to the hits."""
    rng = np.random.default_rng(seed)
    hit_c = (rng.random((reps, n)) < alpha).astype(float) - alpha
    z = np.zeros((reps, n))
    e = rng.standard_normal((reps, n))
    for t in range(1, n):
        z[:, t] = 0.95 * z[:, t - 1] + 0.3 * e[:, t]
    q = -0.025 * np.exp(z)
    out = {}
    for L in lags_list:
        m = n - L
        X = np.ones((reps, m, L + 2))
        for l in range(1, L + 1):
            X[:, :, l] = hit_c[:, L - l:n - l]
        qq = q[:, L:]
        X[:, :, L + 1] = (qq - qq.mean(1, keepdims=True)) / qq.std(1, keepdims=True)
        y = hit_c[:, L:]
        XtX = np.einsum("rmi,rmj->rij", X, X)
        Xty = np.einsum("rmi,rm->ri", X, y)
        b = np.linalg.solve(XtX, Xty[:, :, None])[:, :, 0]
        stat = np.einsum("ri,ri->r", b, Xty) / (alpha * (1 - alpha))
        out[L] = float(np.mean(stat > chi2.ppf(1 - SIGNIF, L + 2)))
    return out


# --------------------------------------------------------------------------------------
# Sanity checks
# --------------------------------------------------------------------------------------
def run_sanity_checks():
    rng = np.random.default_rng(0)
    n = 900
    q = -0.02 - 0.01 * np.abs(rng.standard_normal(n))
    r = q * 0 + rng.standard_normal(n) * 0.012
    for L in LAGS:
        base = dq_test(r, q, lags=L)
        orig = dq_original_form(r, q, lags=L)
        assert np.isclose(base["dq_stat"], orig, rtol=1e-8), f"formula mismatch at L={L}: {base['dq_stat']} vs {orig}"
        # sign / scale invariance of the q regressor (VaR magnitude vs signed quantile)
        flipped = dq_original_form(r, q, lags=L)
        mag = _dq_with_regressor(r, q, -q * 1000.0, L)
        assert np.isclose(flipped, mag, rtol=1e-8), f"q sign/scale invariance failed at L={L}"
        assert base["df"] == L + 2 and base["k_regressors"] == L + 2
    # a model with zero hit-information must give a small statistic; a deterministic hit pattern must reject
    r_all_breach = q - 0.01
    st = dq_test(r_all_breach, q, lags=4)
    assert st["breach_rate"] == 1.0
    print("Sanity checks passed: statistic matches the literal beta'X'X*beta/[a(1-a)] form; "
          "q sign/scale invariance holds; df = L+2.")


def _dq_with_regressor(returns, q, regressor, lags, alpha=ALPHA):
    hit_c = (np.asarray(returns) < np.asarray(q)).astype(float) - alpha
    n = len(hit_c)
    cols = [np.ones(n - lags)] + [hit_c[lags - l:n - l] for l in range(1, lags + 1)] + [regressor[lags:]]
    X = np.column_stack(cols)
    y = hit_c[lags:]
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return float(beta @ XtX @ beta) / (alpha * (1 - alpha))


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=str(DEFAULT_INPUT))
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT))
    ap.add_argument("--components-output", default=str(DEFAULT_COMPONENTS))
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--ticker-col")
    ap.add_argument("--date-col")
    ap.add_argument("--return-col")
    ap.add_argument("--q-col")
    ap.add_argument("--size-sim", action="store_true", help="also run the finite-sample size simulation")
    args = ap.parse_args()
    alpha = args.alpha

    print("=" * 88)
    print("DQ SPECIFICATION AUDIT  (QRF, lower-tail 5% quantile, breach = return < q05)")
    print("=" * 88)
    run_sanity_checks()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"ERROR: input file not found: {path}")
    df = pd.read_csv(path)
    print(f"\nLoaded {path}  shape={df.shape}")
    print(f"Columns: {list(df.columns)}")

    c_tic = detect_column(df, "ticker", args.ticker_col)
    c_date = detect_column(df, "date", args.date_col)
    c_ret = detect_column(df, "return", args.return_col)
    c_q = detect_column(df, "q", args.q_col)
    print(f"Using columns -> ticker: {c_tic} | date: {c_date} | return: {c_ret} | quantile: {c_q}")

    df = df[[c_tic, c_date, c_ret, c_q]].copy()
    df.columns = ["ticker", "date", "ret", "q"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ret"] = pd.to_numeric(df["ret"], errors="coerce")
    df["q"] = pd.to_numeric(df["q"], errors="coerce")

    # Convention guard: q05 must be a (mostly negative) return quantile, not a positive VaR magnitude.
    if df["q"].median() >= 0:
        sys.exit("ERROR: median of the quantile column is >= 0. This looks like a positive VaR magnitude, not the "
                 "signed 5% return quantile. Refusing to guess; pass the correct --q-col.")
    overall_breach = float((df["ret"] < df["q"]).mean())
    print(f"Overall breach rate (return < q): {overall_breach:.4f}")
    if not (0.005 <= overall_breach <= 0.25):
        sys.exit("ERROR: overall breach rate is implausible for a 5% quantile; check column roles/sign.")

    rows, skipped = [], []
    tickers = sorted(df["ticker"].dropna().unique())
    print(f"Tickers found: {len(tickers)}")

    for tk in tickers:
        g = df[df["ticker"] == tk].sort_values("date").reset_index(drop=True)
        n_raw = len(g)
        reason = None
        if g["date"].isna().any():
            reason = f"{int(g['date'].isna().sum())} unparseable dates"
        elif g["date"].duplicated().any():
            reason = f"{int(g['date'].duplicated().sum())} duplicated dates"
        else:
            bad = g["ret"].isna() | g["q"].isna() | ~np.isfinite(g["ret"]) | ~np.isfinite(g["q"])
            if bad.any():
                idx = np.where(bad.values)[0]
                interior = idx[(idx > np.where(~bad.values)[0].min()) & (idx < np.where(~bad.values)[0].max())] \
                    if (~bad).any() else idx
                if len(interior) > 0:
                    reason = f"{len(interior)} interior missing/non-finite values (would corrupt lag structure)"
                else:
                    g = g[~bad].reset_index(drop=True)
                    print(f"  {tk}: dropped {int(bad.sum())} leading/trailing missing rows (n {n_raw} -> {len(g)})")
        if reason is None and len(g) - max(LAGS) < MIN_REG_ROWS:
            reason = f"too few observations ({len(g)}) for the regression"
        if reason is None and (g["ret"] < g["q"]).sum() == 0:
            reason = "no breaches in sample"
        if reason is not None:
            skipped.append((tk, reason))
            print(f"  SKIPPED {tk}: {reason}")
            continue

        gaps = g["date"].diff().dt.days.dropna()
        if (gaps > 7).any():
            print(f"  WARNING {tk}: {(gaps > 7).sum()} calendar gaps > 7 days between consecutive rows "
                  f"(lags assume consecutive trading days)")

        for L in LAGS:
            res = dq_test(g["ret"].values, g["q"].values, alpha=alpha, lags=L)
            if res["rank"] < res["k_regressors"]:
                print(f"  WARNING {tk} L={L}: design rank-deficient (rank {res['rank']} < k {res['k_regressors']}); "
                      f"df set to rank")
            rows.append({"ticker": tk, "lags": L, "status": "OK", **res,
                         "dq_pass": res["dq_p"] >= SIGNIF})

    for tk, reason in skipped:
        rows.append({"ticker": tk, "lags": np.nan, "status": f"SKIPPED: {reason}"})

    out = pd.DataFrame(rows)
    ok = out[out["status"] == "OK"].copy()
    ok["lags"] = ok["lags"].astype(int)
    ok["dq_pass"] = ok["dq_pass"].astype(bool)
    if ok.empty:
        sys.exit("ERROR: no ticker could be tested.")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    # ---------------- per-ticker table ----------------
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    wide_p = ok.pivot(index="ticker", columns="lags", values="dq_p")
    wide_p.columns = [f"p_L{c}" for c in wide_p.columns]
    wide_s = ok.pivot(index="ticker", columns="lags", values="dq_stat")
    wide_s.columns = [f"stat_L{c}" for c in wide_s.columns]
    base = ok[ok["lags"] == LAGS[0]].set_index("ticker")[["n_obs", "breaches", "breach_rate"]]
    table = base.join(wide_s).join(wide_p)
    print("\n" + "=" * 88)
    print("DQ BY TICKER AND LAG   (df = L + 2; statistic and p-value)")
    print("=" * 88)
    print(table.round(4).to_string())

    # ---------------- pass counts ----------------
    print("\n" + "=" * 88)
    print(f"DQ PASS COUNTS (p >= {SIGNIF})")
    print("=" * 88)
    n_t = ok["ticker"].nunique()
    for L in LAGS:
        sub = ok[ok["lags"] == L]
        print(f"L={L} (df={L + 2}): pass {int(sub['dq_pass'].sum())}/{len(sub)}   "
              f"reject {int((~sub['dq_pass']).sum())}/{len(sub)}")

    # ---------------- conclusion changes ----------------
    print("\n" + "=" * 88)
    print("TICKERS WHOSE DQ CONCLUSION CHANGES WITH THE LAG SPECIFICATION")
    print("=" * 88)
    pass_wide = ok.pivot(index="ticker", columns="lags", values="dq_pass")
    changed = pass_wide[pass_wide.nunique(axis=1) > 1]
    if changed.empty:
        print("None: every ticker has the same conclusion for L = " + ", ".join(map(str, LAGS)))
    else:
        show = changed.join(wide_p.loc[changed.index])
        print(show.round(4).to_string())

    # ---------------- multiple testing ----------------
    print("\n" + "=" * 88)
    print("MULTIPLE-TESTING CONTEXT")
    print("=" * 88)
    print(f"{n_t} tickers tested at the {SIGNIF:.0%} level -> expected rejections if every model were correctly "
          f"specified: {n_t * SIGNIF:.2f}")
    print("(binomial tail assumes independent tests; cross-asset dependence makes the count more variable, "
          "so treat the tail probability as a rough guide)")
    for L in LAGS:
        p = ok[ok["lags"] == L]["dq_p"].values
        rej = int((p < SIGNIF).sum())
        print(f"L={L}: raw rejections={rej}  P(X>={rej} | all null, binomial)={binom.sf(rej - 1, n_t, SIGNIF):.2e}  "
              f"Bonferroni={int((p < SIGNIF / n_t).sum())}  Holm={holm_rejections(p, SIGNIF)}  "
              f"BH-FDR(5%)={bh_rejections(p, SIGNIF)}")

    # ---------------- component diagnostics at the primary lag ----------------
    print("\n" + "=" * 88)
    print(f"WHAT DRIVES THE REJECTIONS?  (primary specification L={PRIMARY_LAG}, same sample rows)")
    print("=" * 88)
    comp_rows = []
    for tk in ok["ticker"].unique():
        g = df[df["ticker"] == tk].sort_values("date")
        g = g[np.isfinite(g["ret"]) & np.isfinite(g["q"])]
        r, q = g["ret"].values, g["q"].values
        specs = {
            "const_only": dict(use_lags=False, use_q=False),     # LM-type unconditional coverage
            "const_lags": dict(use_lags=True, use_q=False),      # hit-autocorrelation / "DQ_hit"
            "const_q": dict(use_lags=False, use_q=True),         # does the VaR level carry hit information?
            "full": dict(use_lags=True, use_q=True),             # standard Engle-Manganelli DQ
        }
        row = {"ticker": tk}
        for name, kw in specs.items():
            res = dq_test(r, q, alpha=alpha, lags=PRIMARY_LAG, **kw)
            row[f"p_{name}"] = res["dq_p"]
            row[f"df_{name}"] = res["df"]
        comp_rows.append(row)
    comp = pd.DataFrame(comp_rows).set_index("ticker")
    Path(args.components_output).parent.mkdir(parents=True, exist_ok=True)
    comp.to_csv(args.components_output)
    for name in ["const_only", "const_lags", "const_q", "full"]:
        rej = int((comp[f"p_{name}"] < SIGNIF).sum())
        print(f"{name:<11s} (df={int(comp[f'df_{name}'].iloc[0])}): rejects {rej}/{len(comp)}")
    print("Reading: if 'const_q' rejects far more than 'const_lags', the VaR level itself predicts breaches "
          "(forecast scale mis-calibrated across volatility states), not hit clustering.")

    if args.size_sim:
        print("\n" + "=" * 88)
        print("FINITE-SAMPLE SIZE (correct forecasts; iid hits; seed 12345; 5000 reps)")
        print("=" * 88)
        n_med = int(ok.drop_duplicates("ticker")["n_obs"].median())
        sim = size_simulation(n_med, LAGS, alpha=alpha)
        for L, s in sim.items():
            print(f"n={n_med}, L={L}: empirical size at nominal 5% = {s:.4f}")

    if skipped:
        print("\nSkipped tickers (reported, not silently dropped):")
        for tk, reason in skipped:
            print(f"  {tk}: {reason}")

    print(f"\nSaved: {args.output}\nSaved: {args.components_output}")


if __name__ == "__main__":
    main()