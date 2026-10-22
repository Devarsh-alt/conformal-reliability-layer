r"""
RS05 - THRESHOLD-FREE COMPARISON: monitor (S6, S4) vs standard rolling backtest statistics

Why: the matched-alert event study has only 1-3 events per model, so "0/8 vs 8/8" is descriptive. Here every
test date is used. Each standard backtest is turned into a continuous SCORE of "this VaR model is failing",
known at date t, and compared with the monitor on the SAME labels (T1 deterioration, T2 coverage failure):

  bt_count250     rolling 250-day breach count             (traffic-light statistic)
  bt_count60      rolling 60-day breach count
  bt_count20      rolling 20-day breach count
  kupiec250       Kupiec unconditional-coverage LR, 250d   (two-sided: too few OR too many breaches)
  christoffersen250  Christoffersen conditional-coverage LR (UC + independence), 250d

Timing: the breach realised at date d (realized_date == d) is known at forecast_date d. A score at t uses
breaches realised through t only. Outcomes of day t+1 onward are never used.

Reported per model and target: AUC and PR-lift of each score, and the DATE-BLOCK BOOTSTRAP CI of
(S6 - score) and (S4 - score) AUC differences, plus a pooled (mean over models) difference.

Usage:
  python backtest_scores_vs_monitor.py
  python backtest_scores_vs_monitor.py --targets T1 T2 --boot 1000 --block 20
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = Path(r"D:\reliability_layer")
RS_DIR = BASE / "data" / "monitoring" / "regime_shifts"
RS05_DIR = RS_DIR / "RS05"
PRED_PATH = RS05_DIR / "RS05_oos_predictions_H10_main.csv.gz"
QUAR_PATH = RS_DIR / "RS04_quarantined_outcomes.csv"
HEADLINE_START = pd.Timestamp("2020-01-01")
MODELS = ["EWMA", "GARCH-t", "HS", "QRF"]
ALPHA = 0.05
SCORES = ["bt_count250", "bt_count60", "bt_count20", "kupiec250", "christoffersen250"]


def xlogy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 0, x * np.log(np.where(y > 0, y, 1.0)), 0.0)


def kupiec_lr(x, n, p=ALPHA):
    pi = x / n
    ll0 = xlogy(n - x, 1 - p) + xlogy(x, p)
    ll1 = xlogy(n - x, 1 - pi) + xlogy(x, pi)
    return -2 * (ll0 - ll1)


def independence_lr(n00, n01, n10, n11):
    n0, n1 = n00 + n01, n10 + n11
    pi01 = np.where(n0 > 0, n01 / np.maximum(n0, 1), 0.0)
    pi11 = np.where(n1 > 0, n11 / np.maximum(n1, 1), 0.0)
    pi = (n01 + n11) / np.maximum(n0 + n1, 1)
    ll0 = xlogy(n00 + n10, 1 - pi) + xlogy(n01 + n11, pi)
    ll1 = xlogy(n00, 1 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1 - pi11) + xlogy(n11, pi11)
    return -2 * (ll0 - ll1)


def build_scores(quar_path):
    q = pd.read_csv(quar_path, usecols=["ticker", "realized_date", "outcome_next_day_breach", "model_name"])
    q["realized_date"] = pd.to_datetime(q["realized_date"])
    q = q.rename(columns={"model_name": "model", "realized_date": "forecast_date",
                          "outcome_next_day_breach": "b"})
    q["b"] = pd.to_numeric(q["b"], errors="coerce")
    q = q.dropna(subset=["b"]).sort_values(["model", "ticker", "forecast_date"]).reset_index(drop=True)
    out = []
    for (m, t), g in q.groupby(["model", "ticker"], sort=False):
        b = g["b"].to_numpy(float)
        s = pd.Series(b)
        prev = s.shift(1).fillna(0.0)
        pair00 = ((1 - prev) * (1 - s)).rolling(250, min_periods=250).sum().to_numpy()
        pair01 = ((1 - prev) * s).rolling(250, min_periods=250).sum().to_numpy()
        pair10 = (prev * (1 - s)).rolling(250, min_periods=250).sum().to_numpy()
        pair11 = (prev * s).rolling(250, min_periods=250).sum().to_numpy()
        c250 = s.rolling(250, min_periods=250).sum().to_numpy()
        lr_uc = kupiec_lr(c250, 250.0)
        lr_ind = independence_lr(pair00, pair01, pair10, pair11)
        out.append(pd.DataFrame({
            "model": m, "ticker": t, "forecast_date": g["forecast_date"].values,
            "bt_count250": c250,
            "bt_count60": s.rolling(60, min_periods=60).sum().to_numpy(),
            "bt_count20": s.rolling(20, min_periods=20).sum().to_numpy(),
            "kupiec250": lr_uc,
            "christoffersen250": lr_uc + lr_ind,
        }))
    return pd.concat(out, ignore_index=True)


def load_preds(path, specs, targets):
    cols = ["forecast_date", "ticker", "model", "target", "H", "spec", "algo", "y", "p_cal"]
    parts = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=1_000_000, low_memory=False):
        ch = ch[ch["spec"].isin(specs) & ch["target"].isin(targets) & (ch["algo"] == "logit")
                & (pd.to_numeric(ch["H"], errors="coerce") == 10) & ch["model"].isin(MODELS)]
        if len(ch):
            parts.append(ch)
    df = pd.concat(parts, ignore_index=True)
    df["forecast_date"] = pd.to_datetime(df["forecast_date"])
    return df[df["forecast_date"] >= HEADLINE_START]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(PRED_PATH))
    ap.add_argument("--quar", default=str(QUAR_PATH))
    ap.add_argument("--targets", nargs="+", default=["T1", "T2"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--out", default=str(RS05_DIR))
    a = ap.parse_args()
    out = Path(a.out)
    rng = np.random.default_rng(a.seed)

    sc = build_scores(Path(a.quar))
    pr = load_preds(Path(a.pred), ["S4", "S6"], a.targets)
    rows, diff_rows = [], []

    for target in a.targets:
        per_model = {}
        for m in MODELS:
            d = pr[(pr["model"] == m) & (pr["target"] == target)]
            w = d[d["spec"] == "S6"][["forecast_date", "ticker", "y", "p_cal"]].rename(columns={"p_cal": "S6"})
            w4 = d[d["spec"] == "S4"][["forecast_date", "ticker", "p_cal"]].rename(columns={"p_cal": "S4"})
            w = w.merge(w4, on=["forecast_date", "ticker"], validate="one_to_one")
            w = w.merge(sc[sc["model"] == m].drop(columns="model"), on=["forecast_date", "ticker"],
                        how="left", validate="one_to_one")
            w = w.dropna(subset=SCORES + ["S6", "S4", "y"]).reset_index(drop=True)
            per_model[m] = w
            y = w["y"].to_numpy(int)
            for name in ["S6", "S4"] + SCORES:
                rows.append(dict(target=target, model=m, score=name, n_rows=len(w), prevalence=y.mean(),
                                 auc=roc_auc_score(y, w[name]),
                                 pr_lift=average_precision_score(y, w[name]) / y.mean()))
        R = pd.DataFrame([r for r in rows if r["target"] == target])
        print("=" * 100 + f"\nTARGET {target}: AUC / PR-lift of monitor vs standard backtest scores (2020+)\n" + "=" * 100)
        print(R.pivot(index="model", columns="score", values="auc")[["S6", "S4"] + SCORES].round(4).to_string())
        print("\nPR-lift")
        print(R.pivot(index="model", columns="score", values="pr_lift")[["S6", "S4"] + SCORES].round(3).to_string())

        # date-block bootstrap of AUC differences
        draws = {(m, mon, s): [] for m in MODELS for mon in ["S6", "S4"] for s in SCORES}
        info = {}
        for m, w in per_model.items():
            ud, didx = np.unique(w["forecast_date"].values, return_inverse=True)
            info[m] = (len(ud), didx)
        nd_common = {m: info[m][0] for m in MODELS}
        B = a.block
        for _ in range(a.boot):
            for m in MODELS:
                nd, didx = info[m]
                nb = int(np.ceil(nd / B))
                st = rng.integers(0, nd, nb)
                idx = ((st[:, None] + np.arange(B)[None, :]) % nd).reshape(-1)[:nd]
                wt = np.bincount(idx, minlength=nd)[didx].astype(float)
                w = per_model[m]
                y = w["y"].to_numpy(int)
                try:
                    base = {k: roc_auc_score(y, w[k], sample_weight=wt) for k in ["S6", "S4"] + SCORES}
                except ValueError:
                    continue
                for mon in ["S6", "S4"]:
                    for s in SCORES:
                        draws[(m, mon, s)].append(base[mon] - base[s])
        print("\n" + "-" * 100 + f"\nDATE-BLOCK BOOTSTRAP (block={B}, reps={a.boot}): AUC(monitor) - AUC(backtest score)\n" + "-" * 100)
        for mon in ["S6", "S4"]:
            for s in SCORES:
                per = []
                for m in MODELS:
                    x = np.array(draws[(m, mon, s)])
                    if len(x) == 0:
                        continue
                    pt = R[(R["model"] == m) & (R["score"] == mon)]["auc"].iloc[0] - \
                        R[(R["model"] == m) & (R["score"] == s)]["auc"].iloc[0]
                    lo, hi = np.percentile(x, [2.5, 97.5])
                    diff_rows.append(dict(target=target, monitor=mon, baseline=s, model=m, auc_diff=pt,
                                          ci_low=lo, ci_high=hi, monitor_better=bool(lo > 0)))
                    per.append(x)
                if per:
                    L = min(len(p) for p in per)
                    pooled = np.mean([p[:L] for p in per], axis=0)
                    lo, hi = np.percentile(pooled, [2.5, 97.5])
                    pt = np.mean([d["auc_diff"] for d in diff_rows if d["target"] == target and d["monitor"] == mon
                                  and d["baseline"] == s])
                    diff_rows.append(dict(target=target, monitor=mon, baseline=s, model="POOLED_MEAN", auc_diff=pt,
                                          ci_low=lo, ci_high=hi, monitor_better=bool(lo > 0)))
        D = pd.DataFrame([d for d in diff_rows if d["target"] == target])
        for mon in ["S6", "S4"]:
            print(f"\n{mon} minus baseline score (AUC difference)")
            print(D[D["monitor"] == mon].pivot(index="baseline", columns="model", values="auc_diff")
                  .round(4).to_string())
            print("CI above zero?")
            print(D[D["monitor"] == mon].pivot(index="baseline", columns="model", values="monitor_better").to_string())

    pd.DataFrame(rows).to_csv(out / "RS05_backtest_vs_monitor_auc.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(out / "RS05_backtest_vs_monitor_auc_diffs.csv", index=False)
    print(f"\nSaved to {out}")
    print("Read: if S4 already matches S6 against these scores, the gain over standard backtests comes from "
          "market/volatility/persistence information, not model state. If S6 > S4 > backtests, both layers help.")


if __name__ == "__main__":
    main()