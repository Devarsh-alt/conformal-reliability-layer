import numpy as np
import pandas as pd


def nonconformity_scores(actual, predicted):
    return np.abs(np.asarray(actual) - np.asarray(predicted))


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, method='higher')


def run_aci(actual, predicted, alpha_target=0.10, gamma=0.01, warmup=30):
    """
    Adaptive Conformal Inference (Gibbs & Candes).
    Runs as one continuous stream across the whole test period.
    """
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    n = len(actual)

    scores = nonconformity_scores(actual, predicted)

    alpha_t = alpha_target
    history = list(scores[:warmup])

    alphas, margins, covered, lowers, uppers = [], [], [], [], []

    for t in range(warmup, n):
        alpha_t_clipped = min(max(alpha_t, 0.001), 0.999)

        q = conformal_quantile(np.array(history), alpha_t_clipped)
        lower = predicted[t] - q
        upper = predicted[t] + q

        err_t = 0 if (lower <= actual[t] <= upper) else 1

        alphas.append(alpha_t_clipped)
        margins.append(q)
        covered.append(1 - err_t)
        lowers.append(lower)
        uppers.append(upper)

        alpha_t = alpha_t + gamma * (alpha_target - err_t)
        history.append(scores[t])

    return {
        'alpha_t': np.array(alphas),
        'margin_t': np.array(margins),
        'covered_t': np.array(covered),
        'lower_t': np.array(lowers),
        'upper_t': np.array(uppers),
    }


def run_aci_all_tickers(tickers, predictions_dir, alpha_target=0.10, gamma=0.01, warmup=30):
    """
    Runs ACI independently per ticker and summarizes realized coverage.
    Returns a pandas DataFrame with one row per ticker.
    """
    import os

    target_coverage = 1 - alpha_target
    rows = []
    for ticker in tickers:
        file_path = os.path.join(predictions_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            print(f"WARNING: no predictions file for {ticker}, skipping.")
            continue

        df = pd.read_csv(file_path)
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) <= warmup:
            print(f"WARNING: {ticker} has too few rows ({len(df)}) for warmup={warmup}, skipping.")
            continue

        result = run_aci(df['actual'], df['predicted'], alpha_target=alpha_target, gamma=gamma, warmup=warmup)
        aci_coverage = result['covered_t'].mean()

        mid = len(df) // 2
        cal_scores = nonconformity_scores(df['actual'][:mid], df['predicted'][:mid])
        q_static = conformal_quantile(cal_scores, alpha_target)
        test_scores = nonconformity_scores(df['actual'][mid:], df['predicted'][mid:])
        static_coverage = (test_scores <= q_static).mean()

        rows.append({
            'ticker': ticker,
            'n_obs': len(df),
            'aci_coverage': aci_coverage,
            'aci_gap': abs(aci_coverage - target_coverage),
            'static_coverage': static_coverage,
            'static_gap': abs(static_coverage - target_coverage),
            'final_alpha_t': result['alpha_t'][-1],
            'mean_margin': result['margin_t'].mean(),
        })

    return pd.DataFrame(rows)

def run_agaci(actual, predicted, alpha_target=0.10, gammas=None, warmup=30, eta=1.0):
    """
    Aggregated ACI (Zaffran et al.) using online exponentially-weighted
    aggregation (EWA) of multiple independent ACI experts, each with a
    different learning rate (gamma).

    Each expert runs its own independent ACI process (own alpha_t, own
    history). At each step, the experts' interval bounds are combined
    using weights that adapt based on each expert's recent interval
    score (Winkler score) -- a loss that penalizes both interval width
    and coverage misses.

    Returns a dict with the aggregated interval bounds, coverage, and
    the mean alpha across experts at each step.
    """
    if gammas is None:
        gammas = [0.001, 0.002, 0.004, 0.008, 0.0125, 0.02, 0.05]

    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    n = len(actual)
    K = len(gammas)

    scores = nonconformity_scores(actual, predicted)

    alpha_t = np.full(K, alpha_target)
    histories = [list(scores[:warmup]) for _ in range(K)]
    weights = np.full(K, 1.0 / K)

    agg_lower, agg_upper, agg_covered, mean_alpha_t = [], [], [], []

    for t in range(warmup, n):
        expert_lowers = np.zeros(K)
        expert_uppers = np.zeros(K)
        expert_losses = np.zeros(K)

        for k in range(K):
            a_clip = min(max(alpha_t[k], 0.001), 0.999)
            q = conformal_quantile(np.array(histories[k]), a_clip)
            lower = predicted[t] - q
            upper = predicted[t] + q
            expert_lowers[k] = lower
            expert_uppers[k] = upper

            err = 0 if (lower <= actual[t] <= upper) else 1

            # Winkler interval score: width, plus a penalty if the point
            # falls outside the interval. Lower is better.
            width = upper - lower
            if actual[t] < lower:
                loss = width + (2 / alpha_target) * (lower - actual[t])
            elif actual[t] > upper:
                loss = width + (2 / alpha_target) * (actual[t] - upper)
            else:
                loss = width
            expert_losses[k] = loss

            # each expert updates its own alpha independently
            alpha_t[k] = alpha_t[k] + gammas[k] * (alpha_target - err)
            histories[k].append(scores[t])

        # combine experts' bounds using current weights
        lower_agg = np.sum(weights * expert_lowers)
        upper_agg = np.sum(weights * expert_uppers)
        covered = 1 if (lower_agg <= actual[t] <= upper_agg) else 0

        agg_lower.append(lower_agg)
        agg_upper.append(upper_agg)
        agg_covered.append(covered)
        mean_alpha_t.append(np.mean(alpha_t))

        # update weights via EWA (subtract min loss first for numerical stability)
        weights = weights * np.exp(-eta * (expert_losses - expert_losses.min()))
        weights = weights / weights.sum()

    return {
        'lower_t': np.array(agg_lower),
        'upper_t': np.array(agg_upper),
        'covered_t': np.array(agg_covered),
        'mean_alpha_t': np.array(mean_alpha_t),
    }

def run_agaci_all_tickers(tickers, predictions_dir, alpha_target=0.10, gammas=None, warmup=30, eta=1.0):
    """
    Runs AgACI independently per ticker and summarizes realized coverage.
    Returns a pandas DataFrame with one row per ticker.
    """
    import os

    target_coverage = 1 - alpha_target
    rows = []
    for ticker in tickers:
        file_path = os.path.join(predictions_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)
        df = df.sort_values('date').reset_index(drop=True)

        if len(df) <= warmup:
            continue

        result = run_agaci(df['actual'], df['predicted'], alpha_target=alpha_target,
                            gammas=gammas, warmup=warmup, eta=eta)
        agaci_coverage = result['covered_t'].mean()
        mean_width = (result['upper_t'] - result['lower_t']).mean()

        rows.append({
            'ticker': ticker,
            'agaci_coverage': agaci_coverage,
            'agaci_gap': abs(agaci_coverage - target_coverage),
            'agaci_mean_width': mean_width,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    import os
    from config import TICKERS, BASE_DIR

    PREDICTIONS_DIR = os.path.join(BASE_DIR, "..", "data", "predictions")

    pd.set_option('display.width', 140)
    pd.set_option('display.max_columns', None)

    aci_summary = run_aci_all_tickers(TICKERS, PREDICTIONS_DIR, alpha_target=0.10, gamma=0.01, warmup=30)
    agaci_summary = run_agaci_all_tickers(TICKERS, PREDICTIONS_DIR, alpha_target=0.10, warmup=30)

    combined = aci_summary.merge(agaci_summary, on='ticker', how='inner')
    combined['aci_beats_agaci'] = combined['aci_gap'] < combined['agaci_gap']

    cols = ['ticker', 'aci_coverage', 'aci_gap', 'agaci_coverage', 'agaci_gap', 'static_gap', 'aci_beats_agaci']
    print(combined[cols].to_string(index=False))

    print("\n--- Aggregate ---")
    print(f"Mean ACI gap:    {combined['aci_gap'].mean():.4f}")
    print(f"Mean AgACI gap:  {combined['agaci_gap'].mean():.4f}")
    print(f"Mean static gap: {combined['static_gap'].mean():.4f}")
    print(f"ACI beats AgACI on {combined['aci_beats_agaci'].sum()} / {len(combined)} tickers")