import os
import numpy as np
import pandas as pd


def nonconformity_scores(actual, predicted):
    """
    Absolute residual nonconformity score.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual and predicted must have the same shape."
        )

    return np.abs(actual - predicted)


def conformal_quantile(scores, alpha):
    """
    Finite-sample conformal quantile using the 'higher' quantile rule.
    """
    scores = np.asarray(scores, dtype=float)

    if len(scores) == 0:
        raise ValueError(
            "Cannot compute a conformal quantile from empty scores."
        )

    if not 0 < alpha < 1:
        raise ValueError(
            f"alpha must be between 0 and 1, got {alpha}"
        )

    n = len(scores)

    q_level = min(
        np.ceil((n + 1) * (1 - alpha)) / n,
        1.0,
    )

    return np.quantile(
        scores,
        q_level,
        method="higher",
    )


def run_aci(
    actual,
    predicted,
    prediction_dates=None,
    outcome_dates=None,
    alpha_target=0.10,
    gamma=0.01,
    warmup=30,
):
    """
    Adaptive Conformal Inference (ACI).

    Important temporal convention
    -----------------------------
    For a prediction made on date t:

        prediction_date = t
        outcome_date    = t+1

    The conformal interval for prediction t is constructed using
    historical nonconformity scores available before prediction t.

    The coverage result for that prediction becomes observable only
    on outcome_date.

    This function therefore explicitly returns both timestamps.

    Parameters
    ----------
    actual : array-like
        Realized outcomes.

    predicted : array-like
        Corresponding predictions.

    prediction_dates : array-like, optional
        Dates on which the predictions were made.

    outcome_dates : array-like, optional
        Dates on which the corresponding outcomes became known.

    alpha_target : float
        Target miscoverage rate.

    gamma : float
        ACI adaptation rate.

    warmup : int
        Number of initial observations used to initialize the
        conformal score history.

    Returns
    -------
    dict
        Arrays containing conformal quantities and explicit
        prediction/outcome timestamps.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual and predicted must have the same shape."
        )

    n = len(actual)

    if n == 0:
        raise ValueError(
            "actual and predicted cannot be empty."
        )

    if warmup < 1:
        raise ValueError(
            f"warmup must be >= 1, got {warmup}"
        )

    if warmup >= n:
        raise ValueError(
            f"warmup={warmup} must be smaller than "
            f"number of observations n={n}"
        )

    if not 0 < alpha_target < 1:
        raise ValueError(
            f"alpha_target must be between 0 and 1, "
            f"got {alpha_target}"
        )

    if gamma <= 0:
        raise ValueError(
            f"gamma must be > 0, got {gamma}"
        )

    # Convert dates only when supplied.
    if prediction_dates is not None:
        prediction_dates = pd.to_datetime(prediction_dates)

        if len(prediction_dates) != n:
            raise ValueError(
                "prediction_dates must have the same length "
                "as actual and predicted."
            )

    if outcome_dates is not None:
        outcome_dates = pd.to_datetime(outcome_dates)

        if len(outcome_dates) != n:
            raise ValueError(
                "outcome_dates must have the same length "
                "as actual and predicted."
            )

    scores = nonconformity_scores(
        actual,
        predicted,
    )

    # Initial score history.
    history = list(scores[:warmup])

    alpha_t = alpha_target

    alphas = []
    margins = []
    covered = []
    lowers = []
    uppers = []

    # Explicit timestamps.
    result_prediction_dates = []
    result_outcome_dates = []
    result_coverage_known_dates = []

    for t in range(warmup, n):

        # Keep alpha in a valid range.
        alpha_t_clipped = min(
            max(alpha_t, 0.001),
            0.999,
        )

        # IMPORTANT:
        # q uses only scores strictly before t.
        q = conformal_quantile(
            np.asarray(history),
            alpha_t_clipped,
        )

        lower = predicted[t] - q
        upper = predicted[t] + q

        # Evaluate against the now-observed outcome.
        err_t = int(
            not (
                lower <= actual[t] <= upper
            )
        )

        alphas.append(alpha_t_clipped)
        margins.append(q)
        covered.append(1 - err_t)
        lowers.append(lower)
        uppers.append(upper)

        if prediction_dates is not None:
            result_prediction_dates.append(
                prediction_dates[t]
            )

        if outcome_dates is not None:
            result_outcome_dates.append(
                outcome_dates[t]
            )

            # Coverage can only become known when the outcome arrives.
            result_coverage_known_dates.append(
                outcome_dates[t]
            )

        # ACI adaptation happens AFTER observing the outcome.
        alpha_t = (
            alpha_t
            + gamma * (alpha_target - err_t)
        )

        # The newly observed score becomes available
        # for future predictions.
        history.append(scores[t])

    result = {
        "alpha_t": np.asarray(alphas),
        "margin_t": np.asarray(margins),
        "covered_t": np.asarray(covered),
        "lower_t": np.asarray(lowers),
        "upper_t": np.asarray(uppers),
    }

    if prediction_dates is not None:
        result["prediction_date"] = pd.DatetimeIndex(
            result_prediction_dates
        )

    if outcome_dates is not None:
        result["outcome_date"] = pd.DatetimeIndex(
            result_outcome_dates
        )

        result["coverage_known_date"] = pd.DatetimeIndex(
            result_coverage_known_dates
        )

    return result


def run_aci_all_tickers(
    tickers,
    predictions_dir,
    alpha_target=0.10,
    gamma=0.01,
    warmup=30,
):
    """
    Runs ACI independently per ticker and summarizes realized coverage.
    """

    target_coverage = 1 - alpha_target
    rows = []

    for ticker in tickers:

        file_path = os.path.join(
            predictions_dir,
            f"{ticker}.csv",
        )

        if not os.path.exists(file_path):
            print(
                f"WARNING: no predictions file for "
                f"{ticker}, skipping."
            )
            continue

        df = pd.read_csv(file_path)

        df = df.sort_values(
            "prediction_date"
        ).reset_index(drop=True)

        if len(df) <= warmup:
            print(
                f"WARNING: {ticker} has too few rows "
                f"({len(df)}) for warmup={warmup}, skipping."
            )
            continue

        result = run_aci(
            df["actual"],
            df["predicted"],
            prediction_dates=df["prediction_date"],
            outcome_dates=df["outcome_date"],
            alpha_target=alpha_target,
            gamma=gamma,
            warmup=warmup,
        )

        aci_coverage = result["covered_t"].mean()

        # Static conformal baseline:
        # use first half for calibration and second half for evaluation.
        mid = len(df) // 2

        cal_scores = nonconformity_scores(
            df["actual"].iloc[:mid],
            df["predicted"].iloc[:mid],
        )

        q_static = conformal_quantile(
            cal_scores,
            alpha_target,
        )

        test_scores = nonconformity_scores(
            df["actual"].iloc[mid:],
            df["predicted"].iloc[mid:],
        )

        static_coverage = (
            test_scores <= q_static
        ).mean()

        rows.append(
            {
                "ticker": ticker,
                "n_obs": len(df),

                "aci_coverage": aci_coverage,
                "aci_gap": abs(
                    aci_coverage - target_coverage
                ),

                "static_coverage": static_coverage,
                "static_gap": abs(
                    static_coverage - target_coverage
                ),

                "final_alpha_t": result["alpha_t"][-1],
                "mean_margin": result["margin_t"].mean(),
            }
        )

    return pd.DataFrame(rows)


def run_agaci(
    actual,
    predicted,
    alpha_target=0.10,
    gammas=None,
    warmup=30,
    eta=1.0,
):
    """
    Aggregated ACI using online exponentially weighted aggregation
    of multiple ACI experts.
    """

    if gammas is None:
        gammas = [
            0.001,
            0.002,
            0.004,
            0.008,
            0.0125,
            0.02,
            0.05,
        ]

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.shape != predicted.shape:
        raise ValueError(
            "actual and predicted must have the same shape."
        )

    n = len(actual)
    K = len(gammas)

    if warmup >= n:
        raise ValueError(
            f"warmup={warmup} must be smaller than n={n}"
        )

    scores = nonconformity_scores(
        actual,
        predicted,
    )

    alpha_t = np.full(
        K,
        alpha_target,
        dtype=float,
    )

    histories = [
        list(scores[:warmup])
        for _ in range(K)
    ]

    weights = np.full(
        K,
        1.0 / K,
        dtype=float,
    )

    agg_lower = []
    agg_upper = []
    agg_covered = []
    mean_alpha_t = []

    for t in range(warmup, n):

        expert_lowers = np.zeros(K)
        expert_uppers = np.zeros(K)
        expert_losses = np.zeros(K)

        for k in range(K):

            a_clip = min(
                max(alpha_t[k], 0.001),
                0.999,
            )

            q = conformal_quantile(
                np.asarray(histories[k]),
                a_clip,
            )

            lower = predicted[t] - q
            upper = predicted[t] + q

            expert_lowers[k] = lower
            expert_uppers[k] = upper

            err = int(
                not (
                    lower <= actual[t] <= upper
                )
            )

            width = upper - lower

            if actual[t] < lower:
                loss = (
                    width
                    + (2 / alpha_target)
                    * (lower - actual[t])
                )
            elif actual[t] > upper:
                loss = (
                    width
                    + (2 / alpha_target)
                    * (actual[t] - upper)
                )
            else:
                loss = width

            expert_losses[k] = loss

            # Update expert after observing the outcome.
            alpha_t[k] = (
                alpha_t[k]
                + gammas[k]
                * (alpha_target - err)
            )

            histories[k].append(scores[t])

        # Aggregate current expert intervals.
        lower_agg = np.sum(
            weights * expert_lowers
        )

        upper_agg = np.sum(
            weights * expert_uppers
        )

        covered = int(
            lower_agg <= actual[t] <= upper_agg
        )

        agg_lower.append(lower_agg)
        agg_upper.append(upper_agg)
        agg_covered.append(covered)
        mean_alpha_t.append(
            np.mean(alpha_t)
        )

        # Exponential weights update.
        stable_losses = (
            expert_losses
            - expert_losses.min()
        )

        weights *= np.exp(
            -eta * stable_losses
        )

        weight_sum = weights.sum()

        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.full(
                K,
                1.0 / K,
                dtype=float,
            )
        else:
            weights /= weight_sum

    return {
        "lower_t": np.asarray(agg_lower),
        "upper_t": np.asarray(agg_upper),
        "covered_t": np.asarray(agg_covered),
        "mean_alpha_t": np.asarray(mean_alpha_t),
    }


def run_agaci_all_tickers(
    tickers,
    predictions_dir,
    alpha_target=0.10,
    gammas=None,
    warmup=30,
    eta=1.0,
):
    """
    Runs AgACI independently per ticker and summarizes realized coverage.
    """

    target_coverage = 1 - alpha_target
    rows = []

    for ticker in tickers:

        file_path = os.path.join(
            predictions_dir,
            f"{ticker}.csv",
        )

        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        if "prediction_date" in df.columns:
            df = df.sort_values(
                "prediction_date"
            ).reset_index(drop=True)
        else:
            df = df.sort_values(
                "date"
            ).reset_index(drop=True)

        if len(df) <= warmup:
            continue

        result = run_agaci(
            df["actual"],
            df["predicted"],
            alpha_target=alpha_target,
            gammas=gammas,
            warmup=warmup,
            eta=eta,
        )

        agaci_coverage = result["covered_t"].mean()

        mean_width = (
            result["upper_t"]
            - result["lower_t"]
        ).mean()

        rows.append(
            {
                "ticker": ticker,
                "agaci_coverage": agaci_coverage,
                "agaci_gap": abs(
                    agaci_coverage
                    - target_coverage
                ),
                "agaci_mean_width": mean_width,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":

    from config import TICKERS, BASE_DIR

    predictions_dir = os.path.join(
        BASE_DIR,
        "..",
        "data",
        "predictions",
    )

    pd.set_option(
        "display.width",
        140,
    )

    pd.set_option(
        "display.max_columns",
        None,
    )

    aci_summary = run_aci_all_tickers(
        TICKERS,
        predictions_dir,
        alpha_target=0.10,
        gamma=0.01,
        warmup=30,
    )

    agaci_summary = run_agaci_all_tickers(
        TICKERS,
        predictions_dir,
        alpha_target=0.10,
        warmup=30,
    )

    combined = aci_summary.merge(
        agaci_summary,
        on="ticker",
        how="inner",
    )

    combined["aci_beats_agaci"] = (
        combined["aci_gap"]
        < combined["agaci_gap"]
    )

    cols = [
        "ticker",
        "aci_coverage",
        "aci_gap",
        "agaci_coverage",
        "agaci_gap",
        "static_gap",
        "aci_beats_agaci",
    ]

    print(
        combined[cols].to_string(
            index=False
        )
    )

    print("\n--- Aggregate ---")

    print(
        f"Mean ACI gap:    "
        f"{combined['aci_gap'].mean():.4f}"
    )

    print(
        f"Mean AgACI gap:  "
        f"{combined['agaci_gap'].mean():.4f}"
    )

    print(
        f"Mean static gap: "
        f"{combined['static_gap'].mean():.4f}"
    )

    print(
        f"ACI beats AgACI on "
        f"{combined['aci_beats_agaci'].sum()} / "
        f"{len(combined)} tickers"
    )