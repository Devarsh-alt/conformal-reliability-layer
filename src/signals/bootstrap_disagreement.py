import numpy as np
import pandas as pd
import lightgbm as lgb


def bootstrap_ensemble_disagreement(train_df, test_df, feature_cols, target_col='target',
                                     n_bootstrap=10, random_state=42):
    """
    Trains n_bootstrap LightGBM models on bootstrap resamples of train_df,
    predicts each on test_df, and returns the disagreement (std across
    the ensemble) at each test point, along with the mean ensemble prediction.
    """
    n = len(train_df)
    rng = np.random.RandomState(random_state)

    all_preds = np.zeros((n_bootstrap, len(test_df)))

    for b in range(n_bootstrap):
        boot_idx = rng.choice(n, size=n, replace=True)
        boot_train = train_df.iloc[boot_idx]

        X_train, y_train = boot_train[feature_cols], boot_train[target_col]
        X_test = test_df[feature_cols]

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=random_state + b,
            verbose=-1
        )
        model.fit(X_train, y_train)
        all_preds[b, :] = model.predict(X_test)

    disagreement = all_preds.std(axis=0)
    ensemble_mean = all_preds.mean(axis=0)

    return {'disagreement': disagreement, 'ensemble_mean': ensemble_mean}


def run_bootstrap_disagreement_walk_forward(df, feature_cols, n_bootstrap=10,
                                              initial_train_days=252 * 5, test_chunk_days=126):
    """
    Walk-forward version, mirroring base_model.py's fold structure.
    Returns a dataframe of date, disagreement, ensemble_mean, fold_id.
    """
    results = []
    n = len(df)
    fold_id = 0
    train_end = initial_train_days

    while train_end < n:
        test_end = min(train_end + test_chunk_days, n)

        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]

        fold_result = bootstrap_ensemble_disagreement(
            train_df, test_df, feature_cols, target_col='target', n_bootstrap=n_bootstrap
        )

        fold_df = pd.DataFrame({
            'date': test_df.index,
            'disagreement': fold_result['disagreement'],
            'ensemble_mean': fold_result['ensemble_mean'],
            'fold_id': fold_id
        })
        results.append(fold_df)

        fold_id += 1
        train_end = test_end

    return pd.concat(results, ignore_index=True)


if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from features import get_all_features
    from base_model import build_target, FEATURE_COLS

    features = get_all_features()
    df = build_target(features['AAPL'])

    print("Running bootstrap ensemble disagreement for AAPL (this will take a while)...")
    result = run_bootstrap_disagreement_walk_forward(df, FEATURE_COLS, n_bootstrap=10)

    print(f"\nTotal test points: {len(result)}")
    print(f"Number of folds: {result['fold_id'].nunique()}")
    print(f"Mean disagreement: {result['disagreement'].mean():.6f}")
    print(f"Max disagreement: {result['disagreement'].max():.6f}")

    max_idx = result['disagreement'].idxmax()
    print(f"Max disagreement occurred at date: {result.loc[max_idx, 'date']}")