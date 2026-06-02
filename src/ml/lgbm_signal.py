import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats


def build_features(ret_wide):
    """
    Build cross-sectional features from returns.
    All features use only past data (.shift(1) minimum).

    Features per stock per day:
    - mom_12_1:  12-1 month momentum
    - mom_1m:    1-month return
    - reversal:  -5 day return (mean-reversion)
    - vol_21:    21-day volatility
    - skew_63:   63-day return skewness
    - vol_ratio: short-term vol / long-term vol (vol regime)
    """
    features = {}
    price = ret_wide.cumsum()  # log-price proxy

    features["mom_12_1"] = (price - price.shift(252)).shift(21)
    features["mom_1m"] = ret_wide.rolling(21).sum().shift(1)
    features["reversal"] = -ret_wide.rolling(5).sum().shift(1)
    features["vol_21"] = ret_wide.rolling(21).std().shift(1)
    features["skew_63"] = ret_wide.rolling(63).skew().shift(1)
    features["vol_ratio"] = (ret_wide.rolling(21).std() /
                              ret_wide.rolling(63).std()).shift(1)

    return features


def build_ml_dataset(ret_wide, features, horizon=21):
    """
    Build flat ML dataset: one row per (date, ticker).
    Target: forward return over `horizon` days.
    """
    target = ret_wide.shift(-horizon)  # future returns

    rows = []
    for date in ret_wide.index:
        for ticker in ret_wide.columns:
            row = {"date": date, "ticker": ticker}
            for feat_name, feat_df in features.items():
                if date in feat_df.index and ticker in feat_df.columns:
                    row[feat_name] = feat_df.loc[date, ticker]
                else:
                    row[feat_name] = np.nan
            if date in target.index and ticker in target.columns:
                row["target"] = target.loc[date, ticker]
            else:
                row["target"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def train_lgbm_signal(ret_wide, horizon=21, n_splits=5):
    """
    Train LightGBM with purged walk-forward CV.
    Returns: pd.DataFrame with predicted scores per (date, ticker)
    """
    features = build_features(ret_wide)
    feature_names = list(features.keys())

    print("Building ML dataset... (this takes a few minutes)")
    dataset = build_ml_dataset(ret_wide, features, horizon=horizon)
    dataset = dataset.dropna().sort_values("date")

    dates = dataset["date"].unique()
    n = len(dates)
    pred_rows = []

    # Purged walk-forward: train on past, predict on future, gap of `horizon` days
    split_size = n // (n_splits + 1)

    for fold in range(n_splits):
        train_end_idx = split_size * (fold + 1)
        purge_end_idx = train_end_idx + (horizon // 5)  # purge gap
        test_end_idx = min(purge_end_idx + split_size, n)

        if purge_end_idx >= n:
            break

        train_dates = dates[:train_end_idx]
        test_dates = dates[purge_end_idx:test_end_idx]

        train_df = dataset[dataset["date"].isin(train_dates)]
        test_df = dataset[dataset["date"].isin(test_dates)]

        X_train = train_df[feature_names].values
        y_train = train_df["target"].values
        X_test = test_df[feature_names].values

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        test_df = test_df.copy()
        test_df["lgbm_raw"] = preds
        pred_rows.append(test_df[["date", "ticker", "lgbm_raw"]])

        print(f"  Fold {fold+1}/{n_splits} done.")

    pred_df = pd.concat(pred_rows).sort_values("date")
    signal_wide = pred_df.pivot(index="date", columns="ticker", values="lgbm_raw")
    signal_wide.index = pd.to_datetime(signal_wide.index)
    return signal_wide