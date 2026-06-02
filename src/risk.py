import numpy as np
import pandas as pd
import pandas_datareader as pdr


def historical_var(returns, confidence=0.99):
    """
    Historical VaR: sort returns, take the (1-confidence) percentile.
    No distributional assumption — uses actual data.
    """
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def expected_shortfall(returns, confidence=0.99):
    """
    ES (CVaR): average of returns worse than VaR.
    More informative than VaR — tells you the expected loss in the tail.
    """
    var = historical_var(returns, confidence)
    tail = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else var


def load_ff3_factors(start="2014-01-01"):
    """
    Download Fama-French 3-factor data (free, from Ken French's website).
    Factors: Mkt-RF (market excess return), SMB (small minus big), HML (value minus growth).
    """
    try:
        ff3 = pdr.get_data_famafrench("F-F_Research_Data_Factors_daily", start=start)[0]
        ff3.columns = [c.strip() for c in ff3.columns]
        ff3 = ff3 / 100  # convert from percent to decimal
        ff3.index = pd.to_datetime(ff3.index)
        return ff3
    except Exception as e:
        print(f"FF3 download failed: {e}")
        return None


def ff3_regression(portfolio_returns, ff3_df):
    """
    OLS regression: portfolio_return = alpha + beta_mkt*Mkt + beta_smb*SMB + beta_hml*HML + e
    Returns alpha, betas, and R-squared.
    Alpha > 0 means your strategy earns more than factor exposure explains.
    """
    from sklearn.linear_model import LinearRegression

    df = pd.concat([portfolio_returns, ff3_df], axis=1).dropna()
    df.columns = ["portfolio", "mkt_rf", "smb", "hml", "rf"]
    df["excess"] = df["portfolio"] - df["rf"]

    X = df[["mkt_rf", "smb", "hml"]].values
    y = df["excess"].values

    model = LinearRegression().fit(X, y)

    return {
        "alpha_annualised": model.intercept_ * 252,
        "beta_mkt": model.coef_[0],
        "beta_smb": model.coef_[1],
        "beta_hml": model.coef_[2],
        "r_squared": model.score(X, y),
    }