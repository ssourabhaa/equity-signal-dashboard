"""
Compare sample covariance vs GNN-learned covariance.
Run with: python -m scripts.compare_cov
"""
import pandas as pd
import torch

from src.db import get_connection
from src.signals import (load_returns_wide, compute_momentum,
                         compute_mean_reversion, cross_sectional_zscore,
                         composite_signal)
from src.backtest import (walk_forward_backtest,
                          compute_performance_metrics)
from src.ml.gnn_cov import GATCovariance, get_gnn_covariance


def make_gnn_cov_func(model):
    """Factory: returns a cov_func compatible with walk_forward_backtest."""
    def cov_func(ret_window):
        return get_gnn_covariance(model, ret_window)
    return cov_func


if __name__ == "__main__":
    con = get_connection()
    ret_wide = load_returns_wide(con)

    # Build the composite signal (same as V2 Portfolio page)
    mom_z = cross_sectional_zscore(compute_momentum(ret_wide))
    rev_z = cross_sectional_zscore(compute_mean_reversion(ret_wide))
    comp_z = composite_signal({"momentum": mom_z, "mean_rev": rev_z})

    # Load trained GNN
    model = GATCovariance()
    model.load_state_dict(torch.load("data/gnn_best.pt"))
    model.eval()

    print("\n=== Backtest with sample covariance ===")
    bt_sample = walk_forward_backtest(comp_z, ret_wide.fillna(0))
    m_sample = compute_performance_metrics(bt_sample["portfolio_return"])
    print(f"  Sharpe:       {m_sample['sharpe']:.3f}")
    print(f"  Sortino:      {m_sample['sortino']:.3f}")
    print(f"  Max drawdown: {m_sample['max_drawdown']:.2%}")

    print("\n=== Backtest with GNN covariance ===")
    bt_gnn = walk_forward_backtest(
        comp_z, ret_wide.fillna(0),
        cov_func=make_gnn_cov_func(model)
    )
    m_gnn = compute_performance_metrics(bt_gnn["portfolio_return"])
    print(f"  Sharpe:       {m_gnn['sharpe']:.3f}")
    print(f"  Sortino:      {m_gnn['sortino']:.3f}")
    print(f"  Max drawdown: {m_gnn['max_drawdown']:.2%}")

    # Save both backtest series for the Streamlit Model Comparison page (Day 15)
    bt_sample.to_parquet("data/bt_sample.parquet")
    bt_gnn.to_parquet("data/bt_gnn.parquet")
    print("\nSaved backtest series to data/bt_sample.parquet and data/bt_gnn.parquet")

    con.close()