# Equity Signal Dashboard

A walk-forward backtested equity portfolio system built around four alpha signals — momentum, mean-reversion, LightGBM, LSTM — plus a Graph Neural Network learned covariance matrix and a FinBERT sentiment overlay. Live Streamlit dashboard with 4 views.

## Architecture

```
yfinance (price data)              FinBERT + yfinance.news (NLP)
        │                                   │
        ▼                                   ▼
DuckDB prices table              Sentiment score per ticker
        │                                   │
        ▼                                   ▼
Log returns (VIEW)        ┌─── Composite z-score signal ───┐
        │                 │  (momentum + mean-rev          │
        ├── Momentum 12-1 ┤   + LightGBM + FinBERT)        │
        ├── Mean-rev 5d ──┤                                │
        ├── LightGBM ─────┤                                │
        └── LSTM ─────────┘                                │
                                            ▼
                  GNN (GAT) ──► Learned covariance matrix
                                            │
                                            ▼
                                  cvxpy MV optimiser
                                            │
                                            ▼
                            Walk-forward backtest
                            VaR / ES / FF3 exposure
                                            │
                                            ▼
                          Streamlit dashboard (4 views)
```

## Key Results

| Metric | Sample Cov | GNN Cov |
|---|---|---|
| Sharpe       | 0.627  | **0.692** (+10.4%) |
| Sortino      | 0.799  | **0.881** |
| Max Drawdown | -36.4% | **-35.7%** |

GNN-learned covariance outperforms sample covariance on every risk-adjusted metric.

## Signal Information Coefficients

| Signal              | Mean IC | ICIR | Type     |
|---------------------|---------|------|----------|
| Momentum 12-1       | 0.0143  | 0.05 | Classical |
| Mean-Reversion 5d   | 0.0006  | 0.00 | Classical |
| LightGBM            | 0.0042  | 0.02 | ML        |
| LSTM                | 0.0093  | 0.05 | ML        |
| FinBERT Sentiment   | N/A (limited history) | — | NLP/SOTA |

## Stack

- **Data**: yfinance, pandas-datareader (Fama-French)
- **Storage**: DuckDB
- **Optimisation**: cvxpy with OSQP solver
- **ML**: LightGBM, PyTorch (LSTM, GAT)
- **NLP**: HuggingFace transformers, ProsusAI/finbert
- **UI**: Streamlit, Plotly
- **Testing**: pytest

## Quick Start

```bash
# Setup
python -m venv venv
source venv/Scripts/activate    # Windows
pip install -r requirements.txt

# Ingest 10 years of price data for 50 stocks
python -m src.ingest

# Train ML signals (one-time, ~30 min on CPU)
python -m scripts.train_ml_signals

# Train GNN covariance (~60 min on CPU)
python -c "from src.db import get_connection; from src.signals import load_returns_wide; from src.ml.gnn_cov import train_gnn; con=get_connection(); train_gnn(load_returns_wide(con).fillna(0), epochs=100); con.close()"

# Compare sample vs GNN covariance
python -m scripts.compare_cov

# Score FinBERT sentiment for current news
python -m scripts.run_finbert

# Launch dashboard
streamlit run app.py
```

## Tests

```bash
python -m pytest tests/ -v
```

## Known Limitations

Being honest about limitations is a sign of research maturity.

1. **FinBERT historical data**: yfinance.news only provides ~2 weeks of headlines. A proper historical sentiment signal would require EDGAR filings (SEC), GDELT, or a paid news archive.
2. **Universe size**: 50 stocks is small for production signal research. Industry standard is 500–3000 stocks.
3. **Transaction costs**: 5 bps one-way is optimistic. Realistic costs for mid-cap stocks can be 20–50 bps.
4. **GNN covariance**: Trained on the same data used for backtesting (limited by data availability). Ideally trained on a separate period.
5. **No live trading**: This is a research system, not an execution system.

## License

MIT