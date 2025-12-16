# MOVE Index Cross-Asset Volatility Research

A buy-side quantitative research study analyzing the predictive power of Treasury implied volatility (MOVE Index) for cross-asset volatility forecasting.

**Author:** Allen Xu  
**Style:** Cubist / Point72 Buy-Side Workflow

## Research Question

Does Treasury Implied Volatility (MOVE Index) predict:
1. Treasury Term-Spread Volatility (TLT/IEF spread proxy)
2. Technology Equity Implied Volatility (VXN)

## Key Findings

| Metric | Value |
|--------|-------|
| Mean Spearman IC | 0.293 |
| Out-of-Sample IC | 0.359 |
| OOS Hit Rate | 62.8% |
| Granger p-value | < 0.001 |
| Best Regularized Model | LASSO |

## Project Structure

```
move_volatility_research/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── config.py            # Configuration parameters
│   ├── data_loader.py       # Data acquisition & preprocessing
│   ├── features.py          # Feature engineering
│   ├── models.py            # Model implementations
│   ├── evaluation.py        # Backtesting & metrics
│   └── visualization.py     # Plotting functions
├── notebooks/               # Jupyter notebooks
├── reports/figures/         # Generated charts
├── data/                    # Processed data
├── tests/                   # Unit tests
├── main.py                  # Entry point
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/allenxu/move-volatility-research.git
cd move-volatility-research
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## License

MIT License
