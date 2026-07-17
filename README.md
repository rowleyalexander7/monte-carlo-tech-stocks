# Monte Carlo Simulation of Big-Tech Stocks

A multi-asset Monte Carlo risk-analysis engine for **Oracle (ORCL), Palantir (PLTR), Meta (META) and NVIDIA (NVDA)**. The project estimates return dynamics from two years of real market data, simulates **10,000 correlated 1-year price paths per stock** with Geometric Brownian Motion, and translates the simulated distributions into decision-relevant risk metrics (expected return, VaR, CVaR, gain/loss probabilities) plus an equal-weighted portfolio study — all wrapped in publication-quality visualizations.

![Dashboard](charts/dashboard.png)

## What it does

1. **Loads** 2 years of daily adjusted-close prices (Yahoo Finance, Jul 17 2024 – Jul 17 2026; 502 trading days) and aligns all four series on common trading dates.
2. **Estimates** per-stock annualized drift (`mu`) and volatility (`sigma`) from daily log returns, plus the 4×4 correlation matrix.
3. **Simulates** 10,000 correlated GBM paths per stock over a 252-trading-day horizon. Cross-stock dependence is preserved by colouring independent Gaussian shocks with the **Cholesky decomposition** of the historical correlation matrix.
4. **Reports** per stock: expected & median 1-year return, 5th–95th percentile prices, P(gain), P(gain > 20%), P(loss > 20%), and 95% VaR / CVaR.
5. **Analyzes** an equal-weighted (25% each), buy-and-hold portfolio: expected return, volatility, VaR/CVaR, Sharpe ratio (rf = 4%).
6. **Renders** six charts and a tidy metrics CSV. Fully reproducible via a fixed random seed (`RANDOM_SEED = 42`).

## Methodology

**Geometric Brownian Motion.** Each stock follows

```
S(t+1) = S(t) · exp( mu·dt + sigma·sqrt(dt)·eps(t) ),   eps ~ N(0,1)
```

where `mu` is the annualized mean log return (continuously compounded drift) and `sigma` the annualized volatility, both estimated from 502 days of log returns. Under this parameterization the expected *arithmetic* 1-year return is `exp(mu + sigma²/2) − 1`, which is why simulated means exceed medians — the lognormal right skew is a feature, not a bug.

**Correlated multi-asset simulation.** Independent standard normals `Z` (shape: days × sims × assets) are correlated via the Cholesky factor `L` of the historical correlation matrix: `eps = Z @ L.T`. Simulated co-movement matches history to within **0.0008** max absolute correlation deviation (validated at run time).

**Risk metrics.** VaR-95% is the 5th-percentile loss of the simulated 1-year return distribution; CVaR-95% (expected shortfall) is the average loss in that worst 5% tail — a coherent tail-risk measure that captures severity beyond the VaR cutoff.

**Portfolio.** Equal weights (25% per stock), buy-and-hold, no rebalancing. Portfolio volatility `sqrt(w'Σw)` from the historical covariance matrix; Sharpe ratio = `(E[R_1y] − rf) / std(R_1y)` with rf = 4%.

## How to run

```bash
pip install -r requirements.txt
python monte_carlo.py
```

Runtime is a few seconds. Outputs are written to `results/` and `charts/`. Key parameters (`TICKERS`, `N_SIMS`, `HORIZON`, `RANDOM_SEED`, `RISK_FREE_RATE`) are configurable constants at the top of `monte_carlo.py`.

## File structure

```
monte-carlo-tech-stocks/
├── monte_carlo.py              # simulation engine + metrics + all charts
├── requirements.txt            # numpy, pandas, matplotlib, seaborn
├── data/                       # 2 years of daily OHLCV per ticker (Yahoo Finance)
│   ├── ORCL_prices.csv
│   ├── PLTR_prices.csv
│   ├── META_prices.csv
│   └── NVDA_prices.csv
├── results/
│   └── summary_metrics.csv     # tidy per-stock + portfolio metrics
└── charts/
    ├── historical_prices.png       # rebased (100) price history
    ├── correlation_heatmap.png     # daily log-return correlations
    ├── fan_charts.png              # 2x2 simulated-path fan charts
    ├── outcome_distributions.png   # 2x2 1-year return histograms w/ VaR
    ├── risk_return_scatter.png     # E[return] vs vol, sized by P(gain)
    └── dashboard.png               # combined portfolio dashboard
```

## Sample results

Parameters estimated from Jul 17, 2024 – Jul 17, 2026 daily log returns; 10,000 simulations, 252-day horizon, seed 42.

| Ticker | Price | Exp. 1y return | Median | Ann. vol | P(gain) | P(>+20%) | P(>−20%) | 95% VaR | 95% CVaR | 5th–95th pct price |
|---|---|---|---|---|---|---|---|---|---|---|
| ORCL | $127.46 | **+12.9%** | −2.6% | 54.1% | 48.2% | 34.5% | 36.2% | 60.3% | 67.9% | $50.54 – $305.34 |
| PLTR | $133.93 | **+168.9%** | +119.2% | 62.6% | 89.6% | 83.2% | 5.2% | 21.1% | 39.1% | $105.63 – $830.73 |
| META | $645.97 | **+27.6%** | +19.0% | 37.3% | 67.9% | 48.9% | 14.8% | 36.1% | 44.8% | $412.64 – $1,429.78 |
| NVDA | $203.90 | **+47.5%** | +31.4% | 47.5% | 71.6% | 57.7% | 14.8% | 39.6% | 49.8% | $123.11 – $589.24 |
| **Portfolio (equal weight)** | — | **+64.2%** | +48.7% | **38.0%** | **83.7%** | 70.4% | 5.9% | **23.1%** | 33.8% | — |

Portfolio Sharpe ratio (rf = 4%): **0.81**. VaR/CVaR are expressed as loss magnitudes.

## Key findings

- **Diversification works, measurably.** Pairwise return correlations are moderate (0.30–0.49), so the equal-weighted portfolio's volatility drops to **38.0%** vs a ~50% average single-stock volatility, while keeping a +64.2% expected return — P(gain) rises to 83.7% and 95% VaR falls to 23.1%, better than three of the four constituents.
- **PLTR is the high-octane outlier.** The strongest drift (+78% annualized log) *and* the highest volatility (62.6%) produce an expected +168.9% 1-year return, yet a 119.2% median — the gap is pure lognormal right-skew, i.e. the mean is carried by rare extreme outcomes.
- **ORCL is the weakest risk profile.** Slightly negative drift over the window (−3.4% log) yields a *negative* median outcome (−2.6%), only a 48.2% chance of finishing higher, and the worst tail risk of the group (95% VaR 60.3%, CVaR 67.9%).
- **NVDA offers the best single-stock risk/return trade-off** (+47.5% expected on 47.5% vol), while META is the most defensive name (lowest vol, 37.3%; smallest 95% VaR after PLTR, 36.1%).
- **Every stock's mean exceeds its median** (e.g. NVDA +47.5% vs +31.4%), a direct consequence of GBM's lognormal terminal distribution — an important reminder that "expected" returns overstate the *typical* outcome for high-volatility names.

*Caveats:* drift and volatility are assumed constant and extrapolated from a trailing 2-year window that contained exceptional tech gains; GBM has no jumps, regime changes, or fat tails. Historical performance does not guarantee future results.

## Resume bullet points

- Built a multi-asset Monte Carlo simulation engine in Python (NumPy, pandas) generating 10,000 correlated 1-year price paths for 4 big-tech stocks via Geometric Brownian Motion with Cholesky-decomposed correlation structure; validated correlation fidelity to within 0.001 of historical estimates.
- Estimated annualized drift, volatility, and a 4×4 correlation matrix from 500+ days of Yahoo Finance price data; computed 95% VaR/CVaR, percentile outcome cones, and gain/loss probabilities, quantifying a diversification benefit of 12+ volatility points (38% portfolio vs ~50% average single-stock) for an equal-weighted portfolio.
- Delivered a reproducible analytics pipeline (fixed-seed, config-driven) that outputs a tidy metrics CSV and six publication-quality matplotlib/seaborn visualizations, including fan charts, return-distribution histograms with VaR markers, and a portfolio risk dashboard.
- Derived actionable risk insights from simulated distributions — e.g. identified an 84% probability of portfolio gain with 23% 95% VaR and a 0.81 Sharpe ratio (rf = 4%), and flagged right-skew distortion where mean 1-year returns exceeded medians by up to 50 percentage points (PLTR).

## Disclaimer

This project is for **educational and portfolio-demonstration purposes only**. It is not investment advice, and simulated outcomes are not predictions of future performance. Consult a licensed financial advisor before making investment decisions.
