#!/usr/bin/env python3
"""
Multi-Asset Monte Carlo Simulation of Big-Tech Stocks
=====================================================

A portfolio-quality Monte Carlo risk-analysis engine for a four-stock
big-tech universe (ORCL, PLTR, META, NVDA).

Methodology
-----------
1.  Load two years of daily adjusted-close prices (Yahoo Finance) and align
    all series on their common trading dates.
2.  Compute daily **log returns** and estimate, per stock, the annualized
    drift ``mu`` (252 x mean daily log return) and volatility ``sigma``
    (sqrt(252) x std of daily log returns), plus the 4x4 cross-stock
    correlation matrix of daily log returns.
3.  Simulate correlated price paths with **Geometric Brownian Motion (GBM)**:

        S(t+1) = S(t) * exp(mu * dt + sigma * sqrt(dt) * eps(t)),
        eps(t) ~ N(0, 1),  Corr(eps_i, eps_j) = Corr(r_i, r_j)

    Cross-stock dependence is preserved by drawing independent standard
    normals and colouring them with the **Cholesky decomposition** ``L`` of
    the historical correlation matrix:  eps = Z @ L.T.

    Note on drift conventions: ``mu`` is the *continuously compounded*
    (log-return) drift estimated directly from the data, so no additional
    Ito (-0.5 * sigma^2) correction appears in the exponent.  Under this
    parameterization the *expected arithmetic* 1-year return is
    ``exp(mu + sigma^2 / 2) - 1`` while the *median* is ``exp(mu) - 1``;
    both fall out of the simulation naturally.

4.  From 10,000 simulated 1-year paths per stock, compute risk metrics:
    expected / median return, 5th-95th percentile terminal prices,
    P(gain), P(gain > 20%), P(loss > 20%), 95% Value-at-Risk (VaR) and
    Conditional VaR (CVaR / expected shortfall).
5.  Build an equal-weighted (25% each), buy-and-hold portfolio and report
    expected return, volatility, VaR/CVaR and Sharpe ratio (rf = 4%).

Outputs
-------
* ``results/summary_metrics.csv`` - tidy per-stock (+ portfolio) metric table
* ``charts/*.png``                - six publication-quality figures
* Console report with sanity checks (path positivity, correlation fidelity)

This project is educational and does not constitute investment advice.
"""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # headless rendering - safe for CI / servers

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

# ---------------------------------------------------------------------------
# Configuration (edit these constants to re-run with different settings)
# ---------------------------------------------------------------------------
TICKERS: list[str] = ["ORCL", "PLTR", "META", "NVDA"]
N_SIMS: int = 10_000            # number of Monte Carlo paths per stock
HORIZON: int = 252              # simulation horizon in trading days (1 year)
TRADING_DAYS: int = 252         # trading days used for annualization
RANDOM_SEED: int = 42           # fixed seed -> fully reproducible results
RISK_FREE_RATE: float = 0.04    # annual risk-free rate for the Sharpe ratio
VAR_CONFIDENCE: float = 0.95    # confidence level for VaR / CVaR
N_SAMPLE_PATHS: int = 200       # faint sample paths drawn on fan charts

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
SUMMARY_CSV = os.path.join(RESULTS_DIR, "summary_metrics.csv")

DT = 1.0 / TRADING_DAYS         # simulation time step (in years)
SQRT_DT = np.sqrt(DT)

# ---------------------------------------------------------------------------
# Visual style - low-saturation warm palette, ample whitespace
# ---------------------------------------------------------------------------
STOCK_COLORS = {
    "ORCL": "#B5533C",   # muted terracotta
    "PLTR": "#D9A441",   # soft ochre
    "META": "#7A8B6F",   # sage green
    "NVDA": "#5F7A75",   # desaturated slate teal
}
INK = "#3B342C"          # warm near-black for text / key lines
MUTED = "#8A8177"        # warm gray for secondary elements
PAPER = "#FCFBF8"        # warm off-white axes background
GRID = "#E7E1D6"         # barely-there warm grid lines
ALERT = "#9C3B2A"        # deep rust for VaR / risk markers

WARM_CMAP = LinearSegmentedColormap.from_list(
    "warm_cream_rust",
    ["#FCFAF3", "#F0DDB4", "#DE9A5B", "#B5533C", "#7E2A1E"],
)


def apply_style() -> None:
    """Set a consistent, professional matplotlib/seaborn theme."""
    sns.set_theme(style="white")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": PAPER,
        "axes.edgecolor": "#C9C2B4",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelsize": 10.5,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "font.family": "DejaVu Sans",
        "figure.dpi": 100,
        "savefig.dpi": 150,
    })


def save_fig(fig: plt.Figure, name: str) -> str:
    """Save a figure into charts/ with consistent settings.

    The PNG is rendered into an in-memory buffer first and then written to
    disk with explicit flush/fsync and retries, which keeps the save robust
    on filesystems with eventually-consistent directory metadata.
    """
    path = os.path.join(CHARTS_DIR, name)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    payload = buffer.getvalue()
    last_err: OSError | None = None
    for _ in range(5):
        os.makedirs(CHARTS_DIR, exist_ok=True)
        try:
            with open(path, "wb") as fh:
                fh.write(payload)
                fh.flush()
                os.fsync(fh.fileno())
            break
        except OSError as err:  # transient metadata lag -> wait and retry
            last_err = err
            time.sleep(0.3)
    else:
        raise last_err  # type: ignore[misc]
    plt.close(fig)
    print(f"  saved charts/{name}")
    return path


# ---------------------------------------------------------------------------
# Data loading & parameter estimation
# ---------------------------------------------------------------------------
def load_prices(tickers: list[str]) -> pd.DataFrame:
    """Load adjusted-close prices for each ticker and align on common dates.

    Returns
    -------
    pd.DataFrame
        Columns = tickers, index = common trading dates (tz-naive), values =
        adjusted close prices. Rows with any missing value are dropped so the
        return series are perfectly aligned across assets.
    """
    series = {}
    for ticker in tickers:
        path = os.path.join(DATA_DIR, f"{ticker}_prices.csv")
        df = pd.read_csv(path, parse_dates=["Date"])
        # Normalize timestamps: strip timezone, keep the calendar date only.
        dates = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
        series[ticker] = pd.Series(df["Close"].to_numpy(), index=dates, name=ticker)

    prices = pd.DataFrame(series).sort_index().dropna()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices


def estimate_parameters(log_returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Estimate annualized drift, volatility and the correlation matrix.

    Parameters
    ----------
    log_returns : pd.DataFrame
        Daily log returns (rows = dates, columns = tickers).

    Returns
    -------
    mu : np.ndarray, shape (n_assets,)
        Annualized continuously compounded drift (252 x mean daily log return).
    sigma : np.ndarray, shape (n_assets,)
        Annualized volatility (sqrt(252) x daily std).
    corr : pd.DataFrame
        Sample correlation matrix of daily log returns.
    """
    mu = log_returns.mean().to_numpy() * TRADING_DAYS
    sigma = log_returns.std(ddof=1).to_numpy() * np.sqrt(TRADING_DAYS)
    corr = log_returns.corr()
    return mu, sigma, corr


# ---------------------------------------------------------------------------
# Correlated multi-asset GBM simulation
# ---------------------------------------------------------------------------
def simulate_gbm(
    S0: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    corr: np.ndarray,
    n_sims: int = N_SIMS,
    horizon: int = HORIZON,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Simulate correlated Geometric Brownian Motion price paths.

    Independent standard normals are drawn for every (day, simulation, asset)
    and then correlated via the Cholesky factor ``L`` of the historical
    correlation matrix, so simulated co-movement matches history:

        L = cholesky(Corr)          (lower-triangular)
        eps = Z @ L.T               (correlated standard normals)
        dlogS = mu*dt + sigma*sqrt(dt) * eps
        S(t)  = S0 * exp(cumsum(dlogS))

    Returns
    -------
    np.ndarray, shape (horizon + 1, n_sims, n_assets)
        Simulated price paths including the initial prices at t = 0.
        Strictly positive by construction (exponential of a real process).
    """
    n_assets = len(S0)
    rng = np.random.default_rng(seed)

    # Cholesky factorization of the (positive-definite) correlation matrix.
    chol = np.linalg.cholesky(corr)

    # Independent normals -> correlated normals, shape (days, sims, assets).
    z = rng.standard_normal((horizon, n_sims, n_assets))
    eps = z @ chol.T

    # GBM log-return increments; mu is the log-drift so no Ito correction.
    increments = mu * DT + sigma * SQRT_DT * eps

    # Accumulate log returns and exponentiate back to price space.
    log_paths = np.concatenate(
        [np.zeros((1, n_sims, n_assets)), np.cumsum(increments, axis=0)],
        axis=0,
    )
    return S0 * np.exp(log_paths)


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------
def var_cvar(returns: np.ndarray, confidence: float = VAR_CONFIDENCE) -> tuple[float, float]:
    """Value-at-Risk and Conditional VaR (expected shortfall) of simple returns.

    Both are returned as *positive* loss magnitudes, e.g. VaR = 0.32 means
    "losses exceed 32% with probability (1 - confidence)".
    """
    cutoff = np.percentile(returns, (1.0 - confidence) * 100.0)
    tail = returns[returns <= cutoff]
    return -cutoff, -tail.mean()


def stock_metrics(
    ticker: str,
    S0: float,
    mu: float,
    sigma: float,
    terminal_prices: np.ndarray,
) -> dict:
    """Compute the full per-stock metric block from simulated terminal prices."""
    rets = terminal_prices / S0 - 1.0
    var95, cvar95 = var_cvar(rets)
    return {
        "ticker": ticker,
        "current_price": round(float(S0), 2),
        "annual_drift_log": round(float(mu), 4),
        "annual_vol": round(float(sigma), 4),
        "expected_return_1y": round(float(rets.mean()), 4),
        "median_return_1y": round(float(np.median(rets)), 4),
        "p05_price_1y": round(float(np.percentile(terminal_prices, 5)), 2),
        "p95_price_1y": round(float(np.percentile(terminal_prices, 95)), 2),
        "prob_gain": round(float((rets > 0).mean()), 4),
        "prob_gain_gt_20pct": round(float((rets > 0.20).mean()), 4),
        "prob_loss_gt_20pct": round(float((rets < -0.20).mean()), 4),
        "var_95_1y": round(var95, 4),
        "cvar_95_1y": round(cvar95, 4),
        "sharpe": np.nan,  # only reported for the portfolio row
    }


def portfolio_analysis(paths: np.ndarray, annual_cov: np.ndarray) -> tuple[np.ndarray, dict]:
    """Equal-weighted (25% per stock), buy-and-hold portfolio of $1.

    No rebalancing: each quarter of capital follows its own stock's path,
    so V(t) = mean_i( S_i(t) / S_i(0) ).

    Two complementary volatility figures are reported:
    * ``annual_vol``    - annualized portfolio volatility sqrt(w' Sigma w)
                          from the historical covariance of daily log returns;
                          directly comparable to the per-stock volatilities and
                          the number that shows the diversification benefit.
    * ``return_vol_1y`` - standard deviation of the simulated 1-year simple
                          returns; the natural denominator for a 1-year-horizon
                          Sharpe ratio,  Sharpe = (E[R_1y] - rf) / std(R_1y).
    """
    n_assets = paths.shape[2]
    weights = np.full(n_assets, 1.0 / n_assets)
    gross = paths / paths[0]                      # growth of $1 per asset
    values = (gross * weights).sum(axis=2)        # portfolio value of $1
    rets = values[-1] - 1.0
    var95, cvar95 = var_cvar(rets)
    stats = {
        "expected_return": float(rets.mean()),
        "median_return": float(np.median(rets)),
        "annual_vol": float(np.sqrt(weights @ annual_cov @ weights)),
        "return_vol_1y": float(rets.std(ddof=1)),
        "var_95": var95,
        "cvar_95": cvar95,
        "prob_gain": float((rets > 0).mean()),
        "sharpe": float((rets.mean() - RISK_FREE_RATE) / rets.std(ddof=1)),
    }
    return values, stats


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def chart_fan(paths: np.ndarray, S0: np.ndarray) -> None:
    """2x2 fan charts: faint sample paths + 5th-95th band + median line."""
    days = np.arange(paths.shape[0])
    rng = np.random.default_rng(RANDOM_SEED + 1)
    sample = rng.choice(paths.shape[1], size=min(N_SAMPLE_PATHS, paths.shape[1]), replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"Monte Carlo fan charts - {N_SIMS:,} simulated 1-year paths per stock",
        fontsize=14, fontweight="bold", color=INK, y=0.995,
    )
    for ax, i, ticker in zip(axes.flat, range(len(TICKERS)), TICKERS):
        color = STOCK_COLORS[ticker]
        p = paths[:, :, i]
        ax.plot(days, p[:, sample], color=color, alpha=0.05, linewidth=0.7, zorder=1)
        p05, p50, p95 = np.percentile(p, [5, 50, 95], axis=1)
        ax.fill_between(days, p05, p95, color=color, alpha=0.22,
                        label="5th-95th percentile", zorder=2)
        ax.plot(days, p50, color=INK, linewidth=2.0, label="Median path", zorder=3)
        ax.scatter([0], [S0[i]], color=INK, s=28, zorder=4)
        ax.annotate(f"current ${S0[i]:,.2f}", (0, S0[i]),
                    textcoords="offset points", xytext=(8, -14), fontsize=9, color=INK)
        ax.set_title(ticker)
        ax.set_xlabel("Trading days ahead")
        ax.set_ylabel("Price ($)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
        # Trim only the most extreme path spikes so the band stays readable.
        ax.set_ylim(0.0, np.percentile(p, 99.5))
        ax.margins(x=0.02)
    axes.flat[0].legend(loc="upper left", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "fan_charts.png")


def chart_outcome_distributions(term_rets: np.ndarray) -> None:
    """2x2 histograms of simulated 1-year returns with VaR / median / mean lines."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        "Distribution of simulated 1-year returns",
        fontsize=14, fontweight="bold", color=INK, y=0.995,
    )
    for ax, i, ticker in zip(axes.flat, range(len(TICKERS)), TICKERS):
        color = STOCK_COLORS[ticker]
        r = term_rets[:, i] * 100.0
        ax.hist(r, bins=60, color=color, alpha=0.82, edgecolor="white", linewidth=0.3)
        p05, med, mean = np.percentile(r, 5), np.median(r), r.mean()
        ax.axvline(p05, color=ALERT, linestyle="--", linewidth=1.8,
                   label=f"VaR 95%: {p05:+.1f}%")
        ax.axvline(med, color=INK, linewidth=1.8, label=f"Median: {med:+.1f}%")
        ax.axvline(mean, color=MUTED, linestyle=":", linewidth=1.8,
                   label=f"Mean: {mean:+.1f}%")
        # Trim extreme tails for readability.
        lo, hi = np.percentile(r, [0.4, 99.6])
        ax.set_xlim(lo, hi)
        ax.set_title(ticker)
        ax.set_xlabel("1-year return (%)")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8.5, loc="upper left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, "outcome_distributions.png")


def chart_correlation_heatmap(corr: pd.DataFrame, date_label: str) -> None:
    """Annotated heatmap of historical daily log-return correlations."""
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap=WARM_CMAP, vmin=0.0, vmax=1.0,
        square=True, linewidths=1.2, linecolor="white",
        cbar_kws={"label": "Correlation", "shrink": 0.82},
        annot_kws={"size": 12, "color": INK}, ax=ax,
    )
    ax.set_title(f"Historical daily log-return correlations\n{date_label}", pad=14)
    ax.tick_params(axis="x", rotation=0)
    ax.tick_params(axis="y", rotation=0)
    ax.grid(False)
    # Keep annotation text legible on the darkest cells.
    for text, val in zip(ax.texts, corr.to_numpy().ravel()):
        if val >= 0.75:
            text.set_color("white")
    fig.tight_layout()
    save_fig(fig, "correlation_heatmap.png")


def chart_risk_return(sigma: np.ndarray, exp_rets: np.ndarray, prob_gain: np.ndarray) -> None:
    """Expected return vs volatility scatter; bubble size = P(gain)."""
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    x, y = sigma * 100.0, exp_rets * 100.0
    for i, ticker in enumerate(TICKERS):
        ax.scatter(x[i], y[i], s=350 + 1400 * prob_gain[i], color=STOCK_COLORS[ticker],
                   alpha=0.85, edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(ticker, (x[i], y[i]), textcoords="offset points", xytext=(0, 16),
                    ha="center", fontsize=11, fontweight="bold", color=INK)
    ax.axhline(0, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_xlim(x.min() - 8, x.max() + 8)
    pad = (y.max() - y.min()) * 0.22
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("Annualized volatility (%)")
    ax.set_ylabel("Expected 1-year return (%)")
    ax.set_title("Risk vs. expected return - bubble size = probability of a 1-year gain")
    fig.tight_layout()
    save_fig(fig, "risk_return_scatter.png")


def chart_historical(prices: pd.DataFrame, date_label: str) -> None:
    """Historical prices rebased to 100 at the first common trading day."""
    rebased = prices / prices.iloc[0] * 100.0
    fig, ax = plt.subplots(figsize=(11.5, 6.3))
    for ticker in TICKERS:
        ax.plot(rebased.index, rebased[ticker], color=STOCK_COLORS[ticker],
                linewidth=1.8, label=ticker)
    ax.axhline(100, color=MUTED, linewidth=0.9, linestyle="--")
    ax.set_title(f"Historical prices, rebased to 100 - {date_label}")
    ax.set_ylabel("Indexed price (100 = first trading day)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", fontsize=10, ncol=4)
    ax.margins(x=0.01)
    fig.tight_layout()
    save_fig(fig, "historical_prices.png")


def chart_dashboard(
    port_values: np.ndarray,
    port_stats: dict,
    summary: pd.DataFrame,
    date_label: str,
) -> None:
    """Combined dashboard: portfolio fan chart, return distribution, metric table."""
    fig = plt.figure(figsize=(14.5, 9.6))
    gs = GridSpec(2, 2, height_ratios=[1.15, 1.0], hspace=0.32, wspace=0.22,
                  left=0.06, right=0.97, top=0.875, bottom=0.07)
    fig.suptitle(
        "Monte Carlo dashboard - equal-weighted ORCL / PLTR / META / NVDA portfolio",
        fontsize=15, fontweight="bold", color=INK, y=0.975,
    )
    fig.text(0.5, 0.935,
             f"{N_SIMS:,} simulations x {HORIZON} trading days | GBM with Cholesky-correlated "
             f"shocks | parameters from {date_label} | seed = {RANDOM_SEED}",
             ha="center", fontsize=9.5, color=MUTED)

    # -- Panel 1 (top, full width): portfolio fan chart of a $100 investment --
    ax1 = fig.add_subplot(gs[0, :])
    days = np.arange(port_values.shape[0])
    rng = np.random.default_rng(RANDOM_SEED + 2)
    sample = rng.choice(port_values.shape[1], size=N_SAMPLE_PATHS, replace=False)
    pv = port_values * 100.0
    ax1.plot(days, pv[:, sample], color="#B5533C", alpha=0.045, linewidth=0.7, zorder=1)
    p05, p50, p95 = np.percentile(pv, [5, 50, 95], axis=1)
    ax1.fill_between(days, p05, p95, color="#B5533C", alpha=0.22,
                     label="5th-95th percentile", zorder=2)
    ax1.plot(days, p50, color=INK, linewidth=2.2, label="Median path", zorder=3)
    ax1.axhline(100, color=MUTED, linewidth=0.9, linestyle="--")
    ax1.annotate("initial $100", (0, 100), textcoords="offset points", xytext=(6, 8),
                 fontsize=9, color=MUTED)
    ax1.set_title("Simulated 1-year value of $100 (buy-and-hold, 25% per stock)")
    ax1.set_xlabel("Trading days ahead")
    ax1.set_ylabel("Portfolio value ($)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.margins(x=0.02)

    # -- Panel 2 (bottom left): portfolio return distribution + VaR / CVaR --
    ax2 = fig.add_subplot(gs[1, 0])
    r = (port_values[-1] - 1.0) * 100.0
    ax2.hist(r, bins=60, color="#B5533C", alpha=0.82, edgecolor="white", linewidth=0.3)
    var_x, cvar_x = -port_stats["var_95"] * 100, -port_stats["cvar_95"] * 100
    med, mean = np.median(r), r.mean()
    ax2.axvline(var_x, color=ALERT, linestyle="--", linewidth=1.8,
                label=f"VaR 95%: {var_x:+.1f}%")
    ax2.axvline(cvar_x, color=ALERT, linewidth=1.8,
                label=f"CVaR 95%: {cvar_x:+.1f}%")
    ax2.axvline(med, color=INK, linewidth=1.8, label=f"Median: {med:+.1f}%")
    ax2.axvline(mean, color=MUTED, linestyle=":", linewidth=1.8, label=f"Mean: {mean:+.1f}%")
    ax2.set_xlim(np.percentile(r, [0.4, 99.6]))
    ax2.set_title("Portfolio 1-year return distribution")
    ax2.set_xlabel("1-year return (%)")
    ax2.set_ylabel("Frequency")
    leg2 = ax2.legend(fontsize=8.5, loc="upper left", frameon=True, framealpha=0.92)
    leg2.get_frame().set_edgecolor(GRID)
    ax2.text(0.985, 0.95,
             f"E[return] = {port_stats['expected_return']:+.1%}\n"
             f"Volatility = {port_stats['annual_vol']:.1%}\n"
             f"Sharpe (rf=4%) = {port_stats['sharpe']:.2f}\n"
             f"P(gain) = {port_stats['prob_gain']:.1%}",
             transform=ax2.transAxes, ha="right", va="top", fontsize=9.5, color=INK,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                       edgecolor=GRID, alpha=0.95))

    # -- Panel 3 (bottom right): per-stock metric table --
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axis("off")
    ax3.set_title("Key metrics (1-year horizon)", pad=8)
    cols = ["Ticker", "E[Return]", "Volatility", "VaR 95%", "P(Gain)"]
    rows = [
        [row.ticker, f"{row.expected_return_1y:+.1%}", f"{row.annual_vol:.1%}",
         f"{row.var_95_1y:.1%}", f"{row.prob_gain:.0%}"]
        for row in summary.itertuples()
    ]
    rows.append(["PORTFOLIO", f"{port_stats['expected_return']:+.1%}",
                 f"{port_stats['annual_vol']:.1%}", f"{port_stats['var_95']:.1%}",
                 f"{port_stats['prob_gain']:.0%}"])
    table = ax3.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.9)
    for (r_i, c_i), cell in table.get_celld().items():
        cell.set_edgecolor("white")
        if r_i == 0:
            cell.set_facecolor("#B5533C")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#F5F0E6" if r_i % 2 else "#FFFFFF")
            if r_i == len(rows):  # portfolio row emphasis
                cell.set_facecolor("#EAD9C2")
                cell.set_text_props(fontweight="bold")
    save_fig(fig, "dashboard.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHARTS_DIR, exist_ok=True)
    apply_style()

    print("=" * 74)
    print("MULTI-ASSET MONTE CARLO SIMULATION - BIG-TECH STOCKS")
    print("=" * 74)

    # --- 1. Data -----------------------------------------------------------
    print("\n[1/6] Loading price data ...")
    prices = load_prices(TICKERS)
    date_label = (f"{prices.index[0]:%b %d, %Y} - {prices.index[-1]:%b %d, %Y}")
    print(f"  {len(prices)} aligned trading days ({date_label}), tickers: {', '.join(TICKERS)}")

    log_returns = np.log(prices / prices.shift(1)).dropna()
    mu, sigma, corr = estimate_parameters(log_returns)
    S0 = prices.iloc[-1].to_numpy()

    print("\n  Estimated annualized parameters (from daily log returns):")
    est = pd.DataFrame(
        {"drift (log)": mu, "volatility": sigma, "current price": S0},
        index=TICKERS,
    )
    print(est.round(4).to_string())
    print("\n  Correlation matrix of daily log returns:")
    print(corr.round(3).to_string())

    # --- 2. Simulation -----------------------------------------------------
    print(f"\n[2/6] Simulating {N_SIMS:,} correlated GBM paths x {HORIZON} days ...")
    corr_np = corr.to_numpy()
    paths = simulate_gbm(S0, mu, sigma, corr_np)

    # --- 3. Sanity checks --------------------------------------------------
    print("\n[3/6] Sanity checks ...")
    assert np.isfinite(paths).all(), "non-finite values in simulated paths"
    assert (paths > 0).all(), "GBM prices must be strictly positive"
    print(f"  all simulated prices positive (min = ${paths.min():,.2f}) - OK")

    # Correlation fidelity: pool simulated daily log returns across paths and
    # compare their correlation matrix with the historical target.
    sim_increments = np.diff(np.log(paths), axis=0).reshape(-1, len(TICKERS))
    sim_corr = np.corrcoef(sim_increments.T)
    max_dev = np.abs(sim_corr - corr_np).max()
    print(f"  max |simulated - historical| correlation deviation = {max_dev:.4f} - OK")
    assert max_dev < 0.05, "correlation structure not preserved within tolerance"

    # --- 4. Per-stock metrics ----------------------------------------------
    print("\n[4/6] Computing per-stock risk metrics ...")
    terminal = paths[-1]                                # (n_sims, n_assets)
    term_rets = terminal / S0 - 1.0
    rows = [
        stock_metrics(t, S0[i], mu[i], sigma[i], terminal[:, i])
        for i, t in enumerate(TICKERS)
    ]
    summary = pd.DataFrame(rows)

    # --- 5. Portfolio -------------------------------------------------------
    print("\n[5/6] Equal-weighted portfolio analysis ...")
    annual_cov = log_returns.cov().to_numpy() * TRADING_DAYS
    port_values, port_stats = portfolio_analysis(paths, annual_cov)
    summary.loc[len(summary)] = {
        "ticker": "PORTFOLIO",
        "current_price": np.nan,
        "annual_drift_log": np.nan,
        "annual_vol": round(port_stats["annual_vol"], 4),
        "expected_return_1y": round(port_stats["expected_return"], 4),
        "median_return_1y": round(port_stats["median_return"], 4),
        "p05_price_1y": np.nan,
        "p95_price_1y": np.nan,
        "prob_gain": round(port_stats["prob_gain"], 4),
        "prob_gain_gt_20pct": round(float((port_values[-1] - 1.0 > 0.20).mean()), 4),
        "prob_loss_gt_20pct": round(float((port_values[-1] - 1.0 < -0.20).mean()), 4),
        "var_95_1y": round(port_stats["var_95"], 4),
        "cvar_95_1y": round(port_stats["cvar_95"], 4),
        "sharpe": round(port_stats["sharpe"], 4),
    }
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"  saved results/summary_metrics.csv")

    print("\n  Summary metrics (returns / VaR as decimals, prices in $):")
    print(summary.to_string(index=False))
    print("\n  Portfolio (equal weight, buy-and-hold, rf = 4%):")
    print(f"    expected 1-year return : {port_stats['expected_return']:+.2%}")
    print(f"    median 1-year return   : {port_stats['median_return']:+.2%}")
    print(f"    annual volatility      : {port_stats['annual_vol']:.2%}")
    print(f"    95% VaR  (1 year)      : {port_stats['var_95']:.2%}")
    print(f"    95% CVaR (1 year)      : {port_stats['cvar_95']:.2%}")
    print(f"    Sharpe ratio           : {port_stats['sharpe']:.2f}")

    # --- 6. Charts ----------------------------------------------------------
    print("\n[6/6] Rendering charts ...")
    chart_historical(prices, date_label)
    chart_correlation_heatmap(corr, date_label)
    chart_fan(paths, S0)
    chart_outcome_distributions(term_rets)
    chart_risk_return(sigma, term_rets.mean(axis=0), summary.iloc[:4]["prob_gain"].to_numpy())
    chart_dashboard(port_values, port_stats, summary.iloc[:4], date_label)

    print("\nDone. Outputs written to results/ and charts/.")
    print("=" * 74)


if __name__ == "__main__":
    main()
