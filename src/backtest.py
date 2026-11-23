"""
12-Month Buy-and-Hold Backtest & Performance Metrics (§6 of `project_plan.md`).

Compares:
- Plain Top-10 (Version A)
- Climate Top-10 (Version B)
- Equal-Weight (across the 20 agro candidates)
- IBOV (^BVSP)
- Agro ETF (AGRI11 / BBGO11)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class BacktestMetrics:
    """Summary performance metrics for a 12-month buy-and-hold strategy (§6)."""

    name: str
    total_return: float  # Decimal cumulative return over the 12 months
    annualized_volatility: float  # Decimal annualized volatility
    max_drawdown: float  # Decimal max drawdown (negative or zero)
    sharpe_vs_cdi: float  # Annualized Sharpe ratio using CDI as risk-free rate
    wealth_index: pd.Series  # Cumulative wealth starting at 1.0 (length T+1)
    monthly_returns: pd.Series  # Realized monthly portfolio returns (length T)


def compute_buy_and_hold_returns(
    R_test: pd.DataFrame, weights: np.ndarray, tickers: List[str]
) -> Tuple[pd.Series, pd.Series]:
    """
    Simulate a strict buy-and-hold portfolio over the test months (§6):
    Optimize once at t=0 with initial weights `w_0`, hold for 12 months with no
    rebalancing and no transaction costs.

    Position values evolve as:
        V_{t,i} = w_{0,i} * prod_{s=1..t} (1 + r_{s,i})
        V_t = sum_i V_{t,i}
    """
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    sub_R = R_test[tickers].fillna(0.0)

    # Gross return factors per asset
    gross = 1.0 + sub_R.values  # (T, N)
    cum_asset_wealth = np.vstack([w, w * np.cumprod(gross, axis=0)])  # (T+1, N)
    port_wealth = cum_asset_wealth.sum(axis=1)  # (T+1,)

    # Build DatetimeIndex with an initial base date 1 month prior to first test month
    first_dt = pd.Timestamp(sub_R.index[0])
    base_dt = (first_dt.to_period("M") - 1).to_timestamp()
    wealth_idx = pd.DatetimeIndex([base_dt] + list(sub_R.index))

    wealth_series = pd.Series(port_wealth, index=wealth_idx, name="wealth")
    port_monthly_rets = wealth_series.pct_change().iloc[1:]
    port_monthly_rets.index = sub_R.index
    return wealth_series, port_monthly_rets


def evaluate_series_metrics(
    name: str,
    monthly_rets: pd.Series,
    cdi_monthly: np.ndarray,
) -> BacktestMetrics:
    """
    Compute total return, annualized volatility, maximum drawdown, and Sharpe vs CDI (§6).
    """
    r = monthly_rets.fillna(0.0).values
    T = len(r)
    cdi = np.asarray(cdi_monthly[:T], dtype=float)

    # Wealth path starting at 1.0
    wealth_vals = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    first_dt = pd.Timestamp(monthly_rets.index[0])
    base_dt = (first_dt.to_period("M") - 1).to_timestamp()
    wealth_idx = pd.DatetimeIndex([base_dt] + list(monthly_rets.index))
    wealth_series = pd.Series(wealth_vals, index=wealth_idx, name=name)

    total_ret = float(wealth_vals[-1] - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(12.0)) if T > 1 else 0.0

    # Max drawdown on the wealth path
    running_max = np.maximum.accumulate(wealth_vals)
    drawdowns = (wealth_vals - running_max) / running_max
    max_dd = float(np.min(drawdowns))

    # Sharpe vs CDI (annualized excess return / annualized volatility)
    cdi_total = float(np.prod(1.0 + cdi) - 1.0)
    excess_monthly = r - cdi
    ann_excess = float(np.mean(excess_monthly) * 12.0)
    excess_vol = float(np.std(excess_monthly, ddof=1) * np.sqrt(12.0))
    sharpe = (ann_excess / ann_vol) if ann_vol > 1e-12 else 0.0

    return BacktestMetrics(
        name=name,
        total_return=total_ret,
        annualized_volatility=ann_vol,
        max_drawdown=max_dd,
        sharpe_vs_cdi=sharpe,
        wealth_index=wealth_series,
        monthly_returns=monthly_rets,
    )


def run_backtest_comparison(
    R_test: pd.DataFrame,
    tickers_20: List[str],
    w_plain_20: np.ndarray,
    w_climate_20: np.ndarray,
    cdi_test: np.ndarray,
    etf_col: Optional[str] = None,
) -> Dict[str, BacktestMetrics]:
    """
    Run the 4-way backtest comparison:
    1. Plain (A)
    2. Climate (B)
    3. Agro ETF (AGRI11 or BBGO11)
    4. CDI
    """
    results: Dict[str, BacktestMetrics] = {}

    # 1. Plain (A)
    _, r_plain = compute_buy_and_hold_returns(R_test, w_plain_20, tickers_20)
    results["Plain (A)"] = evaluate_series_metrics("Plain (A)", r_plain, cdi_test)

    # 2. Climate (B)
    _, r_clim = compute_buy_and_hold_returns(R_test, w_climate_20, tickers_20)
    results["Climate (B)"] = evaluate_series_metrics("Climate (B)", r_clim, cdi_test)

    # 3. Agro ETF (auto-select AGRI11 or BBGO11 based on availability)
    if etf_col is None:
        for cand in ["AGRI11", "BBGO11", "FIND11"]:
            if cand in R_test.columns and R_test[cand].notna().sum() >= 6:
                etf_col = cand
                break
    if etf_col and etf_col in R_test.columns:
        r_etf = R_test[etf_col].fillna(0.0)
        label = f"Agro ETF ({etf_col})"
        results[label] = evaluate_series_metrics(label, r_etf, cdi_test)

    # 4. CDI benchmark
    r_cdi = pd.Series(cdi_test[: len(R_test)], index=R_test.index, name="CDI")
    results["CDI"] = evaluate_series_metrics("CDI", r_cdi, cdi_test)

    return results
