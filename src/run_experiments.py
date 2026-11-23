"""
End-to-end experiment runner for the COS360 Climate-Aware Agro Portfolio project.

Executes:
1. Data loading & ENSO climate scenario weighting (2015-2024 estimation, 2025 test)
2. KKT closed-form benchmark vs. Log-Barrier (Gradient, Newton, BFGS) validation
3. Conditioning & convergence analysis across barrier stages t = 10^1 .. 10^6
4. Direct 20-stock portfolio comparison: Version A (Plain) vs Version B (Climate)
   (showing natural KKT sparsity: 8 and 9 active stocks)
5. Parametric sensitivity: Efficient frontier sweep (gamma in [0.5, 50])
6. 12-month buy-and-hold out-of-sample backtest (2025) & metrics table
7. Publication-quality figures saved to `outputs/` and markdown tables to `outputs/summary_tables.md`
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import BacktestMetrics, run_backtest_comparison
from src.barrier_solvers import (
    BarrierResult,
    barrier_method,
    closed_form_kkt,
    compute_efficient_frontier,
    solve_scipy_reference,
)
from src.data_loader import (
    EstimationDataset,
    load_estimation_data,
    moments,
)

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def plot_kkt_and_weights(
    tickers: List[str],
    w_kkt: np.ndarray,
    res_plain: BarrierResult,
    res_climate: BarrierResult,
    out_path: str,
) -> None:
    """
    Generate Figure 1:
      Left panel: Unconstrained KKT closed-form weights vs. Log-Barrier weights (w >= 0).
      Right panel: Direct Log-Barrier weights for Plain (A) vs. Climate (B) on active stocks,
                   illustrating natural KKT sparsity (8 vs. 9 active stocks out of 20).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # --- Panel A: KKT Closed-Form (shorting allowed) vs Log-Barrier (w >= 0) ---
    ax = axes[0]
    order = np.argsort(w_kkt)[::-1]
    t_ord = [tickers[i] for i in order]
    x_pos = np.arange(len(tickers))
    width = 0.38

    ax.bar(
        x_pos - width / 2,
        w_kkt[order] * 100,
        width=width,
        label="KKT Closed Form (Budget only, shorting allowed)",
        color="#64748b",
        alpha=0.85,
    )
    ax.bar(
        x_pos + width / 2,
        res_plain.weights[order] * 100,
        width=width,
        label="Log-Barrier NLP ($w \\geq 0$, $\\gamma=5$)",
        color="#0284c7",
        alpha=0.90,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(t_ord, rotation=45, ha="right")
    ax.set_ylabel("Portfolio Weight (%)")
    ax.set_title("KKT Closed Form vs. Log-Barrier ($w \\geq 0$)")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # --- Panel B: Active Stocks (> 0.1%) in Plain (A) vs Climate (B) ---
    ax2 = axes[1]
    active_idx = [
        i
        for i in range(len(tickers))
        if res_plain.weights[i] > 1e-3 or res_climate.weights[i] > 1e-3
    ]
    active_idx.sort(
        key=lambda i: max(res_plain.weights[i], res_climate.weights[i]),
        reverse=True,
    )
    t_act = [tickers[i] for i in active_idx]
    x2 = np.arange(len(active_idx))

    ax2.bar(
        x2 - width / 2,
        res_plain.weights[active_idx] * 100,
        width=width,
        label=f"Plain (A): $p_t = 1/T$ ({int((res_plain.weights > 5e-3).sum())} active)",
        color="#0284c7",
    )
    ax2.bar(
        x2 + width / 2,
        res_climate.weights[active_idx] * 100,
        width=width,
        label=f"Climate (B): 11-Var Kernel ({int((res_climate.weights > 5e-3).sum())} active)",
        color="#ea580c",
    )
    ax2.set_xticks(x2)
    ax2.set_xticklabels(t_act, rotation=45, ha="right")
    ax2.set_ylabel("Optimal Weight (%)")
    ax2.set_title("KKT Active Set: Plain (A) vs. Climate (B) ($\\gamma=5$)")
    ax2.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_convergence_comparison(
    res_newton: BarrierResult,
    res_bfgs: BarrierResult,
    res_grad: BarrierResult,
    out_path: str,
) -> None:
    """
    Generate Figure 2:
      Left panel: Inner-loop gradient norm ||nabla phi_t(x_k)||_2 vs. iteration at t=1,000.
      Right panel: Iterations per barrier stage & Hessian condition number kappa(nabla^2 phi_t)
                   across t = 10^1 -> 10^6.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # --- Panel A: Gradient norm trajectory (Stage 3, t=10^3) ---
    ax = axes[0]
    stage_idx = 2  # t = 1000
    t_val = res_newton.stages[stage_idx].t
    for res, label, color, ls in [
        (res_newton, "Newton's Method (Quadratic)", "#16a34a", "-"),
        (res_bfgs, "Quasi-Newton BFGS (Superlinear)", "#2563eb", "--"),
        (res_grad, "Gradient Method (Linear)", "#dc2626", "-."),
    ]:
        g_hist = res.stages[stage_idx].grad_norm_history
        iters = np.arange(len(g_hist))
        ax.semilogy(
            iters,
            g_hist,
            label=f"{label} ({len(g_hist)-1} iters)",
            color=color,
            linestyle=ls,
            linewidth=2.0,
        )

    ax.axhline(
        1e-6, color="black", linestyle=":", linewidth=1.0, label="Tolerance ($10^{-6}$)"
    )
    if len(res_grad.stages[stage_idx].grad_norm_history) > 200:
        ax.set_xscale("symlog", linthresh=10)
    ax.set_xlim(left=0)
    ax.set_xlabel("Inner Iterations $k$ (symlog scale)")
    ax.set_ylabel("Gradient Norm $\\|\\nabla \\phi_t(x_k)\\|_2$ (log scale)")
    ax.set_title(f"Inner-Loop Convergence at Barrier Stage $t = {int(t_val):,d}$")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    # --- Panel B: Iterations per stage vs Hessian Condition Number across t = 10^1..10^6 ---
    ax2 = axes[1]
    t_grid = [s.t for s in res_newton.stages]
    x_idx = np.arange(len(t_grid))
    width = 0.25

    ax2.bar(
        x_idx - width,
        res_newton.stage_iters,
        width=width,
        label="Newton",
        color="#16a34a",
    )
    ax2.bar(x_idx, res_bfgs.stage_iters, width=width, label="BFGS", color="#2563eb")
    ax2.bar(
        x_idx + width,
        res_grad.stage_iters,
        width=width,
        label="Gradient",
        color="#dc2626",
    )
    ax2.set_yscale("log")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels([f"$10^{i+1}$" for i in range(len(t_grid))])
    ax2.set_xlabel("Barrier Parameter $t$")
    ax2.set_ylabel("Iterations per Stage (log scale)")
    ax2.set_title(
        "Iterations per Stage & Hessian Conditioning $\\kappa(\\nabla^2 \\phi_t)$"
    )
    ax2.grid(True, axis="y", linestyle="--", alpha=0.4)

    ax2_twin = ax2.twinx()
    conds = [s.hess_cond for s in res_newton.stages]
    ax2_twin.plot(
        x_idx,
        conds,
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
        linestyle=":",
        label="$\\kappa(\\nabla^2 \\phi_t)$",
    )
    ax2_twin.set_yscale("log")
    ax2_twin.set_ylabel(
        "Condition Number $\\kappa(\\nabla^2 \\phi_t)$", color="#7c3aed"
    )
    ax2_twin.tick_params(axis="y", labelcolor="#7c3aed")

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_efficient_frontier(
    mu_a: np.ndarray,
    S_a: np.ndarray,
    mu_b: np.ndarray,
    S_b: np.ndarray,
    tickers: List[str],
    res_plain: BarrierResult,
    res_climate: BarrierResult,
    out_path: str,
) -> None:
    """
    Generate Figure 3:
      Parametric Markowitz efficient frontier (gamma in [0.5, 50]) for Plain (A) and
      Climate (B), marking the base-case gamma=5 portfolios, Equal-Weight, and stocks.
    """
    front_a = compute_efficient_frontier(mu_a, S_a)
    front_b = compute_efficient_frontier(mu_b, S_b)

    fig, ax = plt.subplots(figsize=(9.5, 5.8))

    ax.plot(
        front_a["volatility"] * 100,
        front_a["expected_return"] * 100,
        color="#0284c7",
        linewidth=2.4,
        label="Efficient Frontier — Plain (A)",
    )
    ax.plot(
        front_b["volatility"] * 100,
        front_b["expected_return"] * 100,
        color="#ea580c",
        linewidth=2.2,
        linestyle="--",
        label="Efficient Frontier — Climate (B, Climate $\\mu,\\Sigma$)",
    )

    stock_vols = np.sqrt(np.diag(S_a)) * 100
    stock_rets = mu_a * 100
    ax.scatter(
        stock_vols,
        stock_rets,
        color="#94a3b8",
        s=35,
        alpha=0.8,
        label="20 Agro Candidate Stocks (Plain $\\mu,\\Sigma$)",
    )
    for i, t in enumerate(tickers):
        dx, dy = (4, 2)
        if t == "AGRO3":
            dx, dy = (-34, -2)
        elif t == "FHER3":
            dx, dy = (-34, 2)
        ax.annotate(
            t,
            (stock_vols[i], stock_rets[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.5,
            color="#475569",
        )

    w_eq = np.full(len(mu_a), 1.0 / len(mu_a))
    eq_vol = float(np.sqrt(w_eq @ S_a @ w_eq)) * 100
    eq_ret = float(mu_a @ w_eq) * 100
    ax.scatter(
        [eq_vol],
        [eq_ret],
        color="#16a34a",
        marker="D",
        s=90,
        zorder=5,
        label="Equal-Weight (20 Stocks)",
    )

    ax.scatter(
        [res_plain.volatility * 100],
        [res_plain.exp_return * 100],
        color="#0284c7",
        marker="*",
        s=220,
        edgecolor="black",
        zorder=6,
        label="Optimal Portfolio A ($\\gamma=5$)",
    )

    ax.scatter(
        [res_climate.volatility * 100],
        [res_climate.exp_return * 100],
        color="#ea580c",
        marker="*",
        s=220,
        edgecolor="black",
        zorder=6,
        label="Optimal Portfolio B ($\\gamma=5$, Climate $\\mu,\\Sigma$)",
    )

    ax.set_xlabel("Annualized Volatility $\\sqrt{w^\\top \\Sigma w}$ (%)")
    ax.set_ylabel("Annualized Expected Return $\\mu^\\top w$ (%)")
    ax.set_title(
        "Parametric Sensitivity in $\\gamma \\in [0.5, 50]$ (Efficient Frontier)"
    )
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_backtest_cumulative(
    bt_results: Dict[str, BacktestMetrics],
    out_path: str,
    title_suffix: str = "2025 Out-of-Sample Test Year",
) -> None:
    """
    Generate Figure 4:
      12-month buy-and-hold cumulative return chart comparing Plain (A), Climate (B),
      Agro ETF (AGRI11), and CDI.
    """
    fig, ax = plt.subplots(figsize=(10, 5.4))

    styles = [
        ("Plain (A)", "#0284c7", "-", 2.5, "o"),
        ("Climate (B)", "#ea580c", "-", 2.5, "o"),
        ("CDI", "#16a34a", "--", 2.0, None),
    ]
    plotted = set()
    for name, color, ls, lw, mk in styles:
        if name in bt_results:
            m = bt_results[name]
            cum_pct = (m.wealth_index - 1.0) * 100.0
            ax.plot(
                cum_pct.index,
                cum_pct.values,
                label=f"{name} ({m.total_return*100:+.1f}%)",
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker=mk,
                markersize=4 if mk else 0,
            )
            plotted.add(name)

    for name, m in bt_results.items():
        if name not in plotted:
            cum_pct = (m.wealth_index - 1.0) * 100.0
            ax.plot(
                cum_pct.index,
                cum_pct.values,
                label=f"{name} ({m.total_return*100:+.1f}%)",
                color="#9333ea",
                linestyle=":",
                linewidth=2.0,
                marker="s",
                markersize=4,
            )

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title(f"12-Month Buy-and-Hold Backtest ({title_suffix})")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def load_test_year_returns_and_cdi(
    returns_csv: str = "data/yf_monthly_returns_2024_2025.csv",
    excel_path: str = "dados_ceres.xlsx",
    test_start: str = "2025-01-01",
    test_end: str = "2025-12-01",
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load cached 2025 monthly decimal returns from `data/yf_monthly_returns_2024_2025.csv`
    and align with monthly decimal CDI from `dados_ceres.xlsx`.
    """
    rets = pd.read_csv(returns_csv, index_col=0, parse_dates=True)
    rets_test = rets[
        (rets.index >= pd.Timestamp(test_start))
        & (rets.index <= pd.Timestamp(test_end))
    ].copy()

    df_x = pd.read_excel(excel_path)
    df_x["dt"] = pd.to_datetime(df_x["date"], format="%m/%Y")
    cdi_map = dict(zip(df_x["dt"], df_x["cdi"].astype(float) / 100.0))

    cdi_vals = []
    last_valid_cdi = 0.0122
    for dt in rets_test.index:
        val = cdi_map.get(pd.Timestamp(dt), np.nan)
        if np.isnan(val) or val < 0.003:
            val = last_valid_cdi
        else:
            last_valid_cdi = float(val)
        cdi_vals.append(float(val))

    return rets_test, np.array(cdi_vals, dtype=float)


def write_summary_markdown(
    est: EstimationDataset,
    w_kkt: np.ndarray,
    nu_kkt: float,
    w_scipy: np.ndarray,
    res_newton: BarrierResult,
    res_bfgs: BarrierResult,
    res_grad: BarrierResult,
    res_climate: BarrierResult,
    bt_results: Dict[str, BacktestMetrics],
    out_md: str,
) -> None:
    """Write slide-ready Markdown tables to `outputs/summary_tables.md`."""
    lines: List[str] = []
    lines.append("# COS360 Results & Slide Tables: Markowitz NLP + Log-Barrier\n")

    lines.append(
        r"## 1. Solver Validation & KKT Conditions ($\gamma = 5$, 20 Stocks, Plain Moments)"
        + "\n"
    )
    lines.append(
        rf"- **KKT Closed-Form Budget Multiplier ($\nu$, shorting allowed):** `{nu_kkt:.6f}`"
    )
    lines.append(
        rf"- **KKT Closed-Form Min / Max Weight:** `{w_kkt.min()*100:.2f}%` / `{w_kkt.max()*100:.2f}%` "
        rf"({int((w_kkt < 0).sum())} of 20 stocks shorted in unconstrained solution)"
    )
    lines.append(
        rf"- **Log-Barrier Estimated Budget Multiplier ($\hat{{\nu}}$, $w \ge 0$):** `{res_newton.nu_dual:.6f}`"
    )
    lines.append(
        rf"- **Max Weight Discrepancy vs. SciPy SLSQP ($\|w_{{\text{{Newton}}}} - w_{{\text{{SLSQP}}}}\|_\infty$):** "
        rf"`{np.max(np.abs(res_newton.weights - w_scipy)):.2e}`"
    )
    lines.append(
        rf"- **KKT Stationarity Residual ($\|\gamma\Sigma w - \mu + \hat{{\nu}}\mathbf{{1}} - \hat{{\lambda}}\|_\infty$):** "
        rf"`{res_newton.stationarity_residual:.2e}`"
    )
    lines.append(
        rf"- **Complementary Slackness ($\max_i |\hat{{\lambda}}_i w_i| = 1/t$):** `{res_newton.comp_slackness_max:.2e}`"
        + "\n"
    )

    lines.append(
        r"## 2. Convergence & Conditioning Table Across Barrier Stages" + "\n"
    )
    lines.append(
        r"| Stage | Barrier $t$ | Duality Gap Bound $N/t$ | Hessian $\kappa(\nabla^2 \phi_t)$ | Newton Iters | BFGS Iters | Gradient Iters |"
    )
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for i in range(len(res_newton.stages)):
        sn = res_newton.stages[i]
        it_n = res_newton.stage_iters[i]
        it_b = res_bfgs.stage_iters[i]
        it_g = res_grad.stage_iters[i]
        g_str = f"{it_g:,d}" + (" (limit)" if it_g >= 20000 else "")
        lines.append(
            f"| {i+1} | $10^{i+1}$ | `{sn.duality_gap_bound:.2e}` | `{sn.hess_cond:,.1f}` | **{it_n}** | {it_b} | {g_str} |"
        )
    lines.append(
        f"| **Total** | — | — | — | **{sum(res_newton.stage_iters)}** | **{sum(res_bfgs.stage_iters)}** | **{sum(res_grad.stage_iters):,d}+** |\n"
    )

    lines.append(
        r"## 3. 20-Stock Universe: KKT Closed Form vs. Log-Barrier (Plain vs. Climate) & Duals $\hat{\lambda}_i$"
        + "\n"
    )
    mu_a, S_a = moments(est.R_monthly, est.p_plain, annualize=True)
    mu_b, S_b = moments(est.R_monthly, est.p_climate, annualize=True)
    lines.append(
        r"| Ticker | Plain $\mu_i$ (%) | Climate $\mu_i$ (%) | KKT Unc. $w_i^*$ (%) | Plain $w_i^*$ (%) | Dual $\hat{\lambda}_i$ (Plain) | Climate $w_i^*$ (%) |"
    )
    lines.append("|:---|---:|---:|---:|---:|---:|---:|")
    for i, t in enumerate(est.tickers):
        lines.append(
            f"| `{t}` | {mu_a[i]*100:6.2f} | {mu_b[i]*100:6.2f} | "
            f"{w_kkt[i]*100:7.2f} | {res_newton.weights[i]*100:6.2f} | "
            f"{res_newton.lambda_dual[i]:8.4f} | {res_climate.weights[i]*100:6.2f} |"
        )
    lines.append("")

    lines.append("## 4. 12-Month Buy-and-Hold Out-of-Sample Backtest (2025)\n")
    lines.append(
        "| Strategy / Benchmark | Total Return (%) | Annualized Volatility (%) | Max Drawdown (%) | Sharpe vs. CDI |"
    )
    lines.append("|:---|---:|---:|---:|---:|")
    for name, m in bt_results.items():
        lines.append(
            f"| **{name}** | `{m.total_return*100:+6.2f}%` | `{m.annualized_volatility*100:6.2f}%` | "
            f"`{m.max_drawdown*100:6.2f}%` | `{m.sharpe_vs_cdi:+5.2f}` |"
        )
    lines.append("")

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    os.makedirs("outputs", exist_ok=True)
    est = load_estimation_data()

    mu_a, S_a = moments(est.R_monthly, est.p_plain, annualize=True)
    mu_b, S_b = moments(est.R_monthly, est.p_climate, annualize=True)

    # 1. Closed-form KKT & SciPy reference
    w_kkt, nu_kkt = closed_form_kkt(mu_a, S_a, gamma=5.0)
    w_scipy = solve_scipy_reference(mu_a, S_a, gamma=5.0)

    # 2. Run all three unconstrained solvers inside the log-barrier method
    print("Running Log-Barrier with Newton, BFGS, and Gradient on 20-stock universe...")
    res_newton = barrier_method(mu_a, S_a, gamma=5.0, method="newton")
    res_bfgs = barrier_method(mu_a, S_a, gamma=5.0, method="bfgs")
    res_grad = barrier_method(mu_a, S_a, gamma=5.0, method="grad")
    res_climate = barrier_method(mu_b, S_b, gamma=5.0, method="newton")

    print(
        f"  Newton stage iterations  : {res_newton.stage_iters} (total={sum(res_newton.stage_iters)})"
    )
    print(
        f"  BFGS stage iterations    : {res_bfgs.stage_iters} (total={sum(res_bfgs.stage_iters)})"
    )
    print(
        f"  Gradient stage iterations: {res_grad.stage_iters} (total={sum(res_grad.stage_iters)})"
    )
    print(
        f"  Max |w_newton - w_scipy| : {np.max(np.abs(res_newton.weights - w_scipy)):.2e}"
    )
    print(
        f"  Active stocks (>0.5%)    : Plain={(res_newton.weights > 5e-3).sum()}, "
        f"Climate={(res_climate.weights > 5e-3).sum()}"
    )

    # 3. Backtest on 2025 test year using direct 20-stock optimal weights
    R_test, cdi_test = load_test_year_returns_and_cdi()
    bt_results = run_backtest_comparison(
        R_test=R_test,
        tickers_20=est.tickers,
        w_plain_20=res_newton.weights,
        w_climate_20=res_climate.weights,
        cdi_test=cdi_test,
    )

    # 4. Generate Figures & Markdown Tables
    plot_kkt_and_weights(
        est.tickers,
        w_kkt,
        res_newton,
        res_climate,
        "outputs/fig1_kkt_vs_barrier_weights.png",
    )
    plot_convergence_comparison(
        res_newton,
        res_bfgs,
        res_grad,
        "outputs/fig2_convergence_comparison.png",
    )
    plot_efficient_frontier(
        mu_a,
        S_a,
        mu_b,
        S_b,
        est.tickers,
        res_newton,
        res_climate,
        "outputs/fig3_efficient_frontier.png",
    )
    plot_backtest_cumulative(
        bt_results,
        "outputs/fig4_backtest_cumulative.png",
    )
    write_summary_markdown(
        est,
        w_kkt,
        nu_kkt,
        w_scipy,
        res_newton,
        res_bfgs,
        res_grad,
        res_climate,
        bt_results,
        "outputs/summary_tables.md",
    )
    print("All streamlined figures and summary_tables.md generated in outputs/!")


if __name__ == "__main__":
    main()
