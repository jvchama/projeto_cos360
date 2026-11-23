# COS360 Results & Slide Tables: Markowitz NLP + Log-Barrier

## 1. Solver Validation & KKT Conditions ($\gamma = 5$, 20 Stocks, Plain Moments)

- **KKT Closed-Form Budget Multiplier ($\nu$, shorting allowed):** `0.060677`
- **KKT Closed-Form Min / Max Weight:** `-23.29%` / `40.66%` (8 of 20 stocks shorted in unconstrained solution)
- **Log-Barrier Estimated Budget Multiplier ($\hat{\nu}$, $w \ge 0$):** `0.082528`
- **Max Weight Discrepancy vs. SciPy SLSQP ($\|w_{\text{Newton}} - w_{\text{SLSQP}}\|_\infty$):** `2.25e-04`
- **KKT Stationarity Residual ($\|\gamma\Sigma w - \mu + \hat{\nu}\mathbf{1} - \hat{\lambda}\|_\infty$):** `2.04e-09`
- **Complementary Slackness ($\max_i |\hat{\lambda}_i w_i| = 1/t$):** `1.00e-06`

## 2. Convergence & Conditioning Table Across Barrier Stages

| Stage | Barrier $t$ | Duality Gap Bound $N/t$ | Hessian $\kappa(\nabla^2 \phi_t)$ | Newton Iters | BFGS Iters | Gradient Iters |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | $10^1$ | `2.00e+00` | `22.8` | **3** | 73 | 125 |
| 2 | $10^2$ | `2.00e-01` | `70.0` | **5** | 28 | 314 |
| 3 | $10^3$ | `2.00e-02` | `130.5` | **6** | 44 | 696 |
| 4 | $10^4$ | `2.00e-03` | `1,103.3` | **7** | 35 | 4,307 |
| 5 | $10^5$ | `2.00e-04` | `11,741.8` | **7** | 55 | 20,000 (limit) |
| 6 | $10^6$ | `2.00e-05` | `118,397.9` | **8** | 50 | 20,000 (limit) |
| **Total** | — | — | — | **36** | **285** | **45,442+** |

## 3. 20-Stock Universe: KKT Closed Form vs. Log-Barrier (Plain vs. Climate) & Duals $\hat{\lambda}_i$

| Ticker | Plain $\mu_i$ (%) | Climate $\mu_i$ (%) | KKT Unc. $w_i^*$ (%) | Plain $w_i^*$ (%) | Dual $\hat{\lambda}_i$ (Plain) | Climate $w_i^*$ (%) |
|:---|---:|---:|---:|---:|---:|---:|
| `AGRO3` |  21.32 |  18.19 |   12.94 |   3.26 |   0.0000 |   0.00 |
| `BEEF3` |   4.85 |  -4.92 |  -23.29 |   0.00 |   0.1778 |   0.00 |
| `BRFS3` |   3.29 |  -9.12 |  -16.64 |   0.00 |   0.1916 |   0.00 |
| `CGRA4` |  15.37 |  16.35 |    5.82 |   0.02 |   0.0044 |   0.00 |
| `CSAN3` |  10.16 |   3.84 |  -18.35 |   0.00 |   0.1179 |   0.00 |
| `DXCO3` |  11.12 |   0.84 |  -22.50 |   0.00 |   0.0977 |   0.00 |
| `EUCA4` |  26.73 |  27.64 |   26.80 |  10.31 |   0.0000 |   0.01 |
| `FHER3` |  32.80 |  71.96 |   -2.96 |   0.00 |   0.0554 |   3.18 |
| `KEPL3` |  18.16 |  17.86 |    0.07 |   0.00 |   0.0640 |   0.00 |
| `MDIA3` |   4.04 |  -7.21 |  -10.83 |   0.00 |   0.1239 |   0.00 |
| `MRFG3` |  26.73 |  27.61 |   26.87 |  11.11 |   0.0000 |  14.25 |
| `MTSA4` |  21.44 |  27.75 |   12.04 |   5.73 |   0.0000 |  25.62 |
| `PTNT4` |  23.16 |  17.07 |   36.46 |  21.34 |   0.0000 |   6.52 |
| `RANI3` |  18.99 |   2.36 |   -8.62 |   0.00 |   0.0353 |   0.00 |
| `RAPT4` |  21.27 |  13.99 |   27.48 |   0.01 |   0.0139 |   0.00 |
| `SLCE3` |  27.58 |  37.42 |   40.66 |  31.74 |   0.0000 |  49.33 |
| `SMTO3` |  15.62 |  12.49 |    4.17 |   0.00 |   0.0486 |   0.00 |
| `SUZB3` |  18.31 |  13.81 |   12.83 |   7.90 |   0.0000 |   0.00 |
| `TUPY3` |  13.03 |  18.06 |  -17.04 |   0.00 |   0.0966 |   0.00 |
| `WLMM4` |  31.32 |  15.37 |   14.10 |   8.57 |   0.0000 |   1.07 |

## 4. 12-Month Buy-and-Hold Out-of-Sample Backtest (2025)

| Strategy / Benchmark | Total Return (%) | Annualized Volatility (%) | Max Drawdown (%) | Sharpe vs. CDI |
|:---|---:|---:|---:|---:|
| **Plain (A)** | ` +0.32%` | ` 11.39%` | `-11.68%` | `-1.11` |
| **Climate (B)** | ` +9.08%` | ` 11.80%` | `-10.56%` | `-0.36` |
| **Agro ETF (AGRI11)** | ` -4.44%` | ` 15.53%` | `-10.75%` | `-1.10` |
| **CDI** | `+14.46%` | `  0.37%` | `  0.00%` | `+0.00` |
