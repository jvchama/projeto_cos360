"""
Markowitz Mean-Variance Optimization via Log-Barrier Method with
Gradient, Newton, and Quasi-Newton (BFGS) Unconstrained Solvers + Armijo Line Search.

Implements all mathematical formulations in §3, §4, and §9 of `project_plan.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


@dataclass
class StageDiagnostics:
    """Diagnostics recorded at a single barrier stage t."""

    t: float
    iterations: int
    f_barrier_final: float
    f_primal_final: float
    grad_norm_final: float
    hess_cond: float
    duality_gap_bound: float
    f_history: List[float] = field(default_factory=list)
    grad_norm_history: List[float] = field(default_factory=list)


@dataclass
class BarrierResult:
    """Full output of the log-barrier method for a given unconstrained solver."""

    method: str
    weights: np.ndarray  # (N,) optimal portfolio weights w*
    x_reduced: np.ndarray  # (N-1,) reduced primal variable x*
    primal_obj: float  # f(w*) = (gamma/2) w^T Sigma w - mu^T w
    exp_return: float  # mu^T w*
    volatility: float  # sqrt(w*^T Sigma w*)
    stage_iters: List[int]  # Iterations per stage t = 10^1 .. 10^6
    stages: List[StageDiagnostics]  # Per-stage diagnostics
    lambda_dual: np.ndarray  # (N,) KKT inequality multipliers lambda_i ≈ 1 / (t * w_i)
    nu_dual: float  # KKT budget multiplier nu
    stationarity_residual: float  # ||gamma*Sigma*w - mu + nu*1 - lambda||_inf
    comp_slackness_max: float  # max_i |lambda_i * w_i|


def closed_form_kkt(
    mu: np.ndarray, S: np.ndarray, gamma: float = 5.0
) -> Tuple[np.ndarray, float]:
    """
    Analytical KKT solution with only the budget constraint 1^T w = 1
    (short selling allowed, §4.1 & §9):
        w* = (1 / gamma) * Sigma^{-1} (mu - nu * 1)
        nu = (1^T Sigma^{-1} mu - gamma) / (1^T Sigma^{-1} 1)
    """
    N = len(mu)
    one = np.ones(N)
    Si_mu = np.linalg.solve(S, mu)
    Si_one = np.linalg.solve(S, one)
    nu = float((one @ Si_mu - gamma) / (one @ Si_one))
    w_star = (Si_mu - nu * Si_one) / gamma
    return w_star, nu


def make_barrier(
    mu: np.ndarray, S: np.ndarray, gamma: float, t: float
) -> Tuple[
    Callable[[np.ndarray], float],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]:
    """
    Construct the equality-eliminated log-barrier objective phi_t(x),
    gradient g(x), Hessian H(x), and affine reconstruction map W(x) = A x + e_N (§4.2 & §9).
    """
    N = len(mu)
    A = np.vstack([np.eye(N - 1), -np.ones((1, N - 1))])
    e = np.zeros(N)
    e[-1] = 1.0
    inv_t = 1.0 / t

    def W(x: np.ndarray) -> np.ndarray:
        return A @ x + e

    def f(x: np.ndarray) -> float:
        w = W(x)
        if np.any(w <= 0.0):
            return float("inf")
        return float(0.5 * gamma * (w @ S @ w) - (mu @ w) - inv_t * np.log(w).sum())

    def g(x: np.ndarray) -> np.ndarray:
        w = W(x)
        return A.T @ (gamma * (S @ w) - mu - inv_t / w)

    def H(x: np.ndarray) -> np.ndarray:
        w = W(x)
        return A.T @ (gamma * S + np.diag(inv_t / (w**2))) @ A

    return f, g, H, W


def armijo(
    f: Callable[[np.ndarray], float],
    x: np.ndarray,
    d: np.ndarray,
    gx: np.ndarray,
    W: Callable[[np.ndarray], np.ndarray],
    a: float = 1.0,
    beta: float = 0.5,
    c: float = 1e-4,
) -> float:
    """
    Backtracking Armijo line search with domain safeguard (§4.3 & §9):
    1. Halve `a` until W(x + a*d) > 0 strictly (stay inside log-barrier domain).
    2. Halve `a` until phi_t(x + a*d) <= phi_t(x) + c * a * (gx^T d).
    """
    while np.any(W(x + a * d) <= 0.0):
        a *= beta
        if a < 1e-16:
            return a

    fx = f(x)
    slope = float(gx @ d)
    while f(x + a * d) > fx + c * a * slope:
        a *= beta
        if a < 1e-15:
            break
    return a


def minimize_unc(
    f: Callable[[np.ndarray], float],
    g: Callable[[np.ndarray], np.ndarray],
    H: Callable[[np.ndarray], np.ndarray],
    W: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    method: str,
    tol: float = 1e-6,
    maxit: int = 20000,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """
    Unconstrained minimization of phi_t(x) via Gradient, Newton, or BFGS (§4.3 & §9).
    Returns (x_opt, f_hist, grad_norm_hist).
    """
    x = x.copy()
    n = len(x)
    B = np.eye(n)
    f_hist: List[float] = [f(x)]
    g_hist: List[float] = []

    for _ in range(maxit):
        gx = g(x)
        gnorm = float(np.linalg.norm(gx))
        g_hist.append(gnorm)
        if gnorm < tol:
            break

        if method == "grad":
            d = -gx
        elif method == "newton":
            d = -np.linalg.solve(H(x), gx)
        elif method == "bfgs":
            d = -B @ gx
            if float(gx @ d) >= 0.0:
                B = np.eye(n)
                d = -gx
        else:
            raise ValueError(f"Unknown method '{method}'. Choose from 'grad', 'newton', 'bfgs'.")

        a = armijo(f, x, d, gx, W)
        s = a * d
        xn = x + s

        if method == "bfgs":
            y = g(xn) - gx
            sy = float(s @ y)
            if sy > 1e-12:
                r = 1.0 / sy
                I = np.eye(n)
                B = (I - r * np.outer(s, y)) @ B @ (I - r * np.outer(y, s)) + r * np.outer(s, s)

        x = xn
        f_hist.append(f(x))

    if len(g_hist) < len(f_hist):
        g_hist.append(float(np.linalg.norm(g(x))))

    return x, f_hist, g_hist


def barrier_method(
    mu: np.ndarray,
    S: np.ndarray,
    gamma: float = 5.0,
    method: str = "newton",
    t0: float = 10.0,
    mult: float = 10.0,
    stages: int = 6,
    tol: float = 1e-6,
    maxit: int = 20000,
) -> BarrierResult:
    """
    Solve the constrained Markowitz NLP via the Log-Barrier central path (§4.2 & §9):
        min_w (gamma/2) w^T S w - mu^T w   s.t. 1^T w = 1, w >= 0
    """
    N = len(mu)
    x = np.full(N - 1, 1.0 / N)
    t = t0
    stage_iters: List[int] = []
    stage_diags: List[StageDiagnostics] = []

    for _ in range(stages):
        f, g, H, W = make_barrier(mu, S, gamma, t)
        x, f_hist, g_hist = minimize_unc(f, g, H, W, x, method, tol=tol, maxit=maxit)
        n_it = len(f_hist) - 1
        stage_iters.append(n_it)

        w_curr = W(x)
        primal_val = float(0.5 * gamma * (w_curr @ S @ w_curr) - (mu @ w_curr))
        cond_val = float(np.linalg.cond(H(x)))
        stage_diags.append(
            StageDiagnostics(
                t=t,
                iterations=n_it,
                f_barrier_final=f_hist[-1],
                f_primal_final=primal_val,
                grad_norm_final=g_hist[-1],
                hess_cond=cond_val,
                duality_gap_bound=N / t,
                f_history=f_hist,
                grad_norm_history=g_hist,
            )
        )
        t *= mult

    t_final = t0 * (mult ** (stages - 1))
    _, _, _, W_final = make_barrier(mu, S, gamma, t_final)
    w = W_final(x)

    # Free KKT dual multiplier estimates (§4.2 item 5): lambda_i ≈ 1 / (t * w_i)
    lambda_dual = 1.0 / (t_final * w)
    # From stationarity gamma*S*w - mu + nu*1 - lambda = 0, eliminated against e_N:
    # the last coordinate gives exact nu on the central path:
    grad_f = gamma * (S @ w) - mu
    nu_dual = float(lambda_dual[-1] - grad_f[-1])
    stat_res = float(np.max(np.abs(grad_f + nu_dual * np.ones(N) - lambda_dual)))
    comp_slack = float(np.max(np.abs(lambda_dual * w)))

    primal_obj = float(0.5 * gamma * (w @ S @ w) - (mu @ w))
    exp_ret = float(mu @ w)
    vol = float(np.sqrt(max(w @ S @ w, 0.0)))

    return BarrierResult(
        method=method,
        weights=w,
        x_reduced=x,
        primal_obj=primal_obj,
        exp_return=exp_ret,
        volatility=vol,
        stage_iters=stage_iters,
        stages=stage_diags,
        lambda_dual=lambda_dual,
        nu_dual=nu_dual,
        stationarity_residual=stat_res,
        comp_slackness_max=comp_slack,
    )


def solve_scipy_reference(
    mu: np.ndarray, S: np.ndarray, gamma: float = 5.0
) -> np.ndarray:
    """Solve the Markowitz NLP using `scipy.optimize.minimize` (SLSQP) for validation."""
    N = len(mu)
    res = minimize(
        fun=lambda w: 0.5 * gamma * (w @ S @ w) - (mu @ w),
        x0=np.full(N, 1.0 / N),
        jac=lambda w: gamma * (S @ w) - mu,
        bounds=[(0.0, 1.0)] * N,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return res.x


@dataclass
class TwoStageTop10Result:
    """Output of the two-stage Top-10 stock selection heuristic (§3.3 & §9)."""

    stage1_result: BarrierResult
    selected_indices: np.ndarray  # (10,) indices in the 20-stock universe
    selected_tickers: List[str]  # 10 ticker symbols
    stage2_result: BarrierResult  # Re-solved BarrierResult on the 10 selected stocks
    full_weights_20: np.ndarray  # (20,) stage-2 weights embedded back into R^20


def two_stage_top10(
    mu: np.ndarray,
    S: np.ndarray,
    tickers: List[str],
    gamma: float = 5.0,
    method: str = "newton",
    k: int = 10,
) -> TwoStageTop10Result:
    """
    Two-stage cardinality heuristic (§3.3 & §9):
    1. Solve with all ~20 candidates.
    2. Keep the 10 largest weights.
    3. Re-solve using only those 10.
    """
    res_all = barrier_method(mu, S, gamma=gamma, method=method)
    idx = np.argsort(res_all.weights)[::-1][:k]
    selected_tickers = [tickers[i] for i in idx]

    mu_10 = mu[idx]
    S_10 = S[np.ix_(idx, idx)]
    res_10 = barrier_method(mu_10, S_10, gamma=gamma, method=method)

    w_full = np.zeros(len(mu))
    w_full[idx] = res_10.weights

    return TwoStageTop10Result(
        stage1_result=res_all,
        selected_indices=idx,
        selected_tickers=selected_tickers,
        stage2_result=res_10,
        full_weights_20=w_full,
    )


def compute_efficient_frontier(
    mu: np.ndarray,
    S: np.ndarray,
    gamma_grid: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Sweep risk aversion gamma across a grid (default 0.5 -> 50) using the log-barrier
    Newton solver and return annualized volatility and expected return arrays (§4.5).
    """
    if gamma_grid is None:
        gamma_grid = np.geomspace(0.5, 50.0, 45)

    vols = np.zeros(len(gamma_grid))
    rets = np.zeros(len(gamma_grid))
    weights_list = []

    for i, g_val in enumerate(gamma_grid):
        res = barrier_method(mu, S, gamma=float(g_val), method="newton")
        vols[i] = res.volatility
        rets[i] = res.exp_return
        weights_list.append(res.weights)

    return {
        "gamma": gamma_grid,
        "volatility": vols,
        "expected_return": rets,
        "weights": np.array(weights_list),
    }
