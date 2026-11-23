"""
Data loading, universe filtering, ENSO climate regime weighting, moment estimation,
and Yahoo Finance test-year data caching for the COS360 Agro Portfolio project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Universe Definitions (§2 of project_plan.md)
# ---------------------------------------------------------------------------

# 8 tickers explicitly listed for removal in §2 (illiquid + non-agro)
DROPPED_EXPLICIT: List[str] = [
    "CTNM4",
    "BAUH4",
    "JOPA4",
    "DOHL4",
    "CEDO4",
    "MNPR3",
    "ABEV3",
    "PCAR3",
]

# 7 additional illiquid/microcap/delisted textile tickers removed to reach 20 liquid stocks
DROPPED_ILLIQUID: List[str] = [
    "MSPA4",  # 82 zero-return months in 2015-2024
    "LUXM4",  # 38 zero-return months in 2015-2024
    "SNSY5",  # 12 zero-return months, distressed microcap
    "CTSA4",  # 8 zero-return months, textile penny stock
    "TXRX4",  # 6 zero-return months, textile microcap
    "SGPS3",  # Delisted / missing after 08/2024
    "CTKA4",  # Non-agro textile microcap with extreme single-month outlier
]

# Final 20 candidate B3 agro-related stocks
TICKERS_20: List[str] = [
    "AGRO3",  # BrasilAgro
    "BEEF3",  # Minerva Foods
    "BRFS3",  # BRF S.A.
    "CGRA4",  # Grazziotin
    "CSAN3",  # Cosan
    "DXCO3",  # Dexco (Forestry / Wood Panels)
    "EUCA4",  # Eucatex (Forestry / Wood Panels)
    "FHER3",  # Fertilizantes Heringer
    "KEPL3",  # Kepler Weber (Grain Storage)
    "MDIA3",  # M. Dias Branco (Wheat / Milling)
    "MRFG3",  # Marfrig
    "MTSA4",  # Metisa (Agricultural Implements)
    "PTNT4",  # Pettenati
    "RANI3",  # Irani (Forestry / Packaging)
    "RAPT4",  # Randoncorp (Grain Trailers / Agro Logistics)
    "SLCE3",  # SLC Agricola
    "SMTO3",  # Sao Martinho (Sugar & Ethanol)
    "SUZB3",  # Suzano (Pulp & Forestry)
    "TUPY3",  # Tupy (Agro & Heavy Castings / Irrigation)
    "WLMM4",  # WLM Participacoes (Agro Machinery Dealerships)
]


# All 11 climate variables in dados_ceres.xlsx:
# 5 regional temperature anomalies, 5 regional rainfall anomalies, and ONI
CLIMATE_COLS: List[str] = [
    "temp_N",
    "temp_NE",
    "temp_CO",
    "temp_SE",
    "temp_S",
    "pluv_N",
    "pluv_NE",
    "pluv_CO",
    "pluv_SE",
    "pluv_S",
    "nio_index",
]


@dataclass
class EstimationDataset:
    """Container for the 2015-2024 estimation window and scenario weights."""

    dates: pd.DatetimeIndex
    tickers: List[str]
    R_monthly: np.ndarray  # (T, N) decimal monthly returns
    oni: np.ndarray  # (T,) contemporaneous ONI
    oni_lag8: np.ndarray  # (T,) 8-month lagged ONI
    regimes_lag8: List[str]  # (T,) 'El Nino', 'Neutral', or 'La Nina'
    decision_regime: str  # Regime at the decision date (12/2024)
    climate_lag8: np.ndarray  # (T, 11) 8-month lagged climate anomaly vector
    p_plain: np.ndarray  # (T,) uniform weights 1/T
    p_climate: np.ndarray  # (T,) 11-feature Gaussian kernel climate weights
    p_enso: np.ndarray  # (T,) ENSO-only 0.70 / 0.30 weights (for reference)
    cdi_monthly: np.ndarray  # (T,) decimal monthly CDI


def classify_enso_regime(oni_val: float) -> str:
    """Classify NOAA Oceanic Niño Index (ONI) into El Niño, La Niña, or Neutral."""
    if pd.isna(oni_val):
        return "Neutral"
    if oni_val >= 0.5:
        return "El Nino"
    if oni_val <= -0.5:
        return "La Nina"
    return "Neutral"


def compute_climate_weights(
    regimes: List[str], target_regime: str, same_weight: float = 0.70
) -> np.ndarray:
    """
    Construct ENSO-only scenario probability weights p_t:
    Months in the same ENSO regime as `target_regime` share `same_weight` (0.70);
    all other months share `1 - same_weight` (0.30).
    """
    mask = np.array([r == target_regime for r in regimes], dtype=bool)
    n_same = int(mask.sum())
    n_diff = int((~mask).sum())
    if n_same == 0 or n_diff == 0:
        return np.full(len(regimes), 1.0 / len(regimes))
    p = np.where(mask, same_weight / n_same, (1.0 - same_weight) / n_diff)
    return p / p.sum()


def compute_climate_kernel_weights(
    climate_matrix: np.ndarray, bandwidth: float = 2.0
) -> np.ndarray:
    """
    Construct Gaussian-kernel scenario weights p_t over all 11 standardized climate
    variables (regional temperature + rainfall anomalies + ONI, lagged 8 months):
        p_t propto exp(-||z_t - z_T||^2 / (2 * bandwidth^2))
    where z_T is the climate state at the decision date.
    """
    C = np.asarray(climate_matrix, dtype=float)
    std = np.where(C.std(axis=0) > 1e-12, C.std(axis=0), 1.0)
    Z = (C - C.mean(axis=0)) / std
    z_now = Z[-1]
    d2 = np.sum((Z - z_now) ** 2, axis=1)
    p = np.exp(-d2 / (2.0 * (bandwidth**2)))
    return p / p.sum()


def moments(
    R: np.ndarray, p: np.ndarray, annualize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scenario-weighted mean vector mu and covariance matrix Sigma (§3.1 & §9).

    Parameters
    ----------
    R : (T, N) array of decimal monthly returns
    p : (T,) array of scenario weights summing to 1
    annualize : bool
        If True, multiply monthly mean and covariance by 12 so mu is annualized
        expected return and sqrt(w^T Sigma w) is annualized volatility.
    """
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    mu = p @ R
    X = R - mu
    Sigma = (X * p[:, None]).T @ X
    if annualize:
        mu = mu * 12.0
        Sigma = Sigma * 12.0
    # Ensure exact symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)
    return mu, Sigma


def load_estimation_data(
    excel_path: str = "dados_ceres.xlsx",
    start_date: str = "2015-01-01",
    end_date: str = "2024-12-01",
    tickers: Optional[List[str]] = None,
    use_lag8_for_decision: bool = True,
    climate_bandwidth: float = 2.0,
) -> EstimationDataset:
    """
    Load `dados_ceres.xlsx`, compute 8-month lagged climate variables (regional
    temp/rainfall anomalies + ONI) prior to slicing the 2015-2024 estimation window
    (120 months), and return clean matrices and scenario weights.
    """
    if tickers is None:
        tickers = list(TICKERS_20)

    df = pd.read_excel(excel_path)
    df["dt"] = pd.to_datetime(df["date"], format="%m/%Y")
    df = df.sort_values("dt").reset_index(drop=True)

    # Lag all 11 climate variables by 8 months before slicing 2015-2024
    lag_clim_cols: List[str] = []
    for c in CLIMATE_COLS:
        lag_name = f"{c}_lag8"
        df[lag_name] = df[c].shift(8)
        lag_clim_cols.append(lag_name)

    df["oni_lag8"] = df["nio_index_lag8"]
    df["regime_lag8"] = df["oni_lag8"].apply(classify_enso_regime)
    df["regime_unlag"] = df["nio_index"].apply(classify_enso_regime)

    mask = (df["dt"] >= pd.Timestamp(start_date)) & (
        df["dt"] <= pd.Timestamp(end_date)
    )
    sub = df.loc[mask].copy().reset_index(drop=True)

    # Extract decimal monthly returns (raw file values are in percentage points)
    R_monthly = sub[tickers].astype(float).values / 100.0
    cdi_monthly = sub["cdi"].astype(float).values / 100.0
    climate_lag8 = sub[lag_clim_cols].astype(float).values

    T = len(sub)
    p_plain = np.full(T, 1.0 / T)

    regimes_lag8 = sub["regime_lag8"].tolist()
    if use_lag8_for_decision:
        decision_regime = regimes_lag8[-1]
    else:
        decision_regime = sub["regime_unlag"].iloc[-1]

    p_enso = compute_climate_weights(regimes_lag8, decision_regime, same_weight=0.70)
    p_climate = compute_climate_kernel_weights(climate_lag8, bandwidth=climate_bandwidth)

    return EstimationDataset(
        dates=pd.DatetimeIndex(sub["dt"]),
        tickers=tickers,
        R_monthly=R_monthly,
        oni=sub["nio_index"].astype(float).values,
        oni_lag8=sub["oni_lag8"].astype(float).values,
        regimes_lag8=regimes_lag8,
        decision_regime=decision_regime,
        climate_lag8=climate_lag8,
        p_plain=p_plain,
        p_climate=p_climate,
        p_enso=p_enso,
        cdi_monthly=cdi_monthly,
    )


def _fetch_yahoo_daily_adjclose(
    symbol: str, start_ts: int, end_ts: int
) -> pd.Series:
    """
    Fetch daily adjusted close series from Yahoo Finance v8 chart API using system `curl`
    (HTTP/2) with StatusInvest fallback for B3 tickers if rate-limited.
    """
    import json
    import subprocess
    import urllib.parse

    encoded = urllib.parse.quote(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d&events=div,splits"
    )
    ua = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    res = subprocess.run(
        ["curl", "-sL", "--max-time", "15", "-A", ua, url],
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode == 0 and res.stdout.strip().startswith("{"):
        payload = json.loads(res.stdout)
        chart_res = payload.get("chart", {}).get("result")
        if chart_res:
            result = chart_res[0]
            timestamps = result.get("timestamp", [])
            indicators = result.get("indicators", {})
            adjclose_list = indicators.get("adjclose", [{}])[0].get("adjclose")
            if not adjclose_list:
                adjclose_list = indicators.get("quote", [{}])[0].get("close", [])
            dates = (
                pd.to_datetime(timestamps, unit="s", utc=True)
                .tz_convert(None)
                .normalize()
            )
            s = pd.Series(adjclose_list, index=dates, name=symbol, dtype=float)
            return s[~s.index.duplicated(keep="last")].sort_index()

    # StatusInvest public 5-year price endpoint (type=4) for B3 stocks/ETFs/indices
    clean_ticker = "ibovespa" if symbol == "^BVSP" else symbol.replace(".SA", "")
    for endpoint in ["acao", "etf", "fundos-de-investimento", "indice"]:
        si_url = f"https://statusinvest.com.br/{endpoint}/tickerprice?ticker={clean_ticker}&type=4&currences%5B%5D=1"
        si_res = subprocess.run(
            ["curl", "-sL", "--max-time", "15", "-A", ua, si_url],
            capture_output=True,
            text=True,
            check=False,
        )
        if si_res.returncode == 0 and si_res.stdout.strip().startswith("["):
            data = json.loads(si_res.stdout)
            if data and isinstance(data, list) and "prices" in data[0]:
                prices = data[0]["prices"]
                if prices:
                    df_p = pd.DataFrame(prices)
                    raw_dates = df_p["date"].astype(str).str.slice(0, 8)
                    df_p["dt"] = pd.to_datetime(
                        raw_dates, format="%d/%m/%y", errors="coerce"
                    )
                    df_p = df_p.dropna(subset=["dt", "price"]).sort_values("dt")
                    s = pd.Series(
                        df_p["price"].astype(float).values,
                        index=df_p["dt"].dt.normalize(),
                        name=symbol,
                    )
                    s = s[
                        (s.index >= pd.Timestamp("2023-12-01"))
                        & (s.index <= pd.Timestamp("2026-01-05"))
                    ]
                    return s[~s.index.duplicated(keep="last")].sort_index()

    raise RuntimeError(f"Could not fetch {symbol} from Yahoo or StatusInvest")


def download_and_cache_test_data(
    tickers: Optional[List[str]] = None,
    prices_csv: str = "data/yf_monthly_prices_2024_2025.csv",
    returns_csv: str = "data/yf_monthly_returns_2024_2025.csv",
    start: str = "2024-01-01",
    end: str = "2026-01-05",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download monthly adjusted close prices from Yahoo Finance / StatusInvest for the
    20 agro stocks, IBOV (^BVSP), and Agro ETFs (AGRI11.SA, BBGO11.SA, BOVA11.SA),
    and save both monthly prices and monthly decimal returns to `data/`.
    """
    import time

    if tickers is None:
        tickers = list(TICKERS_20)

    yf_map: Dict[str, str] = {t: f"{t}.SA" for t in tickers}
    benchmarks = {
        "IBOV": "^BVSP",
        "AGRI11": "AGRI11.SA",
        "BBGO11": "BBGO11.SA",
        "BOVA11": "BOVA11.SA",
        "MBRF3": "MBRF3.SA",
    }
    all_map = {**yf_map, **benchmarks}

    start_ts = int(pd.Timestamp(start, tz="UTC").timestamp())
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp())

    series_dict: Dict[str, pd.Series] = {}
    for name, sym in all_map.items():
        try:
            s = _fetch_yahoo_daily_adjclose(sym, start_ts, end_ts)
            series_dict[name] = s
            time.sleep(0.1)
        except Exception as exc:
            print(f"Warning: could not fetch {name} ({sym}): {exc}")

    if not series_dict:
        raise RuntimeError("No Yahoo Finance series could be downloaded.")

    close = pd.DataFrame(series_dict)
    close.index = pd.to_datetime(close.index)

    # Resample daily adjusted close to month-end last available trading price
    monthly_prices = close.resample("ME").last()
    monthly_prices.index = monthly_prices.index.to_period("M").to_timestamp()
    monthly_prices.index.name = "date"

    # Handle BRFS3 / MRFG3 merger into MBRF3 in 2025
    if "MBRF3" in monthly_prices.columns:
        for merged_t in ["BRFS3", "MRFG3"]:
            if merged_t not in monthly_prices.columns:
                monthly_prices[merged_t] = monthly_prices["MBRF3"]
            else:
                monthly_prices[merged_t] = monthly_prices[merged_t].fillna(
                    monthly_prices["MBRF3"]
                )

    # Compute monthly returns (decimal)
    monthly_returns = monthly_prices.pct_change(fill_method=None)

    # Ensure column order keeps all 20 tickers first
    ordered_cols = [c for c in tickers if c in monthly_prices.columns] + [
        c for c in monthly_prices.columns if c not in tickers
    ]
    monthly_prices = monthly_prices[ordered_cols]
    monthly_returns = monthly_returns[ordered_cols]

    os.makedirs(os.path.dirname(prices_csv), exist_ok=True)
    monthly_prices.to_csv(prices_csv)
    monthly_returns.to_csv(returns_csv)
    return monthly_prices, monthly_returns


if __name__ == "__main__":
    est = load_estimation_data()
    print(
        f"Loaded estimation dataset: T={len(est.dates)} months "
        f"({est.dates[0].strftime('%Y-%m')} to {est.dates[-1].strftime('%Y-%m')}), "
        f"N={len(est.tickers)} stocks."
    )
    print(f"Decision-date (12/2024) 8m-lagged ENSO regime: {est.decision_regime}")
    prices, rets = download_and_cache_test_data()
    print("Downloaded monthly prices shape:", prices.shape)
    print("Downloaded monthly returns shape:", rets.shape)
    print("2025 non-null return counts per column:")
    rets_2025 = rets[(rets.index >= "2025-01-01") & (rets.index <= "2025-12-01")]
    print(rets_2025.notna().sum())
