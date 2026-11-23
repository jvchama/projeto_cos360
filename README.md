# Otimização Convexa de Portfólio via Método de Barreira Logarítmica

Trabalho da disciplina **COS360 — Otimização (UFRJ)** aplicado à cadeia ampliada do agronegócio brasileiro (B3), comparando duas instâncias de estimação de $(\mu, \Sigma)$ na janela jan/2015–dez/2024 ($T = 120$ meses, $N = 20$ ativos):
- **Base:** pesos uniformes ($p_t = 1/T$).
- **Climática:** pesos via kernel Gaussiano 11D sobre anomalias regionais de temperatura, pluviosidade e índice ONI (defasagem de 8 meses).

## Estrutura
- `cos360_portfolio_optimization.ipynb`: notebook principal com formulação, derivadas analíticas, caminho central (Newton, BFGS e Gradiente), validação KKT/SLSQP e backtest 2025.
- `src/`: módulos auxiliares (`data_loader.py`, `barrier_solvers.py`, `backtest.py`, `run_experiments.py`).
- `dados_ceres.xlsx` e `data/`: dados históricos (2015–2024) e cache de retornos de 2025.
- `apresentacao_cos360.tex` e `outputs/`: slides Beamer e figuras geradas.

## Como executar
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Em seguida, abra e execute `cos360_portfolio_optimization.ipynb`.
