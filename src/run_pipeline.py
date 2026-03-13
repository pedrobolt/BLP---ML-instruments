"""
run_pipeline.py
Pipeline completo: dados sintéticos → candidatos → seleção ML → diagnóstico → BLP

Uso:
    python src/run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from simulate_data import simulate_blp_data
from instruments import (
    build_instrument_candidates,
    select_instruments_lasso,
    select_instruments_rf,
    select_instruments_combined,
)
from diagnostics import first_stage, sargan_hansen_test, plot_first_stage, plot_instrument_importance

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CHAR_COLS = ["x1", "x2", "x3"]
SELECTION_METHOD = "combined"   # "lasso" | "rf" | "combined"
SAVE_PLOTS = True

os.makedirs("data/raw", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Dados
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 1 — Geração de dados sintéticos")
print("=" * 60)

df = simulate_blp_data(T=50, J=10, seed=42)
df.to_csv("data/raw/simulated_markets.csv", index=False)
print(f"Dados gerados: {df.shape[0]} observações, {df['market_id'].nunique()} mercados")
print(df[["market_id", "product_id", "price", "shares", *CHAR_COLS]].head(5).to_string(index=False))

# ─────────────────────────────────────────────
# 2. Candidatos a instrumento
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 2 — Construção dos candidatos")
print("=" * 60)

candidates = build_instrument_candidates(df, CHAR_COLS)
print(f"Candidatos gerados: {candidates.shape[1]} variáveis")
print(f"Colunas: {list(candidates.columns)}")

price  = df["price"]
X_ctrl = df[CHAR_COLS]

# ─────────────────────────────────────────────
# 3. Seleção ML
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"ETAPA 3 — Seleção de instrumentos: {SELECTION_METHOD.upper()}")
print("=" * 60)

if SELECTION_METHOD == "lasso":
    Z_selected, _ = select_instruments_lasso(candidates, price, X_ctrl)

elif SELECTION_METHOD == "rf":
    Z_selected, importances = select_instruments_rf(candidates, price)
    if SAVE_PLOTS:
        plot_instrument_importance(importances, save_path="outputs/rf_importance.png")

else:  # combined
    Z_selected = select_instruments_combined(candidates, price, X_ctrl)
    _, importances = select_instruments_rf(candidates, price, verbose=False)
    if SAVE_PLOTS:
        plot_instrument_importance(importances, save_path="outputs/rf_importance.png")

print(f"\nInstrumentos finais ({Z_selected.shape[1]}): {list(Z_selected.columns)}")

# ─────────────────────────────────────────────
# 4. Diagnóstico
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 4 — Diagnóstico dos instrumentos")
print("=" * 60)

fs_results = first_stage(price, Z_selected, X_ctrl)

if SAVE_PLOTS:
    plot_first_stage(price, Z_selected, X_ctrl, save_path="outputs/first_stage.png")

sargan_results = sargan_hansen_test(
    price=price,
    outcome=np.log(df["shares"] / df["outside_share"]),
    Z_selected=Z_selected,
    X_controls=X_ctrl,
)

# ─────────────────────────────────────────────
# 5. Estimação BLP via PyBLP
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 5 — Estimação BLP (pyblp)")
print("=" * 60)

try:
    import pyblp

    # PyBLP exige nomes específicos de coluna (tudo no plural)
    df_blp = df.copy()
    df_blp["market_ids"] = df_blp["market_id"]
    df_blp["firm_ids"]   = df_blp["firm_id"]
    df_blp["prices"]     = df_blp["price"]

    # Instrumentos: PyBLP lê demand_instruments0, demand_instruments1, ...
    for i, col in enumerate(Z_selected.columns):
        df_blp[f"demand_instruments{i}"] = Z_selected[col].values

    # X1: utilidade linear (deve incluir prices)
    # X2: características com heterogeneidade aleatória (sigma)
    X1_form = pyblp.Formulation("1 + x1 + x2 + x3 + prices")
    X2_form = pyblp.Formulation("0 + x1 + x2 + x3")

    problem = pyblp.Problem(
        product_formulations=(X1_form, X2_form),
        product_data=df_blp,
        integration=pyblp.Integration("monte_carlo", size=50, specification_options={"seed": 42}),
    )

    print(problem)

    results = problem.solve(
        sigma=np.diag([0.5, 0.5, 0.5]),
        optimization=pyblp.Optimization("bfgs"),
        iteration=pyblp.Iteration("squarem"),
    )

    print(results)

    print("\nElasticidades — mercado 0 (primeiros 5 produtos):")
    elasticities = results.compute_elasticities()
    print(pd.DataFrame(elasticities[0]).round(3).iloc[:5, :5].to_string())

except ImportError:
    print("[INFO] pyblp não instalado: pip install pyblp")

except Exception as e:
    print(f"[ERRO PyBLP] {type(e).__name__}: {e}")

# ─────────────────────────────────────────────
# 6. Sumário
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("SUMÁRIO")
print("=" * 60)
print(f"Candidatos gerados        : {candidates.shape[1]}")
print(f"Instrumentos selecionados : {Z_selected.shape[1]}")
print(f"F-stat first stage        : {fs_results['f_stat']:.2f}")
print(f"R² parcial                : {fs_results['partial_r2']:.4f}")
print(f"J-stat (Sargan)           : {sargan_results['j_stat']:.4f}")
print(f"p-valor Sargan            : {sargan_results['j_pval']:.4f}")

if SAVE_PLOTS:
    print("\nGráficos salvos em: outputs/")
