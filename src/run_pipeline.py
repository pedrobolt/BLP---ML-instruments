"""
run_pipeline.py
Pipeline completo: CSV de smartphones → candidatos → seleção ML → diagnóstico → BLP

Uso:
    python src/run_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd

from instruments import (
    remove_collinear_instruments,
    build_instrument_candidates,
    select_instruments_lasso,
    select_instruments_rf,
    select_instruments_combined,
)
from diagnostics import first_stage, sargan_hansen_test, plot_first_stage, plot_instrument_importance

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
CHAR_COLS        = ["x1_ram", "x2_battery", "x3_screen"]
SELECTION_METHOD = "combined"   # "lasso" | "rf" | "combined"
SAVE_PLOTS       = True
CSV_PATH         = "data/raw/smartphone_blp_real.csv"

# Limite de instrumentos: evita colinearidade com poucos dados
# Regra prática: máx N_obs / 15
MAX_INSTRUMENTS  = 6

os.makedirs("data/raw", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────
# 1. Dados
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 1 — Carregamento dos dados")
print("=" * 60)

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Arquivo não encontrado: {CSV_PATH}\n"
        "Coloque smartphone_blp_real.csv em data/raw/"
    )

df = pd.read_csv(CSV_PATH)
df["outside_share"] = 1 - df.groupby("market_id")["shares"].transform("sum")

print(f"Dados carregados : {df.shape[0]} observações")
print(f"Mercados         : {df['market_id'].nunique()} ({df['market_id'].min()}–{df['market_id'].max()})")
print(f"Produtos únicos  : {df['product_id'].nunique()}")
print(f"Firmas           : {sorted(df['firm_id'].unique())}")
print()
print(df[["market_id", "product_id", "firm_id", "price", "shares", *CHAR_COLS]].head(8).to_string(index=False))

# ─────────────────────────────────────────────
# 2. Candidatos a instrumento
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ETAPA 2 — Construção dos candidatos")
print("=" * 60)

candidates = build_instrument_candidates(df, CHAR_COLS)
print(f"Candidatos gerados: {candidates.shape[1]} variáveis")

price  = df["price"]
X_ctrl = df[CHAR_COLS]

# ─────────────────────────────────────────────
# 3. Seleção ML + limitação por colinearidade
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
    Z_selected = select_instruments_combined(
        candidates, price, X_ctrl,
        rf_threshold=0.04,   # mais restritivo: só instrumentos com importância > 4%
    )
    _, importances = select_instruments_rf(candidates, price, verbose=False)
    if SAVE_PLOTS:
        plot_instrument_importance(importances, save_path="outputs/rf_importance.png")

# Remove colinearidade antes de passar pro BLP
Z_selected = remove_collinear_instruments(Z_selected, corr_threshold=0.90)

# Limita número de instrumentos para evitar colinearidade (N pequeno)
if Z_selected.shape[1] > MAX_INSTRUMENTS:
    _, importances_all = select_instruments_rf(candidates, price, verbose=False)
    # Pega os top-N por importância RF dentro dos já selecionados
    top_cols = [c for c in importances_all.index if c in Z_selected.columns][:MAX_INSTRUMENTS]
    Z_selected = candidates[top_cols]
    print(f"[INFO] Reduzido para {MAX_INSTRUMENTS} instrumentos (regra: N/15 = {len(price)//15})")

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

    df_blp = df.copy()
    df_blp["market_ids"] = df_blp["market_id"]
    df_blp["firm_ids"]   = df_blp["firm_id"]
    df_blp["prices"]     = df_blp["price"]

    for i, col in enumerate(Z_selected.columns):
        df_blp[f"demand_instruments{i}"] = Z_selected[col].values

    char_formula = " + ".join(CHAR_COLS)
    X1_form = pyblp.Formulation(f"1 + {char_formula} + prices")
    X2_form = pyblp.Formulation(f"0 + {char_formula}")

    problem = pyblp.Problem(
        product_formulations=(X1_form, X2_form),
        product_data=df_blp,
        integration=pyblp.Integration("monte_carlo", size=50, specification_options={"seed": 42}),
    )

    print(problem)

    results = problem.solve(
        sigma=np.diag([0.5] * len(CHAR_COLS)),
        optimization=pyblp.Optimization("l-bfgs-b"),
        iteration=pyblp.Iteration("squarem"),
    )

    print(results)

    print(f"\nElasticidades — mercado {df['market_id'].iloc[0]} (primeiros 5 produtos):")
    elasticities = results.compute_elasticities()
    n = min(5, elasticities[0].shape[0])
    print(pd.DataFrame(elasticities[0]).round(3).iloc[:n, :n].to_string())

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
print(f"Observações               : {len(df)}")
print(f"Mercados                  : {df['market_id'].nunique()}")
print(f"Candidatos gerados        : {candidates.shape[1]}")
print(f"Instrumentos selecionados : {Z_selected.shape[1]}")
print(f"F-stat first stage        : {fs_results['f_stat']:.2f}")
print(f"R² parcial                : {fs_results['partial_r2']:.4f}")
print(f"J-stat (Sargan)           : {sargan_results['j_stat']:.4f}")
print(f"p-valor Sargan            : {sargan_results['j_pval']:.4f}")

if SAVE_PLOTS:
    print("\nGráficos salvos em: outputs/")
