"""
diagnostics.py
Testes de diagnóstico para validade e força dos instrumentos.

- F-stat do first stage (regra: F > 10, Stock & Yogo)
- R² parcial dos instrumentos
- Teste de Sargan-Hansen (sobreidentificação)
- Gráficos de diagnóstico
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


# ─────────────────────────────────────────────
# 1. First Stage
# ─────────────────────────────────────────────

def first_stage(
    price: pd.Series,
    Z_selected: pd.DataFrame,
    X_controls: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Regride preço em instrumentos selecionados + controles.
    Retorna dicionário com estatísticas do first stage.
    """
    regressors = pd.concat([
        pd.DataFrame({"const": 1}, index=price.index),
        Z_selected,
        X_controls,
    ], axis=1)

    model = sm.OLS(price, regressors).fit()

    # F-stat apenas dos instrumentos (excluindo constante e controles)
    n_instruments = Z_selected.shape[1]
    instrument_names = list(Z_selected.columns)

    # F-test restrito aos instrumentos
    hypotheses = [f"{col} = 0" for col in instrument_names]
    try:
        f_test = model.f_test(hypotheses)
        f_stat = float(f_test.fvalue)
        f_pval = float(f_test.pvalue)
    except Exception:
        f_stat = np.nan
        f_pval = np.nan

    # R² parcial dos instrumentos
    # R²_partial = (RSS_restrito - RSS_irrestrito) / RSS_restrito
    regressors_no_z = regressors.drop(columns=instrument_names)
    model_restricted = sm.OLS(price, regressors_no_z).fit()
    rss_restricted = model_restricted.ssr
    rss_full = model.ssr
    partial_r2 = (rss_restricted - rss_full) / rss_restricted if rss_restricted > 0 else np.nan

    results = {
        "f_stat": f_stat,
        "f_pval": f_pval,
        "partial_r2": partial_r2,
        "r2": model.rsquared,
        "n_instruments": n_instruments,
        "n_obs": len(price),
        "model": model,
    }

    if verbose:
        print("=" * 50)
        print("DIAGNÓSTICO — FIRST STAGE")
        print("=" * 50)
        print(f"Instrumentos selecionados : {n_instruments}")
        print(f"Observações               : {len(price)}")
        print(f"F-stat (instrumentos)     : {f_stat:.2f}  {'✓ forte' if f_stat > 10 else '✗ fraco (F < 10)'}")
        print(f"p-valor F                 : {f_pval:.4f}")
        print(f"R² parcial dos IVs        : {partial_r2:.4f}")
        print(f"R² total                  : {model.rsquared:.4f}")
        print()
        if f_stat < 10:
            print("[AVISO] F < 10: instrumentos potencialmente fracos.")
            print("        Considere usar Differentiation IVs ou mais candidatos.")

    return results


# ─────────────────────────────────────────────
# 2. Teste de Sargan-Hansen (sobreidentificação)
# ─────────────────────────────────────────────

def sargan_hansen_test(
    price: pd.Series,
    outcome: pd.Series,
    Z_selected: pd.DataFrame,
    X_controls: pd.DataFrame,
    verbose: bool = True,
) -> dict:
    """
    Teste J de Hansen (sobreidentificação).
    H0: todos os instrumentos são exógenos.
    Rejeitar H0 sugere que pelo menos um instrumento é inválido.

    Só faz sentido quando n_instruments > n_endogenous (aqui: n_IV > 1).
    """
    if Z_selected.shape[1] <= 1:
        print("[Sargan] Modelo exatamente identificado — teste não disponível.")
        return {"j_stat": np.nan, "j_pval": np.nan}

    X = pd.concat([
        pd.DataFrame({"const": 1}, index=price.index),
        X_controls,
    ], axis=1).values

    Z = pd.concat([
        pd.DataFrame({"const": 1}, index=price.index),
        Z_selected,
        X_controls,
    ], axis=1).values

    try:
        iv_model = IV2SLS(outcome.values, X, Z).fit()
        residuals = iv_model.resid

        # J stat: n × R² da regressão dos resíduos nos instrumentos
        aux = sm.OLS(residuals, Z).fit()
        j_stat = len(residuals) * aux.rsquared
        df = Z_selected.shape[1] - 1  # graus de liberdade
        from scipy import stats
        j_pval = 1 - stats.chi2.cdf(j_stat, df)

        results = {"j_stat": j_stat, "j_pval": j_pval, "df": df}

        if verbose:
            print("=" * 50)
            print("DIAGNÓSTICO — SARGAN-HANSEN (sobreidentificação)")
            print("=" * 50)
            print(f"J-estatística : {j_stat:.4f}")
            print(f"Graus de lib. : {df}")
            print(f"p-valor       : {j_pval:.4f}")
            if j_pval < 0.05:
                print("[AVISO] H0 rejeitada: possível instrumento inválido.")
            else:
                print("[OK] Não rejeita H0: instrumentos consistentes com exogeneidade.")

    except Exception as e:
        print(f"[Sargan] Erro na estimação: {e}")
        results = {"j_stat": np.nan, "j_pval": np.nan}

    return results


# ─────────────────────────────────────────────
# 3. Gráficos
# ─────────────────────────────────────────────

def plot_first_stage(
    price: pd.Series,
    Z_selected: pd.DataFrame,
    X_controls: pd.DataFrame,
    save_path: str = None,
):
    """
    Plota fitted values do first stage vs. preço observado.
    Boa aderência = instrumentos com poder preditivo.
    """
    regressors = pd.concat([
        pd.DataFrame({"const": 1}, index=price.index),
        Z_selected,
        X_controls,
    ], axis=1)

    model = sm.OLS(price, regressors).fit()
    fitted = model.fittedvalues

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Fitted vs. observado
    axes[0].scatter(price, fitted, alpha=0.4, edgecolors="none", color="steelblue")
    lims = [min(price.min(), fitted.min()), max(price.max(), fitted.max())]
    axes[0].plot(lims, lims, "r--", linewidth=1)
    axes[0].set_xlabel("Preço observado")
    axes[0].set_ylabel("Fitted (first stage)")
    axes[0].set_title(f"First Stage — R²={model.rsquared:.3f}")

    # Resíduos
    axes[1].scatter(fitted, model.resid, alpha=0.4, edgecolors="none", color="salmon")
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Fitted values")
    axes[1].set_ylabel("Resíduos")
    axes[1].set_title("Resíduos do First Stage")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    plt.close()
    return fig


def plot_instrument_importance(importances: pd.Series, top_n: int = 20, save_path: str = None):
    """
    Plota importâncias do Random Forest.
    """
    top = importances.head(top_n)

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    top[::-1].plot(kind="barh", ax=ax, color="steelblue", edgecolor="none")
    ax.set_xlabel("Importância (RF)")
    ax.set_title(f"Top {top_n} instrumentos candidatos — Random Forest")
    ax.axvline(0.01, color="red", linestyle="--", linewidth=1, label="threshold=0.01")
    ax.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Gráfico salvo em: {save_path}")

    plt.close()
    return fig
