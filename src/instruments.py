"""
instruments.py
Construção de candidatos a instrumento e seleção via ML.

Etapas:
1. BLP instruments clássicos (soma das características dos rivais)
2. Differentiation IVs (Gandhi & Houde, 2020)
3. Seleção via Lasso (Belloni et al., 2012)
4. Seleção via Random Forest (exploratório)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. Construção dos candidatos
# ─────────────────────────────────────────────

def build_instrument_candidates(
    df: pd.DataFrame,
    char_cols: list[str],
    market_col: str = "market_id",
    firm_col: str = "firm_id",
    product_col: str = "product_id",
) -> pd.DataFrame:
    """
    Para cada produto, constrói:
    - BLP instruments: soma das características de (a) mesma firma, (b) firmas rivais
    - Differentiation IVs: distância L2 quadrática das características

    Retorna DataFrame indexado igual ao df original.
    """
    rows = []

    for market, mkt_df in df.groupby(market_col):
        for idx, product in mkt_df.iterrows():
            same_firm = mkt_df[
                (mkt_df[firm_col] == product[firm_col]) &
                (mkt_df[product_col] != product[product_col])
            ]
            rival_firms = mkt_df[mkt_df[firm_col] != product[firm_col]]

            row = {"_idx": idx}

            for col in char_cols:
                # BLP clássicos
                row[f"blp_same_{col}"]  = same_firm[col].sum()
                row[f"blp_rival_{col}"] = rival_firms[col].sum()
                row[f"n_same_{col}"]    = len(same_firm)
                row[f"n_rival_{col}"]   = len(rival_firms)

                # Differentiation IVs (Gandhi & Houde)
                row[f"div_same_{col}"]  = ((product[col] - same_firm[col]) ** 2).sum()
                row[f"div_rival_{col}"] = ((product[col] - rival_firms[col]) ** 2).sum()

                # Mínima distância ao rival mais próximo
                if len(rival_firms) > 0:
                    row[f"mindist_rival_{col}"] = (
                        (product[col] - rival_firms[col]).abs().min()
                    )
                else:
                    row[f"mindist_rival_{col}"] = 0.0

            rows.append(row)

    candidates = pd.DataFrame(rows).set_index("_idx")
    candidates.index.name = None
    return candidates


# ─────────────────────────────────────────────
# 2. Seleção via Lasso (base teórica sólida)
# ─────────────────────────────────────────────

def select_instruments_lasso(
    X_candidates: pd.DataFrame,
    price: pd.Series,
    X_controls: pd.DataFrame = None,
    cv: int = 5,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Index]:
    """
    First stage: regride preço nos candidatos via LassoCV.
    Remove variação das características observáveis antes (partial out).

    Retorna (instrumentos selecionados, colunas selecionadas).

    Referência: Belloni, Chernozhukov & Hansen (2012)
    """
    y = price.values

    # Partial out controles (características observáveis) via OLS
    if X_controls is not None:
        from numpy.linalg import lstsq
        X_ctrl = X_controls.values
        coef, _, _, _ = lstsq(
            np.hstack([np.ones((len(X_ctrl), 1)), X_ctrl]),
            y, rcond=None
        )
        y_resid = y - np.hstack([np.ones((len(X_ctrl), 1)), X_ctrl]) @ coef

        Z_resid = X_candidates.copy()
        for col in X_candidates.columns:
            z = X_candidates[col].values
            c, _, _, _ = lstsq(
                np.hstack([np.ones((len(X_ctrl), 1)), X_ctrl]),
                z, rcond=None
            )
            Z_resid[col] = z - np.hstack([np.ones((len(X_ctrl), 1)), X_ctrl]) @ c
    else:
        y_resid = y
        Z_resid = X_candidates.copy()

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("lasso", LassoCV(cv=cv, max_iter=20000, n_alphas=100)),
    ])

    pipe.fit(Z_resid.values, y_resid)

    coef = pipe.named_steps["lasso"].coef_
    selected_mask = coef != 0
    selected_cols = X_candidates.columns[selected_mask]

    if verbose:
        print(f"[Lasso] α ótimo: {pipe.named_steps['lasso'].alpha_:.4f}")
        print(f"[Lasso] Selecionados: {selected_mask.sum()} / {len(selected_mask)} candidatos")
        if len(selected_cols) > 0:
            print(f"[Lasso] Variáveis: {list(selected_cols)}")

    return X_candidates[selected_cols], selected_cols


# ─────────────────────────────────────────────
# 3. Seleção via Random Forest (exploratório)
# ─────────────────────────────────────────────

def select_instruments_rf(
    X_candidates: pd.DataFrame,
    price: pd.Series,
    threshold: float = 0.01,
    n_estimators: int = 500,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Usa importância de features do Random Forest como filtro de candidatos.

    ATENÇÃO: RF não garante exogeneidade — use apenas como pré-filtro.
    Sempre valide os instrumentos selecionados com teste de Sargan/Hansen.

    Retorna (instrumentos selecionados, série de importâncias).
    """
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_candidates.values)

    rf.fit(X_scaled, price.values)

    importances = pd.Series(
        rf.feature_importances_,
        index=X_candidates.columns,
    ).sort_values(ascending=False)

    selected = importances[importances >= threshold].index

    if verbose:
        print(f"[RF] Selecionados: {len(selected)} / {len(importances)} candidatos")
        print(f"[RF] Top 10 importâncias:")
        print(importances.head(10).to_string())

    return X_candidates[selected], importances


# ─────────────────────────────────────────────
# 4. Combinação: interseção Lasso ∩ RF
# ─────────────────────────────────────────────

def select_instruments_combined(
    X_candidates: pd.DataFrame,
    price: pd.Series,
    X_controls: pd.DataFrame = None,
    lasso_cv: int = 5,
    rf_threshold: float = 0.01,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Seleciona instrumentos que passam em AMBOS os filtros (Lasso e RF).
    Estratégia conservadora: reduz candidatos fracos preservando os mais robustos.
    """
    _, lasso_cols = select_instruments_lasso(
        X_candidates, price, X_controls, cv=lasso_cv, verbose=verbose
    )
    _, importances = select_instruments_rf(
        X_candidates, price, threshold=rf_threshold, verbose=verbose
    )

    rf_cols = importances[importances >= rf_threshold].index
    combined = lasso_cols.intersection(rf_cols)

    if verbose:
        print(f"\n[Combinado] Interseção: {len(combined)} instrumentos")
        print(f"  Apenas Lasso: {len(lasso_cols.difference(rf_cols))}")
        print(f"  Apenas RF:    {len(rf_cols.difference(lasso_cols))}")

    if len(combined) == 0:
        print("[AVISO] Interseção vazia — retornando seleção do Lasso")
        return X_candidates[lasso_cols]

    return X_candidates[combined]


# ─────────────────────────────────────────────
# 5. Remoção de colinearidade (VIF ou correlação)
# ─────────────────────────────────────────────

def remove_collinear_instruments(
    Z: pd.DataFrame,
    corr_threshold: float = 0.90,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Remove instrumentos com correlação >= corr_threshold entre si.
    Mantém o primeiro de cada par colinear (priorizando ordem de importância).
    """
    corr = Z.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] >= corr_threshold)]

    if verbose:
        print(f"[Colinearidade] Removidos: {len(to_drop)} — {to_drop}")
        print(f"[Colinearidade] Restantes: {Z.shape[1] - len(to_drop)}")

    return Z.drop(columns=to_drop)

