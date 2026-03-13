"""
simulate_data.py
Gera dados sintéticos de mercados diferenciados para testar o pipeline BLP + ML.

Estrutura:
- T mercados (ex: cidades × períodos)
- J produtos por mercado
- Características observáveis (x1, x2, x3)
- Qualidade não observada (xi) correlacionada com preço → endogeneidade
- Shares derivados de logit misto
"""

import numpy as np
import pandas as pd
from scipy.special import softmax


def simulate_blp_data(
    T: int = 50,          # número de mercados
    J: int = 10,          # produtos por mercado
    K: int = 3,           # características observáveis
    n_agents: int = 500,  # consumidores simulados por mercado (integração)
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    records = []

    for t in range(T):
        # --- Características dos produtos ---
        X = rng.normal(0, 1, size=(J, K))  # (J, K)

        # --- Qualidade não observada (xi): correlacionada com custo → preço endógeno ---
        xi = rng.normal(0, 1, size=J)

        # --- Custo marginal (não observado pelo econometrista) ---
        marginal_cost = np.exp(0.5 * X[:, 0] + rng.normal(0, 0.3, J))

        # --- Preço: markup + xi (aqui está a endogeneidade) ---
        price = marginal_cost * 1.3 + 0.5 * xi + rng.normal(0, 0.1, J)
        price = np.clip(price, 0.5, None)

        # --- Heterogeneidade dos consumidores ---
        nu = rng.normal(0, 1, size=(n_agents, K))   # preferências aleatórias

        # --- Parâmetros verdadeiros ---
        beta = np.array([1.0, 0.5, -0.3])    # utilidade linear
        alpha = 1.5                            # sensibilidade ao preço
        sigma = np.array([0.8, 0.5, 0.3])     # heterogeneidade (desvio-padrão)

        # --- Utilidade: delta_j + mu_ij ---
        delta = X @ beta - alpha * price + xi   # (J,)

        # mu_ij: interação nu_i × características
        mu = nu @ (sigma * X).T  # (n_agents, J)

        # Utilidade total (inclui outside option j=0 com u=0)
        U = delta[None, :] + mu   # (n_agents, J)
        U_with_outside = np.hstack([np.zeros((n_agents, 1)), U])

        # --- Shares via softmax por agente ---
        probs = softmax(U_with_outside, axis=1)[:, 1:]  # descarta outside option
        shares = probs.mean(axis=0)   # (J,)

        # --- Firma aleatória para cada produto ---
        firm_ids = rng.integers(0, max(1, J // 3), size=J)

        for j in range(J):
            records.append({
                "market_id": t,
                "product_id": j,
                "firm_id": firm_ids[j],
                "price": price[j],
                "shares": shares[j],
                "xi": xi[j],             # verdadeiro — só para diagnóstico
                "marginal_cost": marginal_cost[j],
                **{f"x{k+1}": X[j, k] for k in range(K)},
            })

    df = pd.DataFrame(records)

    # Outside share por mercado
    df["outside_share"] = 1 - df.groupby("market_id")["shares"].transform("sum")

    return df


if __name__ == "__main__":
    df = simulate_blp_data(T=50, J=10)
    df.to_csv("data/raw/simulated_markets.csv", index=False)
    print(df.head(10).to_string())
    print(f"\nShape: {df.shape}")
    print(f"Mercados: {df['market_id'].nunique()}")
    print(f"Produtos únicos por mercado: {df.groupby('market_id')['product_id'].count().mean():.1f}")
