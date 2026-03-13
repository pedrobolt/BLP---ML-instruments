# BLP + ML Instruments

Pipeline para estimação de demanda com **modelos de demanda diferenciada (BLP)** usando **Machine Learning para seleção automática de instrumentos**.

---

## Motivação

O estimador BLP (Berry, Levinsohn & Pakes, 1995) resolve o problema da endogeneidade do preço via variáveis instrumentais. O desafio prático: com muitos candidatos a instrumento, é difícil saber quais usar sem introduzir viés por instrumentos fracos.

Este repositório implementa uma estratégia de dois estágios:

1. **Construir** um conjunto amplo de candidatos (BLP clássicos + Differentiation IVs)
2. **Filtrar** automaticamente via Lasso e/ou Random Forest, seguindo Belloni et al. (2012) e Gandhi & Houde (2020)

---

## Estrutura

```
blp-ml-instruments/
├── data/
│   └── raw/                    # dados brutos (gerados por simulate_data.py)
├── src/
│   ├── simulate_data.py        # gerador de dados sintéticos
│   ├── instruments.py          # construção e seleção de candidatos
│   ├── diagnostics.py          # F-stat, Sargan-Hansen, gráficos
│   └── run_pipeline.py         # pipeline completo
├── notebooks/
│   └── full_pipeline.ipynb     # walkthrough interativo
├── outputs/                    # gráficos gerados
├── requirements.txt
└── README.md
```

---

## Instalação

```bash
git clone https://github.com/seu-usuario/blp-ml-instruments.git
cd blp-ml-instruments
pip install -r requirements.txt
```

---

## Como usar

### Pipeline completo (linha de comando)

```bash
python src/run_pipeline.py
```

### Passo a passo no Python

```python
from src.simulate_data import simulate_blp_data
from src.instruments import build_instrument_candidates, select_instruments_combined
from src.diagnostics import first_stage, sargan_hansen_test

# 1. Dados
df = simulate_blp_data(T=50, J=10)

# 2. Candidatos
char_cols = ["x1", "x2", "x3"]
candidates = build_instrument_candidates(df, char_cols)

# 3. Seleção ML
Z_selected = select_instruments_combined(
    candidates, df["price"], X_controls=df[char_cols]
)

# 4. Diagnóstico
fs = first_stage(df["price"], Z_selected, df[char_cols])
# Regra: F-stat > 10 → instrumentos fortes
```

---

## Tipos de candidatos construídos

| Tipo | Descrição | Referência |
|---|---|---|
| `blp_same_*` | Soma das características da mesma firma | BLP (1995) |
| `blp_rival_*` | Soma das características de firmas rivais | BLP (1995) |
| `div_same_*` | Distância L2 quadrática — mesma firma | Gandhi & Houde (2020) |
| `div_rival_*` | Distância L2 quadrática — firmas rivais | Gandhi & Houde (2020) |
| `mindist_rival_*` | Distância mínima ao rival mais próximo | Gandhi & Houde (2020) |

---

## Métodos de seleção

### Lasso (base teórica sólida)
Aplica `LassoCV` com partial-out dos controles observáveis.
Fundamentado em **Belloni, Chernozhukov & Hansen (2012)**.

```python
from src.instruments import select_instruments_lasso
Z, cols = select_instruments_lasso(candidates, price, X_controls)
```

### Random Forest (exploratório)
Usa importância de features como filtro de pré-seleção.
**Não garante exogeneidade** — use como primeiro filtro apenas.

```python
from src.instruments import select_instruments_rf
Z, importances = select_instruments_rf(candidates, price, threshold=0.01)
```

### Combinado (recomendado)
Interseção das duas seleções — conservador e robusto.

```python
from src.instruments import select_instruments_combined
Z = select_instruments_combined(candidates, price, X_controls)
```

---

## Diagnósticos

```python
from src.diagnostics import first_stage, sargan_hansen_test

# F-stat do first stage (regra de bolso: F > 10)
fs = first_stage(price, Z_selected, X_controls)

# Teste de sobreidentificação (H0: instrumentos exógenos)
sargan = sargan_hansen_test(price, outcome, Z_selected, X_controls)
```

---

## Estimação BLP (via PyBLP)

Com os instrumentos selecionados, a estimação final usa o pacote `pyblp`:

```python
import pyblp

# Adiciona instrumentos ao DataFrame
for i, col in enumerate(Z_selected.columns):
    df[f"demand_instruments{i}"] = Z_selected[col].values

problem = pyblp.Problem(
    product_formulations=(
        pyblp.Formulation("1 + x1 + x2 + x3 + price"),
        pyblp.Formulation("0 + x1 + x2 + x3"),
    ),
    product_data=df,
)

results = problem.solve(sigma=np.diag([0.5, 0.5, 0.5]))
elasticities = results.compute_elasticities()
```

---

## Referências

- **Berry, Levinsohn & Pakes (1995)** — Automobile prices in market equilibrium. *Econometrica*
- **Berry (1994)** — Estimating discrete-choice models of product differentiation. *RAND Journal*
- **Belloni, Chernozhukov & Hansen (2012)** — Sparse models and methods for optimal instruments. *Econometrica*
- **Gandhi & Houde (2020)** — Measuring substitution patterns in differentiated-products industries. *NBER WP*
- **Conlon & Gortmaker (2020)** — Best practices for differentiated products demand estimation with PyBLP. *RAND Journal*
- **Mikusheva & Sun (2022)** — Inference with many weak instruments. *Review of Economic Studies*

---

## Aviso importante

ML seleciona o que **prediz preço**, não o que é **exógeno**. A validade dos instrumentos depende de argumentos econômicos — o pipeline filtra candidatos fracos, mas não substitui o raciocínio teórico. Sempre valide com o teste de Sargan-Hansen.
