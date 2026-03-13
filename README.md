# Churn Prediction — Banco

Modelo de Machine Learning para previsão de churn de clientes bancários,
com pipeline completo de dados, modelagem, API e MLOps.

## Resultados do Modelo
| Métrica | Valor |
|--------|-------|
| AUC-ROC | 0.8847 |
| Recall Churn | 77.2% |
| F1 Churn | 0.64 |
| Modelo | XGBoost |

## Principais Insights
- Clientes entre 40-60 anos têm maior propensão ao churn
- Clientes com 3+ produtos têm churn próximo de 100%
- Alemanha tem taxa de churn de 37.9% vs média de 21.2%
- Mulheres churnam mais (28%) que homens (15.9%)
- Membros inativos têm correlação negativa com retenção

## Estrutura do Projeto
```
churn-prediction/
├── data/
│   ├── raw/          # dados originais
│   └── processed/    # dados tratados
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modelagem.ipynb
├── src/
│   ├── api.py        # FastAPI
│   ├── train.py      # script de treino com MLflow
│   ├── data.py
│   ├── features.py
│   └── model.py
├── models/           # modelos e scaler salvos
└── requirements.txt
```

## Setup
```bash
# Clone o repositório
git clone https://github.com/0Ryab/churn-prediction.git
cd churn-prediction

# Crie e ative o ambiente virtual
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Instale as dependências
pip install -r requirements.txt
```

## Reproduzindo o Projeto
```bash
# 1. Treinar o modelo
python src/train.py

# 2. Visualizar experimentos
python -m mlflow ui

# 3. Subir a API
python -m uvicorn src.api:app --reload
```

## Usando a API
```bash
# Verificar saúde da API
curl http://127.0.0.1:8000/health

# Fazer previsão
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "CreditScore": 600,
    "Gender": 0,
    "Age": 52,
    "Tenure": 3,
    "Balance": 120000,
    "NumOfProducts": 3,
    "HasCrCard": 1,
    "IsActiveMember": 0,
    "EstimatedSalary": 80000,
    "Geography_France": 0,
    "Geography_Germany": 1,
    "Geography_Spain": 0,
    "balance_salary_ratio": 1.5,
    "has_zero_balance": 0,
    "age_group": 2,
    "high_risk": 1
  }'

# Resposta esperada
# {"churn_probability": 0.9982, "prediction": 1, "risk": "Alto"}
```

## Pipeline Completo
```
Dados Brutos → EDA → Feature Engineering → Modelagem → MLflow → FastAPI
```

## Tecnologias
- Python 3.13
- XGBoost, Scikit-learn
- MLflow
- FastAPI, Uvicorn
- Pandas, NumPy
- SHAP
- Docker (em breve)

## Etapas
- [x] Fase 1 — Setup e estrutura do projeto
- [x] Fase 2 — EDA
- [x] Fase 3 — Feature Engineering
- [x] Fase 4 — Modelagem
- [x] Fase 5 — MLOps
- [x] Fase 6 — Documentação