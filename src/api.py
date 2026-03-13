from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Churn Prediction API")

model = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

class Cliente(BaseModel):
    CreditScore: float
    Gender: int
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography_France: int
    Geography_Germany: int
    Geography_Spain: int
    balance_salary_ratio: float
    has_zero_balance: int
    age_group: int
    high_risk: int

@app.get("/")
def root():
    return {"message": "Churn Prediction API funcionando!"}

@app.post("/predict")
def predict(cliente: Cliente):
    data = np.array([[
        cliente.CreditScore, cliente.Gender, cliente.Age,
        cliente.Tenure, cliente.Balance, cliente.NumOfProducts,
        cliente.HasCrCard, cliente.IsActiveMember, cliente.EstimatedSalary,
        cliente.Geography_France, cliente.Geography_Germany, cliente.Geography_Spain,
        cliente.balance_salary_ratio, cliente.has_zero_balance,
        cliente.age_group, cliente.high_risk
    ]])

    cols_to_scale = [0, 2, 3, 4, 8, 12]
    data[:, cols_to_scale] = scaler.transform(
        np.zeros((1, 6))
    )

    prob = model.predict_proba(data)[0][1]
    pred = int(prob >= 0.5)

    return {
        "churn_probability": round(float(prob), 4),
        "prediction": pred,
        "risk": "Alto" if prob >= 0.7 else "Médio" if prob >= 0.4 else "Baixo"
    }

@app.get("/health")
def health():
    return {"status": "ok"}