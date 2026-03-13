import pandas as pd
import mlflow
import mlflow.sklearn
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

def train():
    # Carrega dados
    df = pd.read_csv('data/processed/train_processed.csv')
    X = df.drop(columns=['Exited'])
    y = df['Exited']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inicia experimento no MLflow
    mlflow.set_experiment("churn-prediction")

    with mlflow.start_run():
        # Parâmetros
        params = {
            'n_estimators': 100,
            'random_state': 42,
            'scale_pos_weight': y_train.value_counts()[0] / y_train.value_counts()[1]
        }

        # Treina modelo
        model = XGBClassifier(**params, eval_metric='auc', verbosity=0)
        model.fit(X_train, y_train)

        # Métricas
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_prob)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Loga no MLflow
        mlflow.log_params(params)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("recall_churn", report['1']['recall'])
        mlflow.log_metric("f1_churn", report['1']['f1-score'])
        mlflow.sklearn.log_model(model, "model")

        # Salva modelo
        joblib.dump(model, 'models/xgb_model.pkl')

        print(f"AUC-ROC: {auc:.4f}")
        print(f"Recall Churn: {report['1']['recall']:.4f}")
        print(f"Experimento registrado no MLflow!")

if __name__ == "__main__":
    train()