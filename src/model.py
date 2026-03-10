import joblib

def save_model(model, path: str) -> None:
    """Salva o modelo treinado."""
    joblib.dump(model, path)

def load_model(path: str):
    """Carrega um modelo salvo."""
    return joblib.load(path)