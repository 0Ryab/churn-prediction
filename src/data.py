import pandas as pd

def load_raw_data(path: str) -> pd.DataFrame:
    """Carrega os dados brutos."""
    return pd.read_csv(path)

def save_processed_data(df: pd.DataFrame, path: str) -> None:
    """Salva os dados processados."""
    df.to_csv(path, index=False)