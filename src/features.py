import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica todas as transformações de features."""
    df = df.copy()
    # transformações virão aqui
    return df