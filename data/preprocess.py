"""
data/preprocess.py - Pulizia e preparazione del dataset per Trading Dashboard V2.0
"""

import pandas as pd
import numpy as np
from label import label_match, extract_minutes


# -------------------------------------------------------
# 🔹 Funzione di pre-elaborazione
# -------------------------------------------------------
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pulisce e arricchisce il dataframe con colonne derivate utili per l'analisi.
    """
    df = df.copy()

    # 1️⃣ Conversione date
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    # 2️⃣ Conversione numeri (virgole → punti)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # 3️⃣ Calcolo Label (una sola volta)
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # 4️⃣ Goals totali, 1T, 2T
    if "Home Goal FT" in df.columns and "Away Goal FT" in df.columns:
        df["goals_total"] = df["Home Goal FT"] + df["Away Goal FT"]

    if "Home Goal 1T" in df.columns and "Away Goal 1T" in df.columns:
        df["goals_1st_half"] = df["Home Goal 1T"] + df["Away Goal 1T"]

    if "goals_total" in df.columns and "goals_1st_half" in df.columns:
        df["goals_2nd_half"] = df["goals_total"] - df["goals_1st_half"]

    # 5️⃣ BTTS
    if "Home Goal FT" in df.columns and "Away Goal FT" in df.columns:
        df["btts"] = ((df["Home Goal FT"] > 0) & (df["Away Goal FT"] > 0)).astype(int)

    # 6️⃣ Minuti goal segnati (estrazione come lista di int)
    if "minuti goal segnato home" in df.columns:
        df["goal_minutes_home"] = df["minuti goal segnato home"].apply(
            lambda x: extract_minutes(pd.Series([x]))
        )

    if "minuti goal segnato away" in df.columns:
        df["goal_minutes_away"] = df["minuti goal segnato away"].apply(
            lambda x: extract_minutes(pd.Series([x]))
        )

    # 7️⃣ Risultato finale come stringa "H-A"
    if "Home Goal FT" in df.columns and "Away Goal FT" in df.columns:
        df["score_ft"] = df["Home Goal FT"].astype(str) + "-" + df["Away Goal FT"].astype(str)

    return df
