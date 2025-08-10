"""
utils/label.py - Funzioni per gestione Label e parsing minuti gol
"""

import numpy as np
import pandas as pd

# -------------------------------------------------------
# 🔹 Classificazione match in base alle quote
# -------------------------------------------------------
def label_match(row):
    """
    Classifica un match in una fascia di quote (Label) basata su odd home e odd away.
    """
    try:
        h = float(row.get("Odd Home", np.nan))
        a = float(row.get("Odd Away", np.nan))
    except Exception:
        return "Others"

    if np.isnan(h) or np.isnan(a):
        return "Others"

    # Super Competitive (quote vicine e basse)
    if h <= 3 and a <= 3:
        return "SuperCompetitive H<=3 A<=3"

    # Classificazione Home favorito
    if h < 1.5:
        return "H_StrongFav <1.5"
    elif 1.5 <= h <= 2:
        return "H_MediumFav 1.5-2"
    elif 2 < h <= 3:
        return "H_SmallFav 2-3"

    # Classificazione Away favorito
    if a < 1.5:
        return "A_StrongFav <1.5"
    elif 1.5 <= a <= 2:
        return "A_MediumFav 1.5-2"
    elif 2 < a <= 3:
        return "A_SmallFav 2-3"

    return "Others"


# -------------------------------------------------------
# 🔹 Estrazione minuti gol
# -------------------------------------------------------
def extract_minutes(series: pd.Series):
    """
    Estrae i minuti di goal da una Serie Pandas con formato stringa.
    Gestisce:
    - valori NaN o vuoti
    - separatori diversi (; ,)
    - conversione in interi
    """
    all_minutes = []

    # Sostituisci NaN con stringa vuota
    series = series.fillna("")

    for val in series:
        val = str(val).strip()
        if val in ["", ";"]:
            continue

        # Uniforma separatori
        parts = val.replace(",", ";").split(";")
        for part in parts:
            part = part.strip()
            if part.replace(".", "", 1).isdigit():
                all_minutes.append(int(float(part)))

    return all_minutes
