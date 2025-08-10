"""
analysis/correct_score_ev.py - Analisi EV+ mercato Correct Score
"""

import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_correct_score_ev(df: pd.DataFrame, db_selected: str):
    st.title(f"🎯 Correct Score EV+ - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Filtra eventuale stagione
    stagioni = sorted(df["Stagione"].dropna().unique()) if "Stagione" in df.columns else []
    if stagioni:
        stagione_scelta = st.selectbox("Seleziona Stagione", options=stagioni, index=len(stagioni) - 1)
        df = df[df["Stagione"] == stagione_scelta]

    # Calcolo EV sul mercato Correct Score
    risultati_ev = calcola_correct_score_ev(df)

    if not risultati_ev.empty:
        st.dataframe(risultati_ev, use_container_width=True)
    else:
        st.info("❌ Nessun dato sufficiente per calcolare EV Correct Score.")


# -------------------------------------------------------
# 🔹 Calcolo EV mercato Correct Score
# -------------------------------------------------------
def calcola_correct_score_ev(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola Expected Value (EV) e ROI per i punteggi esatti più comuni.
    """
    # Controllo colonne essenziali
    colonne_richieste = ["Home Goal FT", "Away Goal FT", "Odd CS 0-0", "Odd CS 1-0", "Odd CS 0-1", "Odd CS 1-1"]
    colonne_presenti = [col for col in colonne_richieste if col in df.columns]

    if len(colonne_presenti) < 5:
        return pd.DataFrame()

    # Lista dei punteggi da analizzare
    cs_markets = {
        "0-0": "Odd CS 0-0",
        "1-0": "Odd CS 1-0",
        "0-1": "Odd CS 0-1",
        "1-1": "Odd CS 1-1"
    }

    risultati = []

    for cs, col_odds in cs_markets.items():
        if col_odds not in df.columns:
            continue

        sub_df = df.dropna(subset=["Home Goal FT", "Away Goal FT", col_odds])
        sub_df = sub_df[sub_df[col_odds] >= 1.01]

        tot_match = len(sub_df)
        if tot_match == 0:
            continue

        # Calcolo win/loss
        vittorie = 0
        profitto = 0
        quota_media = sub_df[col_odds].mean()

        for _, row in sub_df.iterrows():
            risultato = f"{int(row['Home Goal FT'])}-{int(row['Away Goal FT'])}"
            if risultato == cs:
                vittorie += 1
                profitto += (row[col_odds] - 1)
            else:
                profitto -= 1

        win_pct = (vittorie / tot_match) * 100
        roi_pct = (profitto / tot_match) * 100
        ev_pct = (quota_media * (vittorie / tot_match) - 1) * 100

        risultati.append({
            "Correct Score": cs,
            "Matches": tot_match,
            "Win %": round(win_pct, 2),
            "ROI %": round(roi_pct, 2),
            "EV %": round(ev_pct, 2),
            "Avg Odds": round(quota_media, 2)
        })

    return pd.DataFrame(risultati).sort_values(by="EV %", ascending=False)
