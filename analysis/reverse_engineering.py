"""
analysis/reverse_engineering.py - Analisi storica e ricerca EV+ pattern
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.label import label_match



# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_reverse_engineering(df: pd.DataFrame, db_selected: str):
    st.title(f"🔍 Reverse Engineering EV+ - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Filtri
    stagioni = sorted(df["Stagione"].dropna().unique()) if "Stagione" in df.columns else []
    if stagioni:
        stagione_scelta = st.selectbox("Seleziona Stagione", options=stagioni, index=len(stagioni) - 1)
        df = df[df["Stagione"] == stagione_scelta]

    # Calcolo Label (range di quota)
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # Analisi per Label
    risultati = analizza_ev_per_label(df)

    if risultati.empty:
        st.info("❌ Nessun pattern EV+ trovato.")
    else:
        st.subheader("📌 Pattern EV+ per range di quota")
        st.dataframe(risultati, use_container_width=True)


# -------------------------------------------------------
# 🔹 Analisi EV per Label
# -------------------------------------------------------
def analizza_ev_per_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola ROI e WinRate per mercati principali raggruppati per Label.
    """
    mercati = [
        ("Over 2.5", "Odd Over 2.5", lambda g: g > 2.5),
        ("Under 2.5", "Odd Under 2.5", lambda g: g < 2.5),
        ("BTTS Sì", "Odd GG", lambda btts: btts is True),
        ("BTTS No", "Odd NG", lambda btts: btts is False)
    ]

    risultati = []

    for label in sorted(df["Label"].dropna().unique()):
        subset = df[df["Label"] == label]
        goals = subset["goals_total"]
        btts_vals = subset["btts"]

        for nome_mercato, col_odds, condizione in mercati:
            if col_odds not in subset.columns:
                continue

            sub_mercato = subset.dropna(subset=[col_odds])
            sub_mercato = sub_mercato[sub_mercato[col_odds] >= 1.01]

            if nome_mercato.startswith("Over") or nome_mercato.startswith("Under"):
                vincite = condizione(goals)
            else:
                vincite = condizione(btts_vals)

            if isinstance(vincite, pd.Series):
                win_rate = vincite.mean()
            else:
                # Se condizione è booleana singola
                win_rate = np.mean(vincite)

            tot_match = len(sub_mercato)
            if tot_match == 0:
                continue

            quota_media = sub_mercato[col_odds].mean()
            roi = (quota_media * win_rate - 1) * 100  # ROI%
            risultati.append({
                "Label": label,
                "Mercato": nome_mercato,
                "Match": tot_match,
                "Quota Media": round(quota_media, 2),
                "WinRate %": round(win_rate * 100, 2),
                "ROI %": round(roi, 2)
            })

    df_out = pd.DataFrame(risultati)
    if not df_out.empty:
        df_out = df_out.sort_values(by="ROI %", ascending=False)
    return df_out
