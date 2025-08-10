"""
analysis/live_minute.py - Analisi live dal minuto selezionato
"""

import streamlit as st
import pandas as pd
from label import label_match, extract_minutes



# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_live_minute_analysis(df: pd.DataFrame):
    st.title("⏱️ Analisi Live da Minuto")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Selezione minuto live
    minuto = st.slider("Minuto live", min_value=1, max_value=90, value=30)

    # Selezione squadra
    squadre_disponibili = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))
    squadra_selezionata = st.selectbox("Seleziona squadra per analisi specifica", options=[""] + squadre_disponibili)

    col1, col2 = st.columns(2)

    # 📊 Statistiche campionato post-minuto
    with col1:
        st.subheader(f"📌 Statistiche Campionato - post {minuto}'")
        df_campionato = analizza_post_minuto(df, minuto)
        st.dataframe(df_campionato, use_container_width=True)

    # 📊 Statistiche squadra post-minuto
    with col2:
        if squadra_selezionata:
            st.subheader(f"📌 Statistiche {squadra_selezionata} - post {minuto}'")
            df_squadra = df[(df["Home"] == squadra_selezionata) | (df["Away"] == squadra_selezionata)]
            df_squadra_post = analizza_post_minuto(df_squadra, minuto)
            st.dataframe(df_squadra_post, use_container_width=True)
        else:
            st.info("Seleziona una squadra per analizzare i dati specifici.")


# -------------------------------------------------------
# 🔹 Analisi post-minuto
# -------------------------------------------------------
def analizza_post_minuto(df_filtrato: pd.DataFrame, minuto: int) -> pd.DataFrame:
    """
    Calcola statistiche dal minuto live selezionato fino a fine partita.
    """
    if df_filtrato.empty:
        return pd.DataFrame()

    risultati = []
    for _, row in df_filtrato.iterrows():
        minuti_gol_home = extract_minutes(pd.Series([row.get("minuti goal segnato home", "")]))
        minuti_gol_away = extract_minutes(pd.Series([row.get("minuti goal segnato away", "")]))

        gol_post = sum(1 for m in minuti_gol_home if m > minuto) + sum(1 for m in minuti_gol_away if m > minuto)
        gol_subiti_post = sum(1 for m in minuti_gol_home if m > minuto) if row["Away"] else 0
        gol_subiti_post += sum(1 for m in minuti_gol_away if m > minuto) if row["Home"] else 0

        risultati.append({
            "Data": row.get("Data"),
            "Home": row.get("Home"),
            "Away": row.get("Away"),
            "Goal segnati post-minuto": gol_post,
            "Partite con ≥1 gol": int(gol_post >= 1),
            "Partite con ≥2 gol": int(gol_post >= 2)
        })

    df_post = pd.DataFrame(risultati)

    # Calcolo percentuali
    tot = len(df_post)
    if tot > 0:
        partite_gte1 = (df_post["Partite con ≥1 gol"].sum() / tot) * 100
        partite_gte2 = (df_post["Partite con ≥2 gol"].sum() / tot) * 100
        return pd.DataFrame({
            "Totale Partite": [tot],
            "% con ≥1 Gol": [round(partite_gte1, 2)],
            "% con ≥2 Gol": [round(partite_gte2, 2)],
            "Media Gol post-minuto": [round(df_post["Goal segnati post-minuto"].mean(), 2)]
        })
    return pd.DataFrame()
