"""
analysis/reverse_batch.py - Analisi EV+ in batch su più campionati
"""

import streamlit as st
import pandas as pd
from analysis.reverse_engineering import analizza_ev_per_label


# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_reverse_batch(df: pd.DataFrame):
    st.title("📊 Reverse Engineering EV+ - Analisi Batch")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Selezione campionati da includere
    campionati_disponibili = sorted(df["country"].dropna().unique())
    campionati_scelti = st.multiselect(
        "Seleziona campionati da includere",
        options=campionati_disponibili,
        default=campionati_disponibili
    )

    stagioni_disponibili = sorted(df["Stagione"].dropna().unique()) if "Stagione" in df.columns else []
    stagione_scelta = None
    if stagioni_disponibili:
        stagione_scelta = st.selectbox("Seleziona stagione", options=["Tutte"] + stagioni_disponibili)

    risultati_batch = []

    for campionato in campionati_scelti:
        df_camp = df[df["country"] == campionato]

        if stagione_scelta and stagione_scelta != "Tutte":
            df_camp = df_camp[df_camp["Stagione"] == stagione_scelta]

        if df_camp.empty:
            continue

        # Analisi EV per questo campionato
        risultati_camp = analizza_ev_per_label(df_camp)
        if not risultati_camp.empty:
            risultati_camp["Campionato"] = campionato
            risultati_batch.append(risultati_camp)

    if not risultati_batch:
        st.info("❌ Nessun campionato ha prodotto risultati EV+.")
        return

    # Combina risultati
    df_finale = pd.concat(risultati_batch, ignore_index=True)

    # Ordina per ROI %
    df_finale = df_finale.sort_values(by="ROI %", ascending=False)

    st.subheader("🏆 Ranking EV+ Campionati/Mercati")
    st.dataframe(df_finale, use_container_width=True)
