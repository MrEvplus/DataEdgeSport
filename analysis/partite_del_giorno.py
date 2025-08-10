"""
analysis/partite_del_giorno.py - Elenco partite in programma oggi
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date

from pre_match import run_pre_match
from live_minute import run_live_minute_analysis
from correct_score_ev import run_correct_score_ev



# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_partite_del_giorno(df: pd.DataFrame, db_selected: str):
    st.title(f"📅 Partite del Giorno - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Selezione data
    data_selezionata = st.date_input("Seleziona data", value=date.today())

    # Filtra partite per data
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
        df_giorno = df[df["Data"].dt.date == data_selezionata]
    else:
        st.error("❌ Colonna 'Data' mancante nel dataset.")
        return

    if df_giorno.empty:
        st.info(f"Nessuna partita trovata per il {data_selezionata}.")
        return

    # Tabella partite
    st.subheader("📌 Elenco partite in programma")
    colonne_da_mostrare = ["Data", "Home", "Away", "Odd home", "Odd Draw", "Odd Away", "Odd Over 2.5", "Odd Under 2.5"]

    for col in colonne_da_mostrare:
        if col not in df_giorno.columns:
            df_giorno[col] = None

    st.dataframe(df_giorno[colonne_da_mostrare], use_container_width=True)

    # Selezione partita per analisi rapida
    st.subheader("🎯 Analisi rapida partita")
    partita_scelta = st.selectbox(
        "Seleziona una partita",
        options=[""] + [f"{row.Home} vs {row.Away}" for _, row in df_giorno.iterrows()]
    )

    if partita_scelta:
        squadra_casa, squadra_ospite = partita_scelta.split(" vs ")

        # Sottomenù analisi
        analisi_opzione = st.radio(
            "Seleziona tipo di analisi",
            ["Pre Match", "Analisi Live", "Correct Score EV"],
            horizontal=True
        )

        if analisi_opzione == "Pre Match":
            df_match = df[(df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)]
            run_pre_match(df_match, db_selected)

        elif analisi_opzione == "Analisi Live":
            df_match = df[(df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)]
            run_live_minute_analysis(df_match)

        elif analisi_opzione == "Correct Score EV":
            df_match = df[(df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)]
            run_correct_score_ev(df_match, db_selected)
