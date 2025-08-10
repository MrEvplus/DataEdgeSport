"""
app.py - Trading Dashboard V2.0
"""

import streamlit as st
import pandas as pd
from datetime import date

# Config & Data
from config import LEAGUE_MAPPING
from dataedgesport.data.loader import (
    load_data_from_supabase,
    load_data_from_file,
    filter_by_league,
)

from data.preprocess import preprocess_dataframe

# Moduli Analisi
from analysis.macros import run_macro_stats
from analysis.team_stats import run_team_stats
from analysis.pre_match import run_pre_match
from analysis.correct_score_ev import run_correct_score_ev
from analysis.live_minute import run_live_minute_analysis
from analysis.partite_del_giorno import run_partite_del_giorno
from analysis.reverse_engineering import run_reverse_engineering

# -------------------------------------------------------
# Configurazione pagina
# -------------------------------------------------------
st.set_page_config(
    page_title="Trading Dashboard V2.0",
    layout="wide"
)

st.sidebar.title("📊 Trading Dashboard V2.0")

# -------------------------------------------------------
# Menu principale
# -------------------------------------------------------
menu_option = st.sidebar.radio(
    "Naviga tra le sezioni:",
    [
        "Macro Stats per Campionato",
        "Statistiche per Squadre",
        "Confronto Pre Match",
        "Correct Score EV",
        "Analisi Live da Minuto",
        "Partite del Giorno",
        "🧠 Reverse Engineering EV+",
    ]
)

# -------------------------------------------------------
# Selezione origine dati
# -------------------------------------------------------
origine_dati = st.sidebar.radio(
    "Seleziona origine dati:",
    ["Supabase", "Upload Manuale"]
)

if origine_dati == "Supabase":
    df = load_data_from_supabase()
else:
    df = load_data_from_file()

# -------------------------------------------------------
# Preprocessamento dati
# -------------------------------------------------------
df = preprocess_dataframe(df)

# -------------------------------------------------------
# Filtro campionato
# -------------------------------------------------------
df, db_selected = filter_by_league(df)

# -------------------------------------------------------
# Gestione stato selezione squadre
# -------------------------------------------------------
if "squadra_casa" not in st.session_state:
    st.session_state["squadra_casa"] = ""
if "squadra_ospite" not in st.session_state:
    st.session_state["squadra_ospite"] = ""
if "campionato_corrente" not in st.session_state:
    st.session_state["campionato_corrente"] = db_selected
else:
    if st.session_state["campionato_corrente"] != db_selected:
        st.session_state["squadra_casa"] = ""
        st.session_state["squadra_ospite"] = ""
        st.session_state["campionato_corrente"] = db_selected

# -------------------------------------------------------
# Debug: colonne presenti
# -------------------------------------------------------
with st.expander("✅ Colonne presenti nel dataset", expanded=False):
    st.write(list(df.columns))

# -------------------------------------------------------
# Selezione stagioni
# -------------------------------------------------------
if "Stagione" in df.columns:
    stagioni_disponibili = sorted(df["Stagione"].dropna().unique())

    opzione_range = st.sidebar.selectbox(
        "Seleziona un intervallo stagioni predefinito:",
        ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"]
    )

    if opzione_range == "Tutte":
        stagioni_scelte = stagioni_disponibili
    elif opzione_range == "Ultime 3":
        stagioni_scelte = stagioni_disponibili[-3:]
    elif opzione_range == "Ultime 5":
        stagioni_scelte = stagioni_disponibili[-5:]
    elif opzione_range == "Ultime 10":
        stagioni_scelte = stagioni_disponibili[-10:]
    else:
        stagioni_scelte = st.sidebar.multiselect(
            "Seleziona manualmente le stagioni da includere:",
            options=stagioni_disponibili,
            default=stagioni_disponibili
        )

    if stagioni_scelte:
        df = df[df["Stagione"].isin(stagioni_scelte)]

# -------------------------------------------------------
# Routing menu
# -------------------------------------------------------
if menu_option == "Macro Stats per Campionato":
    run_macro_stats(df, db_selected)

elif menu_option == "Statistiche per Squadre":
    run_team_stats(df, db_selected)

elif menu_option == "Confronto Pre Match":
    run_pre_match(df, db_selected)

elif menu_option == "Correct Score EV":
    run_correct_score_ev(df, db_selected)

elif menu_option == "Analisi Live da Minuto":
    run_live_minute_analysis(df)

elif menu_option == "Partite del Giorno":
    run_partite_del_giorno(df, db_selected)

elif menu_option == "🧠 Reverse Engineering EV+":
    run_reverse_engineering(df)
