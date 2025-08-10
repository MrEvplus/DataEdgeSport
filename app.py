# app.py — Trading Dashboard V2.0 (robust path & clean imports)

import os, sys
import streamlit as st
import pandas as pd

# ------------------------------------------------------------------
# PATH FIX: consente import sia se i moduli sono nella stessa cartella
# di app.py, sia se sono nella cartella "parent" (repo root).
# Es: .../repo/dataedgesport/app.py  e .../repo/loader.py
# ------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))          # .../dataedgesport  (o repo root)
PARENT_DIR = os.path.dirname(APP_DIR)                         # repo root (se app.py è in sottocartella)
for p in (APP_DIR, PARENT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

# -----------------------------
# Config & Data
# -----------------------------
from config import LEAGUE_MAPPING
from loader import (
    load_data_from_supabase,
    load_data_from_file,
    filter_by_league,
)
from preprocess import preprocess_dataframe

# -----------------------------
# Moduli Analisi
# -----------------------------
from macros import run_macro_stats
from team_stats import run_team_stats
from pre_match import run_pre_match
from correct_score_ev import run_correct_score_ev
from live_minute import run_live_minute_analysis
from partite_del_giorno import run_partite_del_giorno
from reverse_engineering import run_reverse_engineering

# =======================================================
# Configurazione pagina
# =======================================================
st.set_page_config(
    page_title="Trading Dashboard V2.0",
    layout="wide",
)

st.sidebar.title("📊 Trading Dashboard V2.0")

# =======================================================
# Menu principale
# =======================================================
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
    ],
    index=0,
    key="menu_main_radio",
)

# =======================================================
# Selezione origine dati
# =======================================================
origine_dati = st.sidebar.radio(
    "Seleziona origine dati:",
    ["Supabase", "Upload Manuale"],
    index=0,
    key="origine_dati_radio",
)

# =======================================================
# Caricamento dati
# =======================================================
if origine_dati == "Supabase":
    df = load_data_from_supabase()
else:
    df = load_data_from_file()

if df is None or len(df) == 0:
    st.error("❌ Nessun dato disponibile. Verifica origine dati o file caricato.")
    st.stop()

# =======================================================
# Preprocessamento dati
# =======================================================
df = preprocess_dataframe(df)

# Alias di colonna per evitare rotture dovute a maiuscole/minuscole diverse
if "Odd Home" in df.columns and "Odd home" not in df.columns:
    df["Odd home"] = df["Odd Home"]

# =======================================================
# Filtro campionato (League)
# =======================================================
df, db_selected = filter_by_league(df)

if df is None or len(df) == 0:
    st.warning("⚠️ Dopo il filtro campionato non ci sono dati.")
    st.stop()

# =======================================================
# Gestione stato selezione squadre (reset al cambio campionato)
# =======================================================
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

# =======================================================
# Debug: colonne presenti
# =======================================================
with st.expander("✅ Colonne presenti nel dataset", expanded=False):
    st.write(list(df.columns))

# =======================================================
# Selezione stagioni (se disponibile)
# =======================================================
if "Stagione" in df.columns:
    stagioni_disponibili = sorted(df["Stagione"].dropna().unique().tolist())
    if stagioni_disponibili:
        opzione_range = st.sidebar.selectbox(
            "Seleziona un intervallo stagioni:",
            ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"],
            index=0,
            key="stagioni_range_select",
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
                default=stagioni_disponibili,
                key="stagioni_multiselect",
            )

        if stagioni_scelte:
            df = df[df["Stagione"].isin(stagioni_scelte)]

# =======================================================
# Routing menu → chiamata ai moduli
# =======================================================
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

# =================== EOF ===================
