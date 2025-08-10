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
from analysis.macros import run_macro_stats
from analysis.team_stats import run_team_stats
from analysis.pre_match import run_pre_match
from analysis.correct_score_ev import run_correct_score_ev
from analysis.live_minute import run_live_minute_analysis
from analysis.partite_del_giorno import run_partite_del_giorno
from analysis.reverse_engineering import run_reverse_engineering

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
)
# =======================================================
# Selezione origine dati
# =======================================================
origine_dati = st.sidebar.radio("Seleziona origine dati:", ["Supabase", "Upload Manuale"])
df = load_data_from_supabase() if origine_dati == "Supabase" else load_data_from_file()
if df is None or len(df) == 0:
    st.error("❌ Nessun dato disponibile.")
    st.stop()

df = preprocess_dataframe(df)

# alias per differenze maiuscole/minuscole su 'Odd home'
if "Odd Home" in df.columns and "Odd home" not in df.columns:
    df["Odd home"] = df["Odd Home"]

df, db_selected = filter_by_league(df)
if df is None or len(df) == 0:
    st.warning("⚠️ Dopo il filtro campionato non ci sono dati.")
    st.stop()

with st.expander("✅ Colonne presenti nel dataset", expanded=False):
    st.write(list(df.columns))

if "Stagione" in df.columns:
    stagioni = sorted(df["Stagione"].dropna().unique().tolist())
    if stagioni:
        scelta = st.sidebar.selectbox(
            "Seleziona intervallo stagioni:", ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"]
        )
        if scelta == "Ultime 3":
            df = df[df["Stagione"].isin(stagioni[-3:])]
        elif scelta == "Ultime 5":
            df = df[df["Stagione"].isin(stagioni[-5:])]
        elif scelta == "Ultime 10":
            df = df[df["Stagione"].isin(stagioni[-10:])]
        elif scelta == "Personalizza":
            pick = st.sidebar.multiselect("Seleziona stagioni:", options=stagioni, default=stagioni)
            if pick:
                df = df[df["Stagione"].isin(pick)]

# Routing
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