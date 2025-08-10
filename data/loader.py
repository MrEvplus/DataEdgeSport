"""
data/loader.py - Caricamento dati da Supabase o file locale per Trading Dashboard V2.0
"""

import os
import pandas as pd
import duckdb
import streamlit as st
from config import COLUMN_MAPPING

# -------------------------------------------------------
# 🔹 Legge credenziali in sicurezza
# -------------------------------------------------------
def get_supabase_credentials():
    """
    Recupera le credenziali da st.secrets (Streamlit Cloud) o da variabili d'ambiente (.env).
    """
    supabase_url = None
    supabase_key = None

    # Da Streamlit Cloud
    if "SUPABASE_URL" in st.secrets and "SUPABASE_KEY" in st.secrets:
        supabase_url = st.secrets["SUPABASE_URL"]
        supabase_key = st.secrets["SUPABASE_KEY"]

    # Da variabili d'ambiente in locale
    elif os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")

    else:
        st.error("❌ Credenziali Supabase non trovate. Aggiungile in `.streamlit/secrets.toml` o in `.env`.")
        st.stop()

    return supabase_url, supabase_key


# -------------------------------------------------------
# 🔹 Carica dati dal Parquet su Supabase Storage
# -------------------------------------------------------
def load_data_from_supabase():
    """
    Carica il dataset da Supabase Storage (Parquet) e applica il mapping colonne.
    """
    st.sidebar.markdown("### 🌐 Origine: Supabase Storage (Parquet via DuckDB)")

    # Leggi URL Parquet
    parquet_url = st.sidebar.text_input(
        "🔗 URL del file Parquet su Supabase Storage:",
        value=""
    )

    if not parquet_url:
        st.warning("Inserisci l'URL del file Parquet.")
        st.stop()

    query_all = f"SELECT * FROM read_parquet('{parquet_url}')"

    try:
        df_all = duckdb.query(query_all).to_df()
    except Exception as e:
        st.error(f"❌ Errore nel caricamento Parquet: {e}")
        st.stop()

    if df_all.empty:
        st.warning("⚠️ Nessun dato trovato nel Parquet.")
        st.stop()

    # Applica mapping colonne
    df_all.rename(columns=COLUMN_MAPPING, inplace=True)

    # Conversione date
    if "Data" in df_all.columns:
        df_all["Data"] = pd.to_datetime(df_all["Data"], errors="coerce")

    st.sidebar.success(f"✅ Righe caricate da Supabase: {len(df_all)}")

    return df_all


# -------------------------------------------------------
# 🔹 Carica dati da upload manuale (CSV o Excel)
# -------------------------------------------------------
def load_data_from_file():
    """
    Carica il dataset da file caricato manualmente (CSV, XLS, XLSX).
    """
    st.sidebar.markdown("### 📂 Origine: Upload Manuale")
    uploaded_file = st.sidebar.file_uploader(
        "Carica il tuo file dati:",
        type=["csv", "xls", "xlsx"]
    )

    if uploaded_file is None:
        st.info("Carica un file per continuare.")
        st.stop()

    # Riconosce CSV o Excel
    if uploaded_file.name.endswith(".csv"):
        try:
            df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError:
            df = pd.read_csv(uploaded_file, encoding="latin1", sep=";")
    else:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    # Applica mapping colonne
    df.rename(columns=COLUMN_MAPPING, inplace=True)

    # Conversione date
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    st.sidebar.success(f"✅ Righe caricate da file: {len(df)}")

    return df


# -------------------------------------------------------
# 🔹 Selezione campionato
# -------------------------------------------------------
def filter_by_league(df):
    """
    Filtra il dataframe in base al campionato selezionato.
    """
    if "country" not in df.columns:
        st.error("⚠️ La colonna 'country' non esiste nel dataset.")
        st.stop()

    leagues = sorted(df["country"].dropna().unique())
    selected_league = st.sidebar.selectbox("Seleziona Campionato:", ["Tutti"] + leagues)

    if selected_league != "Tutti":
        df = df[df["country"] == selected_league]

    return df, selected_league
