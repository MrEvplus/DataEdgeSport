import numpy as np
import pandas as pd
import streamlit as st
import duckdb

# ----------------------------------------------------------
# Variabili di connessione Supabase (rimosse da versione pubblica)
# ----------------------------------------------------------
# Inserire in .streamlit/secrets.toml oppure in variabili d'ambiente
SUPABASE_URL = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY = st.secrets.get("SUPABASE_KEY", "")

# ----------------------------------------------------------
# Caricamento da DuckDB + Parquet su Supabase Storage
# ----------------------------------------------------------
def load_data_from_supabase(parquet_label="Parquet file URL (Supabase Storage):", selectbox_key="selectbox_campionato_duckdb"):
    st.sidebar.markdown("### 🌐 Origine: Supabase Storage (Parquet via DuckDB)")

    parquet_url = st.sidebar.text_input(
        parquet_label,
        value=st.secrets.get("SUPABASE_PARQUET_URL", "")
    )

    query_all = f"SELECT * FROM read_parquet('{parquet_url}')"

    try:
        df_all = duckdb.query(query_all).to_df()
    except Exception as e:
        st.error(f"❌ Errore DuckDB: {str(e)}")
        st.stop()

    if df_all.empty:
        st.warning("⚠️ Nessun dato trovato nel Parquet.")
        st.stop()

    # Lista campionati
    campionati_disponibili = sorted(df_all["country"].dropna().unique()) if "country" in df_all.columns else []

    campionato_scelto = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + campionati_disponibili,
        index=1 if campionati_disponibili else 0,
        key=selectbox_key
    )

    if not campionato_scelto:
        st.info("ℹ️ Seleziona un campionato per procedere.")
        st.stop()

    # Filtra campionato
    df_filtered = df_all if campionato_scelto == "Tutti" else df_all[df_all["country"] == campionato_scelto]

    # Lista stagioni
    stagioni_disponibili = sorted(df_filtered["sezonul"].dropna().unique()) if "sezonul" in df_filtered.columns else []

    stagioni_scelte = st.sidebar.multiselect(
        "Seleziona le stagioni:",
        options=stagioni_disponibili,
        default=stagioni_disponibili,
        key="multiselect_stagioni_duckdb"
    )

    if stagioni_scelte:
        df_filtered = df_filtered[df_filtered["sezonul"].isin(stagioni_scelte)]

    # Mappatura colonne
    col_map = {
        "country": "country",
        "sezonul": "Stagione",
        "txtechipa1": "Home",
        "txtechipa2": "Away",
        "scor1": "Home Goal FT",
        "scor2": "Away Goal FT",
        "scorp1": "Home Goal 1T",
        "scorp2": "Away Goal 1T",
        "cotaa": "Odd home",
        "cotad": "Odd Away",
        "cotae": "Odd Draw",
        "mgolh": "minuti goal segnato home",
        "mgola": "minuti goal segnato away"
    }
    df_filtered.rename(columns=col_map, inplace=True)

    # Pulizia valori numerici
    for col in df_filtered.columns:
        if df_filtered[col].dtype == object:
            df_filtered[col] = df_filtered[col].str.replace(",", ".")
    df_filtered = df_filtered.apply(pd.to_numeric, errors="ignore")

    # Conversione date
    if "Data" in df_filtered.columns:
        df_filtered["Data"] = pd.to_datetime(df_filtered["Data"], errors="coerce")

    st.sidebar.success(f"✅ Righe caricate: {len(df_filtered)}")
    return df_filtered, campionato_scelto

# ----------------------------------------------------------
# Upload Manuale (Excel o CSV)
# ----------------------------------------------------------
def load_data_from_file():
    st.sidebar.markdown("### 📂 Origine: Upload Manuale")

    uploaded_file = st.sidebar.file_uploader(
        "Carica il tuo file Excel o CSV:",
        type=["xls", "xlsx", "csv"],
        key="file_uploader_upload"
    )

    if uploaded_file is None:
        st.info("ℹ️ Carica un file per continuare.")
        st.stop()

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        df = pd.read_excel(xls, sheet_name=xls.sheet_names[0])

    # Pulizia colonne
    df.columns = df.columns.str.strip().str.lower()
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.replace(",", ".")
    df = df.apply(pd.to_numeric, errors="ignore")

    if "datameci" in df.columns:
        df["datameci"] = pd.to_datetime(df["datameci"], errors="coerce")

    campionati_disponibili = sorted(df["country"].dropna().unique()) if "country" in df.columns else []

    campionato_scelto = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + campionati_disponibili,
        key="selectbox_campionato_upload"
    )

    if not campionato_scelto:
        st.info("ℹ️ Seleziona un campionato per procedere.")
        st.stop()

    df_filtered = df if campionato_scelto == "Tutti" else df[df["country"] == campionato_scelto]

    stagioni_disponibili = sorted(df_filtered["sezonul"].dropna().unique()) if "sezonul" in df_filtered.columns else []
    stagioni_scelte = st.sidebar.multiselect(
        "Seleziona le stagioni:",
        options=stagioni_disponibili,
        default=stagioni_disponibili,
        key="multiselect_stagioni_upload"
    )

    if stagioni_scelte:
        df_filtered = df_filtered[df_filtered["sezonul"].isin(stagioni_scelte)]

    st.sidebar.success(f"✅ Righe caricate: {len(df_filtered)}")
    return df_filtered, campionato_scelto

# ----------------------------------------------------------
# Classificazione match per quota
# ----------------------------------------------------------
def label_match(row):
    try:
        h = float(row.get("Odd home", np.nan))
        a = float(row.get("Odd Away", np.nan))
    except:
        return "Others"

    if np.isnan(h) or np.isnan(a):
        return "Others"

    if h <= 3 and a <= 3:
        return "SuperCompetitive H<=3 A<=3"
    if h < 1.5:
        return "H_StrongFav <1.5"
    elif h <= 2:
        return "H_MediumFav 1.5-2"
    elif h <= 3:
        return "H_SmallFav 2-3"
    if a < 1.5:
        return "A_StrongFav <1.5"
    elif a <= 2:
        return "A_MediumFav 1.5-2"
    elif a <= 3:
        return "A_SmallFav 2-3"
    return "Others"

# ----------------------------------------------------------
# Estrazione minuti goal
# ----------------------------------------------------------
def extract_minutes(series):
    all_minutes = []
    series = series.fillna("")
    for val in series:
        val = str(val).strip()
        if not val or val == ";":
            continue
        parts = val.replace(",", ";").split(";")
        for part in parts:
            part = part.strip()
            if part.replace(".", "", 1).isdigit():
                all_minutes.append(int(float(part)))
    return all_minutes
