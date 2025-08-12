# utils.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from typing import Iterable, Tuple, List

# DuckDB opzionale: fallback a pandas/pyarrow se non disponibile
try:
    import duckdb  # type: ignore
    _DUCKDB_OK = True
except Exception:
    duckdb = None  # type: ignore
    _DUCKDB_OK = False

# ---------------------------------------------------------------------
# Secrets (niente credenziali hard-coded nel repo)
# ---------------------------------------------------------------------
SUPABASE_URL: str = st.secrets.get("SUPABASE_URL", "")
SUPABASE_KEY: str = st.secrets.get("SUPABASE_KEY", "")

# ---------------------------------------------------------------------
# Helper: accesso case-insensitive a colonne quota
# ---------------------------------------------------------------------
def _get_odd(row: dict | pd.Series, *candidates: str) -> float | np.nan:
    """Ritorna il primo valore disponibile tra i nomi colonna candidati (case-insensitive)."""
    # 1) match diretto
    for c in candidates:
        if c in row and pd.notna(row[c]):
            return row[c]
    # 2) fallback case-insensitive
    keys = row.keys() if isinstance(row, dict) else row.index
    lower_map = {str(k).lower(): k for k in keys}
    for c in candidates:
        k = lower_map.get(c.lower())
        if k is not None and pd.notna(row[k]):
            return row[k]
    return np.nan

# ---------------------------------------------------------------------
# Label range in base alle quote (robusto ai casi Odd home/Odd Home)
# ---------------------------------------------------------------------
def label_match(row: dict | pd.Series) -> str:
    """
    Classifica il match in una fascia di quote in base a 'Odd home'/'Odd Home' e 'Odd Away'/'Odd away'.
    """
    try:
        h = float(_get_odd(row, "Odd home", "Odd Home"))
        a = float(_get_odd(row, "Odd Away", "Odd away"))
    except Exception:
        return "Others"

    if np.isnan(h) or np.isnan(a):
        return "Others"

    # SuperCompetitive
    if h <= 3 and a <= 3:
        return "SuperCompetitive H<=3 A<=3"

    # Classificazione Home
    if h < 1.5:
        return "H_StrongFav <1.5"
    if 1.5 <= h <= 2:
        return "H_MediumFav 1.5-2"
    if 2 < h <= 3:
        return "H_SmallFav 2-3"

    # Classificazione Away
    if a < 1.5:
        return "A_StrongFav <1.5"
    if 1.5 <= a <= 2:
        return "A_MediumFav 1.5-2"
    if 2 < a <= 3:
        return "A_SmallFav 2-3"

    return "Others"

# ---------------------------------------------------------------------
# Estrazione minuti goal da Serie di stringhe tipo "12;45;78" o "12,45"
# ---------------------------------------------------------------------
def extract_minutes(series: pd.Series) -> List[int]:
    """
    Estrae i minuti di goal da una Serie (stringhe tipo '12; 45; 78' o '12,45').
    Ignora valori nulli, vuoti e non numerici. Ritorna una lista di interi.
    """
    if series is None or len(series) == 0:
        return []
    series = series.fillna("").astype(str)

    out: List[int] = []
    for val in series:
        s = val.strip().replace(",", ";")
        if s in ("", ";"):
            continue
        for part in s.split(";"):
            p = part.strip()
            if not p:
                continue
            try:
                out.append(int(float(p)))
            except Exception:
                # ignora scarti
                pass
    return out

# ---------------------------------------------------------------------
# Loader Parquet via DuckDB+HTTPFS con filtro server-side e cache
# (UI wrapper + funzioni cache con fallback pandas)
# ---------------------------------------------------------------------
def load_data_from_supabase(
    parquet_label: str = "Parquet file URL (Supabase Storage):",
    selectbox_key: str = "selectbox_campionato_duckdb",
) -> Tuple[pd.DataFrame, str]:
    """
    Wrapper UI: chiede URL Parquet, campionato e stagioni.
    Esegue poi un fetch *filtrato* (DuckDB se disponibile, altrimenti pandas in memoria) con cache.
    Ritorna (df, campionato_selezionato).
    """
    st.sidebar.markdown("### 🌐 Origine: Supabase Storage (Parquet via DuckDB)")

    parquet_url: str = st.sidebar.text_input(
        parquet_label,
        value=st.secrets.get(
            "PARQUET_URL",
            "https://<TUO-PROGETTO>.supabase.co/storage/v1/object/public/partite.parquet/latest.parquet",
        ),
        key=f"{selectbox_key}__url",
    ).strip()

    if not parquet_url:
        st.warning("Inserisci l'URL del Parquet in Supabase Storage.")
        st.stop()

    # Carico SOLO meta per popolare i menu (campionati/stagioni)
    leagues_df = _duckdb_select_distinct(parquet_url, cols=("country", "sezonul"))
    if leagues_df.empty:
        st.warning("⚠️ Nessun dato trovato nel Parquet.")
        st.stop()

    leagues = sorted(leagues_df["country"].dropna().astype(str).unique()) if "country" in leagues_df.columns else []
    league = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + leagues,
        index=1 if leagues else 0,
        key=selectbox_key,
    )

    if league != "Tutti" and "sezonul" in leagues_df.columns:
        seasons_all = sorted(
            leagues_df.loc[leagues_df["country"].astype(str) == league, "sezonul"]
            .dropna().astype(str).unique()
        )
    else:
        seasons_all = sorted(leagues_df["sezonul"].dropna().astype(str).unique()) if "sezonul" in leagues_df.columns else []

    seasons_sel = st.sidebar.multiselect(
        "Seleziona stagioni:",
        options=seasons_all,
        default=seasons_all,
        key=f"{selectbox_key}__seasons",
    )

    df = _read_parquet_filtered(parquet_url, league, tuple(seasons_sel))

    # Mapping canonico essenziale (se arrivano colonne grezze)
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
        "mgola": "minuti goal segnato away",
        "datameci": "Data",
        "orameci": "Orario",
    }
    df = df.rename(columns=col_map)

    # Cast mirati e dtypes stretti
    for col in ("Odd home", "Odd Away", "Odd Draw"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False)
                  .replace("nan", np.nan).astype(float)
            ).astype("Float32")

    for col in ("Home Goal FT", "Away Goal FT", "Home Goal 1T", "Away Goal 1T"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int16")

    if "country" in df.columns:
        df["country"] = df["country"].astype("category")
    if "Home" in df.columns:
        df["Home"] = df["Home"].astype("category")
    if "Away" in df.columns:
        df["Away"] = df["Away"].astype("category")
    if "Data" in df.columns:
        df["Data"] = pd.to_datetime(df["Data"], errors="coerce")

    st.sidebar.write(f"✅ Righe caricate: {len(df):,}")
    return df, league

@st.cache_data(show_spinner=False, ttl=900)
def _read_parquet_filtered(parquet_url: str, league: str, seasons: Tuple[str, ...]) -> pd.DataFrame:
    """Legge il Parquet remoto applicando i filtri. DuckDB+HTTPFS se disponibile, altrimenti pandas in memoria."""
    if _DUCKDB_OK:
        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass

        where = []
        if league and league != "Tutti":
            where.append(f"country = {duckdb.literal(league)}")
        if seasons:
            seasons_sql = ", ".join(duckdb.literal(s) for s in seasons)
            where.append(f"sezonul IN ({seasons_sql})")
        where_sql = (" WHERE " + " AND ".join(where)) if where else ""

        query = f"SELECT * FROM read_parquet({duckdb.literal(parquet_url)}){where_sql}"
        return con.execute(query).df()
    else:
        # Fallback: scarico e filtro localmente
        df = pd.read_parquet(parquet_url, engine="pyarrow")
        if league and league != "Tutti" and "country" in df.columns:
            df = df[df["country"].astype(str) == league]
        if seasons and "sezonul" in df.columns:
            df = df[df["sezonul"].astype(str).isin(seasons)]
        return df

@st.cache_data(show_spinner=False, ttl=900)
def _duckdb_select_distinct(parquet_url: str, cols: Iterable[str]) -> pd.DataFrame:
    """DISTINCT su poche colonne per popolare i menu. DuckDB se c’è, sennò pandas."""
    col_list = ", ".join(cols)
    if _DUCKDB_OK:
        con = duckdb.connect()
        try:
            con.execute("INSTALL httpfs; LOAD httpfs;")
        except Exception:
            pass
        q = f"SELECT DISTINCT {col_list} FROM read_parquet({duckdb.literal(parquet_url)})"
        return con.execute(q).df()
    else:
        df = pd.read_parquet(parquet_url, engine="pyarrow", columns=list(cols))
        return df.drop_duplicates(list(cols))

# ---------------------------------------------------------------------
# Upload manuale (CSV/XLSX) con pulizia minima e dtypes
# ---------------------------------------------------------------------
def load_data_from_file() -> Tuple[pd.DataFrame, str]:
    st.sidebar.markdown("### 📂 Origine: Upload Manuale")

    uploaded_file = st.sidebar.file_uploader(
        "Carica il tuo file Excel o CSV:",
        type=["xls", "xlsx", "csv"],
        key="file_uploader_upload",
    )

    if uploaded_file is None:
        st.info("ℹ️ Carica un file per continuare.")
        st.stop()

    # Riconosci CSV o Excel
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        xls = pd.ExcelFile(uploaded_file)
        sheet_name = xls.sheet_names[0]
        df = pd.read_excel(xls, sheet_name=sheet_name)

    # Normalizzazione basilare
    df.columns = df.columns.astype(str).str.strip()

    # Cast quote note se presenti
    for col in ("cotaa", "cotae", "cotad", "Odd home", "Odd Home", "Odd Away", "Odd Draw"):
        if col in df.columns:
            df[col] = (
                df[col].astype(str).str.replace(",", ".", regex=False)
                  .replace("nan", np.nan).astype(float)
            )

    # Date se presenti
    for dcol in ("datameci", "Data"):
        if dcol in df.columns:
            df[dcol] = pd.to_datetime(df[dcol], errors="coerce")

    # Selezione campionato
    if "country" in df.columns:
        leagues = sorted(df["country"].dropna().astype(str).unique())
    else:
        leagues = []

    league = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + leagues,
        index=1 if leagues else 0,
        key="selectbox_campionato_upload",
    )

    df_filtered = df.copy()
    if league != "Tutti" and "country" in df.columns:
        df_filtered = df[df["country"].astype(str) == league]

    if "sezonul" in df_filtered.columns:
        seasons_all = sorted(df_filtered["sezonul"].dropna().astype(str).unique())
    else:
        seasons_all = []

    seasons_sel = st.sidebar.multiselect(
        "Seleziona le stagioni:",
        options=seasons_all,
        default=seasons_all,
        key="multiselect_stagioni_upload",
    )

    if seasons_sel and "sezonul" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["sezonul"].astype(str).isin(seasons_sel)]

    st.sidebar.write(f"✅ Righe caricate da Upload Manuale: {len(df_filtered):,}")
    return df_filtered, league
