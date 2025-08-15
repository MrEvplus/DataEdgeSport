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
    ui_mode: str = "full",           # "full" = selettori in sidebar; "minimal" = niente Campionato/Stagioni
    show_url_input: bool = True,     # mostra/nascondi input URL nel sidebar
) -> Tuple[pd.DataFrame, str]:
    """
    Carica il parquet da Supabase Storage (via DuckDB).
    - ui_mode="minimal": NON mostra i selettori 'Seleziona Campionato' e 'Seleziona stagioni' nel sidebar.
                         Restituisce il DataFrame completo (filtri gestiti altrove, es. Pre-Match Hub).
    - ui_mode="full":    comportamento precedente con selettori nel sidebar e fetch filtrato.

    Ritorna: (df, db_selected) dove db_selected è una stringa descrittiva dell'origine.
    """
    st.sidebar.markdown("**🗄️ Origine: Supabase Storage (Parquet via DuckDB)**")

    parquet_url: str = st.secrets.get(
        "PARQUET_URL",
        "https://<TUO-PROGETTO>.supabase.co/storage/v1/object/public/partite.parquet/latest.parquet",
    )
    if show_url_input:
        parquet_url = st.sidebar.text_input(
            parquet_label,
            value=st.session_state.get("parquet_url", parquet_url),
            key=f"{selectbox_key}__url",
            help="Incolla l'URL (o path) al file parquet su Supabase Storage."
        ).strip()
        st.session_state["parquet_url"] = parquet_url

    if not parquet_url:
        st.warning("Inserisci l'URL del parquet per continuare.")
        return pd.DataFrame(), "Supabase: (URL mancante)"

    # --- Modalità MINIMAL: nessun selettore in sidebar, carico tutto e i filtri si applicano nel Pre-Match ---
    if ui_mode == "minimal":
        df = _read_parquet_filtered(parquet_url, league="", seasons=tuple())
        st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key="rows_loaded_checkbox", disabled=True)
        return df, f"Supabase: {parquet_url}"

    # --- Modalità FULL (comportamento precedente): selettori nel sidebar ---
    # Carico SOLO meta per popolare i menu (campionati/stagioni)
    leagues_df = _duckdb_select_distinct(parquet_url, cols=("country", "sezonul"))
    if leagues_df.empty:
        st.warning("⚠️ Nessun dato trovato nel Parquet.")
        return pd.DataFrame(), f"Supabase: {parquet_url}"

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
    st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key="rows_loaded_checkbox", disabled=True)
    return df, f"Supabase: {parquet_url}"

@st.cache_data(show_spinner=False, ttl=900)
def _read_parquet_filtered(parquet_url: str, league: str, seasons: Tuple[str, ...]) -> pd.DataFrame:
    """Legge il Parquet remoto applicando i filtri. DuckDB+HTTPFS se disponibile, altrimenti pandas in memoria."""
    # Prova via DuckDB
    if _DUCKDB_OK:
        try:
            con = duckdb.connect()
            httpfs_ok = False
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                httpfs_ok = True
            except Exception:
                httpfs_ok = False

            # Se è URL HTTP/S ma httpfs non è disponibile -> fallback
            if parquet_url.startswith(("http://", "https://")) and not httpfs_ok:
                raise RuntimeError("HTTPFS non disponibile: fallback a pandas")

            # Query parametrizzata
            params: list = [parquet_url]
            q = "SELECT * FROM read_parquet(?)"
            where = []

            if league and league != "Tutti":
                where.append("country = ?")
                params.append(league)

            if seasons:
                placeholders = ",".join(["?"] * len(seasons))
                where.append(f"sezonul IN ({placeholders})")
                params.extend(list(seasons))

            if where:
                q += " WHERE " + " AND ".join(where)

            return con.execute(q, params).df()
        except Exception:
            # Qualsiasi errore con DuckDB -> fallback pandas
            pass

    # Fallback pandas/pyarrow (funziona su HTTPS pubblici)
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
        try:
            con = duckdb.connect()
            httpfs_ok = False
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                httpfs_ok = True
            except Exception:
                httpfs_ok = False

            if parquet_url.startswith(("http://", "https://")) and not httpfs_ok:
                raise RuntimeError("HTTPFS non disponibile: fallback a pandas")

            q = f"SELECT DISTINCT {col_list} FROM read_parquet(?)"
            return con.execute(q, [parquet_url]).df()
        except Exception:
            # qualsiasi errore -> pandas
            pass

    df = pd.read_parquet(parquet_url, engine="pyarrow", columns=list(cols))
    return df.drop_duplicates(list(cols))

# ---------------------------------------------------------------------
# Upload manuale (CSV/XLSX) con pulizia minima e dtypes
# ---------------------------------------------------------------------
def load_data_from_file(
    ui_mode: str = "minimal",   # di default nascondiamo i selettori; i filtri globali si scelgono nel Pre-Match
) -> Tuple[pd.DataFrame, str]:
    st.sidebar.markdown("**📂 Origine: Upload Manuale**")

    uploaded_file = st.sidebar.file_uploader(
        "Carica il tuo file Excel o CSV:",
        type=["xls", "xlsx", "csv"],
        key="file_uploader_upload",
    )

    if uploaded_file is None:
        st.info("ℹ️ Carica un file per continuare.")
        return pd.DataFrame(), "Upload: (nessun file)"

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

    # --- Modalità MINIMAL: niente selettori nel sidebar, i filtri si applicano altrove ---
    if ui_mode == "minimal":
        st.sidebar.checkbox(f"Righe caricate da Upload: {len(df):,}", value=True, key="rows_loaded_upload", disabled=True)
        return df, f"Upload: {uploaded_file.name}"

    # --- Modalità FULL (comportamento precedente) ---
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

    st.sidebar.checkbox(f"Righe caricate da Upload: {len(df_filtered):,}", value=True, key="rows_loaded_upload", disabled=True)
    return df_filtered, f"Upload: {uploaded_file.name}"
