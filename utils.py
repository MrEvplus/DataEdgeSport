# utils.py — Loader unico con filtro server-side (lega+stagioni) + cache
#            + "fonte globale" condivisa da tutti i moduli dell’app.
from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
import streamlit as st

# =====================================================
# DuckDB (opzionale) per lettura Parquet via HTTPFS
# =====================================================
try:
    import duckdb  # type: ignore
    _DUCKDB_OK = True
except Exception:
    duckdb = None  # type: ignore
    _DUCKDB_OK = False


# =====================================================
# Label di base (match buckets) — compat con codice esistente
# =====================================================
def _get_odd(row: pd.Series, *candidates: str) -> float:
    for c in candidates:
        if c in row:
            try:
                return float(str(row[c]).replace(",", "."))
            except Exception:
                pass
    return float("nan")

def label_match(row: pd.Series) -> str:
    """
    Classifica la partita in macro-bucket in base alle quote 1x2 pre-match.
    Usata per etichettare e filtrare (anche live). Compatibile con codice esistente.
    """
    try:
        h = float(_get_odd(row, "Odd home", "Odd Home"))
        a = float(_get_odd(row, "Odd Away", "Odd away"))
    except Exception:
        return "Others"

    if np.isnan(h) or np.isnan(a):
        return "Others"

    # Entrambe basse → partita molto equilibrata
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


# =====================================================
# Estrazione/normalizzazione minuti (utility leggera)
# =====================================================
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
                pass
    return out


# =====================================================
# ------- CACHE GLOBALE (fonte per tutti i moduli) -----
# =====================================================
_GLOBAL_DF_KEY = "GLOBAL_SOURCE_DF"
_GLOBAL_INFO_KEY = "GLOBAL_SOURCE_INFO"  # dict: origin/url/league/seasons/rows

def _set_global_source(df: pd.DataFrame, info: Dict[str, Any]) -> None:
    st.session_state[_GLOBAL_DF_KEY] = df
    st.session_state[_GLOBAL_INFO_KEY] = info

def get_global_source_df() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Ritorna la fonte dati condivisa (df + info).
    Se non presente, restituisce (DataFrame vuoto, {}).
    """
    return st.session_state.get(_GLOBAL_DF_KEY, pd.DataFrame()), st.session_state.get(_GLOBAL_INFO_KEY, {})


# =====================================================
# --------- Loader Parquet via DuckDB+HTTPFS ----------
# --------- con filtro server-side + cache 15’ --------
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)
def _read_parquet_filtered(parquet_url: str, league: str, seasons: Tuple[str, ...]) -> pd.DataFrame:
    """
    Legge il Parquet remoto applicando i filtri direttamente a server (DuckDB).
    Fallback a pandas/pyarrow se DuckDB/HTTPFS non disponibili.
    Cache di 15 minuti per combinazione (url, league, seasons).
    """
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

            if parquet_url.startswith(("http://", "https://")) and not httpfs_ok:
                raise RuntimeError("HTTPFS non disponibile: fallback a pandas")

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
            # qualsiasi errore -> fallback pandas
            pass

    # Fallback pandas/pyarrow
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
            pass

    df = pd.read_parquet(parquet_url, engine="pyarrow", columns=list(cols))
    return df.drop_duplicates(list(cols))


# =====================================================
# ------------------- UI WRAPPERS ---------------------
# =====================================================
def load_data_from_supabase(
    parquet_label: str = "Parquet file URL (Supabase Storage):",
    selectbox_key: str = "selectbox_campionato_duckdb",
    ui_mode: str = "full",           # "full" = selettori; "minimal" = nessun selettore
    show_url_input: bool = True,     # mostra/nascondi l'input URL
) -> Tuple[pd.DataFrame, str]:
    """
    Carica il parquet da Supabase Storage (via DuckDB) con filtro server-side e cache.
    - ui_mode="minimal": nessun selettore; ritorna tutto il dataset (filtri gestiti altrove).
    - ui_mode="full":    mostra selettori e scarica SOLO (lega+stagioni) selezionati.

    Ritorna: (df, db_selected). Pubblica anche la FONTE GLOBALE in session_state.
    """
    st.sidebar.markdown("**🗄️ Origine: Supabase Storage (Parquet via DuckDB)**")

    parquet_url: str = st.secrets.get(
        "PARQUET_URL",
        "https://<TUO-PROGETTO>.supabase.co/storage/v1/object/public/partite.parquet/latest.parquet",
    )
    if show_url_input:
        parquet_url = st.sidebar.text_input(
            parquet_label,
            value=st.session_state.get(f"{selectbox_key}__url", parquet_url),
            key=f"{selectbox_key}__url",
            help="Incolla l'URL (o path) al file parquet su Supabase Storage."
        ).strip()

    if not parquet_url:
        st.warning("Inserisci l'URL del parquet per continuare.")
        return pd.DataFrame(), "Supabase: (URL mancante)"

    # ---- modalità MINIMAL: nessun selettore in sidebar ----
    if ui_mode == "minimal":
        df = _read_parquet_filtered(parquet_url, league="", seasons=tuple())
        st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key=f"{selectbox_key}__rows", disabled=True)
        _set_global_source(df, {"origin": "supabase", "url": parquet_url, "league": "", "seasons": [], "rows": len(df)})
        return df, f"Supabase: {parquet_url}"

    # ---- modalità FULL: prima leggo meta (league + seasons) ----
    meta = _duckdb_select_distinct(parquet_url, cols=("country", "sezonul"))
    if meta.empty:
        st.warning("⚠️ Nessun dato trovato nel Parquet.")
        return pd.DataFrame(), f"Supabase: {parquet_url}"

    leagues = sorted(meta["country"].dropna().astype(str).unique()) if "country" in meta.columns else []
    league = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + leagues,
        index=1 if leagues else 0,
        key=selectbox_key,
    )

    if league != "Tutti" and "sezonul" in meta.columns:
        seasons_all = sorted(
            meta.loc[meta["country"].astype(str) == league, "sezonul"]
                .dropna().astype(str).unique()
        )
    else:
        seasons_all = sorted(meta["sezonul"].dropna().astype(str).unique()) if "sezonul" in meta.columns else []

    seasons_sel = st.sidebar.multiselect(
        "Seleziona stagioni:",
        options=seasons_all,
        default=seasons_all,
        key=f"{selectbox_key}__seasons",
    )

    # Esponi l’elenco stagioni a chi vuole costruire preset (app.py)
    st.session_state["supabase_seasons_choices"] = seasons_all

    # Lettura filtrata (server-side) + cache
    df = _read_parquet_filtered(parquet_url, league, tuple(seasons_sel))
    st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key=f"{selectbox_key}__rows", disabled=True)

    # Pubblica FONTE GLOBALE per l’intera app
    _set_global_source(
        df,
        {
            "origin": "supabase",
            "url": parquet_url,
            "league": league,
            "seasons": list(seasons_sel),
            "rows": len(df),
        },
    )
    return df, f"Supabase: {parquet_url}"


# -----------------------------------------------------
# Upload manuale (CSV/XLSX) con pulizia minima e dtypes
# -----------------------------------------------------
def load_data_from_file(
    ui_mode: str = "minimal",
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

    # CSV o Excel
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

    # ---- MINIMAL ----
    if ui_mode == "minimal":
        st.sidebar.checkbox(f"Righe caricate da Upload: {len(df):,}", value=True, key="rows_loaded_upload", disabled=True)
        _set_global_source(df, {"origin": "upload", "name": uploaded_file.name, "rows": len(df)})
        return df, f"Upload: {uploaded_file.name}"

    # ---- FULL (selettori locali) ----
    leagues = sorted(df["country"].dropna().astype(str).unique()) if "country" in df.columns else []
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
    _set_global_source(df_filtered, {"origin": "upload", "name": uploaded_file.name, "league": league, "seasons": seasons_sel, "rows": len(df_filtered)})
    return df_filtered, f"Upload: {uploaded_file.name}"
