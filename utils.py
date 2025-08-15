# utils.py — Loader unico con filtro server-side (lega+stagioni) + cache
#            + fonte globale condivisa per tutti i moduli dell'app.

from __future__ import annotations

from typing import Iterable, List, Tuple, Dict, Any
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
# Odds helpers + Labeling coerente
# =====================================================
def _get_odd(row, *candidates):
    """Ritorna la prima quota trovata tra gli alias passati.
    Accetta Series/dict; converte '2,05' -> 2.05.
    """
    try:
        keys = list(candidates)
    except Exception:
        keys = candidates
    # Consenti accesso case-insensitive
    try:
        available = {str(k).lower(): k for k in row.keys()}  # type: ignore[attr-defined]
    except Exception:
        available = None

    for c in keys:
        if available is not None:
            k = available.get(str(c).lower())
            if k is None:
                continue
            val = row[k]
        else:
            # fallback generico
            if c not in row:
                continue
            val = row[c]
        try:
            return float(str(val).replace(",", "."))
        except Exception:
            continue
    return float("nan")


def label_match(row: pd.Series) -> str:
    """Classifica la partita in macro-bucket in base alle quote 1x2 pre-match.

    Label compatibili:
      - SuperCompetitive H<=3 A<=3
      - H_StrongFav <1.5 / A_StrongFav <1.5
      - H_MediumFav 1.5-2 / A_MediumFav 1.5-2
      - H_SmallFav 2-3   / A_SmallFav 2-3
      - Others
    """
    # 🔎 Alias molto ampi per intercettare colonne diverse nei vari dataset
    HOME_ALIASES = (
        "Odd home", "Odd Home", "cotaa", "oddhome", "homeodds",
        "odds1", "cota1", "home", "1"
    )
    AWAY_ALIASES = (
        "Odd Away", "Odd away", "cotad", "oddaway", "awayodds",
        "odds2", "cota2", "away", "2"
    )
    try:
        h = float(_get_odd(row, *HOME_ALIASES))
        a = float(_get_odd(row, *AWAY_ALIASES))
    except Exception:
        return "Others"

    if np.isnan(h) or np.isnan(a):
        return "Others"

    # Entrambe basse → match equilibrato e aperto
    if h <= 3 and a <= 3:
        return "SuperCompetitive H<=3 A<=3"

    # Favorite nette
    if h < 1.5:
        return "H_StrongFav <1.5"
    if a < 1.5:
        return "A_StrongFav <1.5"

    # Favorite medie
    if 1.5 <= h <= 2:
        return "H_MediumFav 1.5-2"
    if 1.5 <= a <= 2:
        return "A_MediumFav 1.5-2"

    # Favorite piccole
    if 2 < h <= 3:
        return "H_SmallFav 2-3"
    if 2 < a <= 3:
        return "A_SmallFav 2-3"

    return "Others"

# =====================================================
# Estrazione/normalizzazione minuti (utility leggera)
# =====================================================
def extract_minutes(series: pd.Series) -> List[int]:
    """Estrae i minuti da una serie di stringhe ("12;45;78" o "12,45").
    Ignora nulli e valori non numerici.
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
    """Ritorna la fonte dati condivisa (df + info).
    Se non presente, restituisce (DataFrame vuoto, {}).
    """
    return st.session_state.get(_GLOBAL_DF_KEY, pd.DataFrame()), st.session_state.get(_GLOBAL_INFO_KEY, {})


# =====================================================
# --------- Loader Parquet via DuckDB+HTTPFS ----------
# --------- con filtro server-side + cache 15' --------
# =====================================================
@st.cache_data(show_spinner=False, ttl=900)
def _read_parquet_filtered(parquet_url: str, league: str = "", seasons: Tuple[str, ...] = tuple()) -> pd.DataFrame:
    """Legge il Parquet remoto applicando i filtri direttamente a server (DuckDB).
    Fallback a pandas/pyarrow se DuckDB/HTTPFS non disponibili.
    Cache 15 minuti per combinazione (url, league, seasons).
    """
    # Via DuckDB
    if _DUCKDB_OK:
        try:
            con = duckdb.connect()
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
                httpfs_ok = True
            except Exception:
                httpfs_ok = False

            if parquet_url.startswith(("http://", "https://")) and not httpfs_ok:
                raise RuntimeError("HTTPFS non disponibile: fallback a pandas")

            # Costruzione query
            base = "SELECT * FROM read_parquet(?)"
            conds = []
            params: list[Any] = [parquet_url]
            if league:
                conds.append("LOWER(country) = LOWER(?)")
                params.append(league)
            if seasons:
                # IN su elenco stagioni
                placeholders = ",".join(["?"] * len(seasons))
                conds.append(f"CAST(sezonul AS VARCHAR) IN ({placeholders})")
                params.extend([str(s) for s in seasons])

            if conds:
                q = f"{base} WHERE " + " AND ".join(conds)
            else:
                q = base

            df = con.execute(q, params).df()
            con.close()
            return df
        except Exception:
            pass  # Fallback a pandas

    # Fallback: leggo e filtro in memoria
    try:
        df = pd.read_parquet(parquet_url, engine="pyarrow")
    except Exception:
        df = pd.read_parquet(parquet_url)
    if league and "country" in df.columns:
        df = df[df["country"].astype(str).str.lower() == str(league).lower()]
    if seasons and "sezonul" in df.columns:
        df = df[df["sezonul"].astype(str).isin([str(s) for s in seasons])]
    return df


def _duckdb_select_distinct(parquet_url: str, cols: Iterable[str]) -> pd.DataFrame:
    """Restituisce DISTINCT delle colonne richieste dal parquet (preferendo DuckDB)."""
    col_list = ", ".join(cols)
    if _DUCKDB_OK:
        try:
            con = duckdb.connect()
            try:
                con.execute("INSTALL httpfs; LOAD httpfs;")
            except Exception:
                pass
            q = f"SELECT DISTINCT {col_list} FROM read_parquet(?)"
            df = con.execute(q, [parquet_url]).df()
            con.close()
            return df
        except Exception:
            pass
    # Fallback
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
    """Carica i dati da Parquet (Supabase Storage).
    - ui_mode="minimal": legge subito i dati (eventualmente usando filtri globali se già impostati in session_state)
    - ui_mode="full"   : mostra in sidebar il selettore Campionato + Stagioni (server-side)
    Pubblica sempre la FONTE GLOBALE in session_state.
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
            help="Incolla l'URL (o path) al file parquet su Supabase Storage.",
        ).strip()

    if not parquet_url:
        st.info("Inserisci l'URL del parquet per continuare.")
        return pd.DataFrame(), "Supabase: (URL mancante)"

    # ---- modalità MINIMAL: usa eventuali filtri globali già presenti ----
    if ui_mode == "minimal":
        league  = st.session_state.get("global_country", "")
        seasons = tuple(st.session_state.get("global_seasons", []) or [])
        df = _read_parquet_filtered(parquet_url, league=league or "", seasons=seasons)
        _set_global_source(df, {"origin": "supabase", "url": parquet_url, "league": league, "seasons": list(seasons), "rows": len(df)})
        st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key=f"{selectbox_key}__rows", disabled=True)
        return df, f"Supabase: {parquet_url}"

    # ---- modalità FULL: prima leggo meta (league + seasons) ----
    meta = _duckdb_select_distinct(parquet_url, cols=("country", "sezonul"))
    leagues = sorted(meta["country"].dropna().astype(str).unique()) if "country" in meta.columns else []
    league = st.sidebar.selectbox(
        "Seleziona Campionato:",
        ["Tutti"] + leagues,
        index=1 if leagues else 0,
        key=selectbox_key,
    )

    if league != "Tutti" and "sezonul" in meta.columns and "country" in meta.columns:
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

    # Esponi l'elenco stagioni a chi vuole costruire preset (app.py)
    st.session_state["supabase_seasons_choices"] = seasons_all

    # Lettura filtrata (server-side) + cache
    df = _read_parquet_filtered(parquet_url, league if league != "Tutti" else "", tuple(seasons_sel))
    st.sidebar.checkbox(f"Righe caricate: {len(df):,}", value=True, key=f"{selectbox_key}__rows", disabled=True)

    # Pubblica FONTE GLOBALE per l'intera app
    _set_global_source(
        df,
        {
            "origin": "supabase",
            "url": parquet_url,
            "league": (league if league != "Tutti" else ""),
            "seasons": seasons_sel,
            "rows": len(df),
        },
    )
    return df, f"Supabase: {league if league != 'Tutti' else 'Tutti'}"


# -----------------------------------------------------
# Upload manuale (CSV/Excel/Parquet) con pulizia minima
# -----------------------------------------------------
def load_data_from_file(
    ui_mode: str = "minimal",
) -> Tuple[pd.DataFrame, str]:
    st.sidebar.markdown("**📂 Origine: Upload Manuale**")

    uploaded_file = st.sidebar.file_uploader(
        "Carica il tuo file Excel o CSV:",
        type=["xls", "xlsx", "csv", "parquet"],
        key="file_uploader_upload",
    )

    if uploaded_file is None:
        st.info("ℹ️ Carica un file per continuare.")
        return pd.DataFrame(), "Upload: (nessun file)"

    # Leggi
    if uploaded_file.name.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif uploaded_file.name.lower().endswith(".parquet"):
        df = pd.read_parquet(uploaded_file)
    else:
        # Excel
        try:
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        except Exception:
            df = pd.read_excel(uploaded_file)

    # Tipi base
    for col in ("cotaa", "cotad", "cotae", "Odd home", "Odd Away", "Odd Draw"):
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

    seasons_all = sorted(df_filtered["sezonul"].dropna().astype(str).unique()) if "sezonul" in df_filtered.columns else []
    seasons_sel = st.sidebar.multiselect(
        "Seleziona le stagioni:",
        options=seasons_all,
        default=seasons_all,
        key="multiselect_stagioni_upload",
    )

    if seasons_sel and "sezonul" in df_filtered.columns:
        df_filtered = df_filtered[df_filtered["sezonul"].astype(str).isin(seasons_sel)]

    st.sidebar.checkbox(f"Righe caricate da Upload: {len(df_filtered):,}", value=True, key="rows_loaded_upload", disabled=True)
    _set_global_source(df_filtered, {"origin": "upload", "name": uploaded_file.name, "league": league if league != "Tutti" else "", "seasons": seasons_sel, "rows": len(df_filtered)})
    return df_filtered, f"Upload: {uploaded_file.name}"
