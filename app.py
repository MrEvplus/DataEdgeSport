# app.py
from __future__ import annotations

import os
import sys
import re
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np

# -------------------------------------------------------
# Loader robusto per moduli locali (evita conflitti nome)
# -------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)

def load_local_module(module_name: str, filename: str):
    """Carica un modulo Python locale da filename e lo registra in sys.modules[module_name]."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        st.error(f"Modulo mancante: {filename}")
        st.stop()
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        st.error(f"Impossibile creare lo spec per {filename}")
        st.stop()
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    except Exception as e:
        st.error(f"Errore nell'import di {filename}: {type(e).__name__} — {e}")
        st.stop()
    return mod

def get_callable(mod, *names, label: str = ""):
    """Ritorna la prima funzione disponibile tra i nomi proposti; altrimenti errore con diagnostica."""
    for n in names:
        if hasattr(mod, n) and callable(getattr(mod, n)):
            if n != names[0] and label:
                st.info(f"ℹ️ In '{label}' uso fallback: `{n}`")
            return getattr(mod, n)
    run_like = [a for a in dir(mod) if a.startswith("run") and callable(getattr(mod, a))]
    raise AttributeError(
        f"Nessuna delle funzioni {names} trovata nel modulo '{mod.__name__}'. "
        f"Funzioni disponibili simili: {', '.join(run_like) or '—'}"
    )

# -------------------------------------------------------
# Carico utils.py in modo speciale e forzo il nome "utils"
# -------------------------------------------------------
_utils = load_local_module("app_utils", "utils.py")
sys.modules["utils"] = _utils  # forza gli import interni

# Estraggo i simboli che servono da utils
SUPABASE_URL = getattr(_utils, "SUPABASE_URL", "")
SUPABASE_KEY = getattr(_utils, "SUPABASE_KEY", "")
load_data_from_supabase = getattr(_utils, "load_data_from_supabase")
load_data_from_file = getattr(_utils, "load_data_from_file")
label_match = getattr(_utils, "label_match")

# -------------------------------------------------------
# Carico tutti i moduli locali con loader esplicito
# -------------------------------------------------------
_macros = load_local_module("macros", "macros.py")
_squadre = load_local_module("squadre", "squadre.py")
_pre_match = load_local_module("pre_match", "pre_match.py")
_correct_score_ev_sezione = load_local_module("correct_score_ev_sezione", "correct_score_ev_sezione.py")
_analisi_live_minuto = load_local_module("analisi_live_minuto", "analisi_live_minuto.py")
_partite_del_giorno = load_local_module("partite_del_giorno", "partite_del_giorno.py")
_reverse_engineering = load_local_module("reverse_engineering", "reverse_engineering.py")

# Funzioni di sezione (con fallback dove serve)
run_macro_stats = get_callable(_macros, "run_macro_stats", label="macros")
run_team_stats = get_callable(_squadre, "run_team_stats", label="squadre")
run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")
run_correct_score_ev = get_callable(_correct_score_ev_sezione, "run_correct_score_ev", label="correct_score_ev_sezione")
run_live_minuto_analysis = get_callable(
    _analisi_live_minuto,
    "run_live_minute_analysis",   # principale
    "run_live_minuto_analysis",   # variante IT
    "run_live_minute",
    "run_live_minuto",
    "run_live",
    "main",
    label="analisi_live_minuto",
)
run_partite_del_giorno = get_callable(_partite_del_giorno, "run_partite_del_giorno", label="partite_del_giorno")
run_reverse_engineering = get_callable(_reverse_engineering, "run_reverse_engineering", label="reverse_engineering")

# -------------------------------------------------------
# SUPPORTO
# -------------------------------------------------------
def _safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")

def _season_sort_key(x: str) -> int:
    """
    Estrarre una chiave numerica per l'ordinamento stagioni.
    Supporta formati: '2025', '2024-2025', '2024/25', '2024–25'.
    Ritorna l'ANNO PIÙ RECENTE presente nella stringa (es. 2025).
    """
    if x is None:
        return -1
    s = str(x)
    # cattura 4 cifre o 2 cifre
    nums = re.findall(r"\d{4}|\d{2}", s)
    if not nums:
        # tenta cast diretto
        try:
            return int(s)
        except Exception:
            return -1
    vals = []
    for n in nums:
        if len(n) == 4:
            vals.append(int(n))
        else:
            # 2 cifre -> 2000+aa (es. '25' -> 2025)
            vals.append(2000 + int(n))
    return max(vals)

# -------------------------------------------------------
# CONFIGURAZIONE PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="Trading Dashboard", page_icon="⚽", layout="wide")
st.sidebar.title("📊 Trading Dashboard")

# -------------------------------------------------------
# MENU PRINCIPALE
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
    ],
    key="menu_principale",
)

# -------------------------------------------------------
# SELEZIONE ORIGINE DATI
# -------------------------------------------------------
origine_dati = st.sidebar.radio("Seleziona origine dati:", ["Supabase", "Upload Manuale"], key="origine_dati")

if origine_dati == "Supabase":
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase")
else:
    df, db_selected = load_data_from_file()

# -------------------------------------------------------
# MAPPING COLONNE E PULIZIA
# -------------------------------------------------------
col_map = {
    "country": "country",
    "sezonul": "Stagione",
    "datameci": "Data",
    "orameci": "Orario",
    "etapa": "Round",
    "txtechipa1": "Home",
    "txtechipa2": "Away",
    "scor1": "Home Goal FT",
    "scor2": "Away Goal FT",
    "scorp1": "Home Goal 1T",
    "scorp2": "Away Goal 1T",
    "place1": "Posizione Classifica Generale",
    "place1a": "Posizione Classifica Home",
    "place2": "Posizione Classifica Away Generale",
    "place2d": "Posizione classifica away",
    "cotaa": "Odd home",
    "cotad": "Odd Away",
    "cotae": "Odd Draw",
    "cotao0": "Odd Over 0.5",
    "cotao1": "Odd Over 1.5",
    "cotao": "Odd Over 2.5",
    "cotao3": "Odd Over 3.5",
    "cotao4": "Odd Over 4.5",
    "cotau0": "Odd Under 0.5",
    "cotau1": "Odd Under 1.5",
    "cotau": "Odd Under 2.5",
    "cotau3": "Odd Under 3.5",
    "cotau4": "Odd Under 4.5",
    "gg": "GG",
    "ng": "NG",
    "elohomeo": "ELO Home",
    "eloawayo": "ELO Away",
    "formah": "Form Home",
    "formaa": "Form Away",
    "suth": "Tiri Totali Home FT",
    "suth1": "Tiri Home 1T",
    "suth2": "Tiri Home 2T",
    "suta": "Tiri Totali Away FT",
    "suta1": "Tiri Away 1T",
    "suta2": "Tiri Away 2T",
    "sutht": "Tiri in Porta Home FT",
    "sutht1": "Tiri in Porta Home 1T",
    "sutht2": "Tiri in Porta Home 2T",
    "sutat": "Tiri in Porta Away FT",
    "sutat1": "Tiri in Porta Away 1T",
    "sutat2": "Tiri in Porta Away 2T",
    "mgolh": "Minuti Goal Home",
    "gh1": "Home Goal 1 (min)",
    "gh2": "Home Goal 2 (min)",
    "gh3": "Home Goal 3 (min)",
    "gh4": "Home Goal 4 (min)",
    "gh5": "Home Goal 5 (min)",
    "gh6": "Home Goal 6 (min)",
    "gh7": "Home Goal 7 (min)",
    "gh8": "Home Goal 8 (min)",
    "gh9": "Home Goal 9 (min)",
    "mgola": "Minuti Goal Away",
    "ga1": "Away Goal 1 (min)",
    "ga2": "Away Goal 2 (min)",
    "ga3": "Away Goal 3 (min)",
    "ga4": "Away Goal 4 (min)",
    "ga5": "Away Goal 5 (min)",
    "ga6": "Away Goal 6 (min)",
    "ga7": "Away Goal 7 (min)",
    "ga8": "Away Goal 8 (min)",
    "ga9": "Away Goal 9 (min)",
    "stare": "Stare",
    "codechipa1": "CodeChipa1",
    "codechipa2": "CodeChipa2",
}
df = df.rename(columns=col_map)

# Normalizzazione leggera dei nomi
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[\n\r\t]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

# Etichetta "Label" se manca
if "Label" not in df.columns:
    if {"Odd home", "Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

# -------------------------------------------------------
# FILTRO STAGIONI (preset + personalizza) — FIX ORDINAMENTO
# -------------------------------------------------------
if "Stagione" in df.columns:
    stag_raw = df["Stagione"].dropna().astype(str).unique().tolist()
    # ordina per anno più recente dentro la stringa, in modo DESC
    stagioni_disponibili = sorted(stag_raw, key=_season_sort_key, reverse=True)

    opzione_range = st.sidebar.selectbox(
        "Seleziona un intervallo stagioni predefinito:",
        ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"],
        key="selettore_range_stagioni",
    )

    if opzione_range == "Tutte":
        stagioni_scelte = stagioni_disponibili
    elif opzione_range == "Ultime 3":
        stagioni_scelte = stagioni_disponibili[:3]
    elif opzione_range == "Ultime 5":
        stagioni_scelte = stagioni_disponibili[:5]
    elif opzione_range == "Ultime 10":
        stagioni_scelte = stagioni_disponibili[:10]
    else:
        stagioni_scelte = st.sidebar.multiselect(
            "Seleziona manualmente le stagioni:",
            options=stagioni_disponibili,
            default=stagioni_disponibili[:5],
            key="multiselect_stagioni_personalizzate",
        )

    if stagioni_scelte:
        df = df[df["Stagione"].astype(str).isin(stagioni_scelte)]

# -------------------------------------------------------
# INFO DATASET & GUARD-RAIL
# -------------------------------------------------------
with st.expander("✅ Colonne presenti nel dataset", expanded=False):
    st.write(list(df.columns))

if df.empty:
    st.error("⚠️ Nessun dato disponibile dopo i filtri applicati.")
    st.stop()

if "Home" not in df.columns or "Away" not in df.columns:
    st.error("⚠️ Colonne 'Home' e/o 'Away' mancanti nel dataset selezionato.")
    st.stop()

# Se c'è "Data", rimuovi partite future
if "Data" in df.columns:
    df["Data"] = _safe_to_datetime(df["Data"])
    today = pd.Timestamp.today().normalize()
    df = df[(df["Data"].isna()) | (df["Data"] <= today)]

# KPI rapidi
c1, c2, c3, c4 = st.columns(4)
c1.metric("Partite DB", f"{len(df):,}")
if "country" in df.columns: c2.metric("Campionati", df["country"].nunique())
if "Home" in df.columns: c3.metric("Squadre", pd.concat([df["Home"], df["Away"]]).nunique())
if "Stagione" in df.columns: c4.metric("Stagioni", df["Stagione"].nunique())
st.caption(f"Campionato selezionato: **{db_selected}**")

# -------------------------------------------------------
# ROUTING SEZIONI
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
    run_live_minuto_analysis(df)

elif menu_option == "Partite del Giorno":
    run_partite_del_giorno(df, db_selected)

elif menu_option == "🧠 Reverse Engineering EV+":
    run_reverse_engineering(df)
