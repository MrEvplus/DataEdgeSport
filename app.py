# app.py — ProTrader Hub (snellito ma con MAPPING & PULIZIA completi)
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
# utils.py come "utils" (per import interni)
# -------------------------------------------------------
_utils = load_local_module("app_utils", "utils.py")
sys.modules["utils"] = _utils

load_data_from_supabase = getattr(_utils, "load_data_from_supabase")
load_data_from_file     = getattr(_utils, "load_data_from_file")
label_match             = getattr(_utils, "label_match")

# -------------------------------------------------------
# Moduli principali
# -------------------------------------------------------
_pre_match            = load_local_module("pre_match", "pre_match.py")
_analisi_live_minuto  = load_local_module("analisi_live_minuto", "analisi_live_minuto.py")
_partite_del_giorno   = load_local_module("partite_del_giorno", "partite_del_giorno.py")
_reverse_engineering  = load_local_module("reverse_engineering", "reverse_engineering.py")

run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")
run_live_minuto_analysis = get_callable(
    _analisi_live_minuto,
    "run_live_minute_analysis", "run_live_minuto", "run_live", "main",
    label="analisi_live_minuto",
)
run_partite_del_giorno = get_callable(_partite_del_giorno, "run_partite_del_giorno", label="partite_del_giorno")
run_reverse_engineering = get_callable(_reverse_engineering, "run_reverse_engineering", label="reverse_engineering")

# -------------------------------------------------------
# Legacy opzionali dietro toggle
# -------------------------------------------------------
LEGACY_OK = True
try:
    _macros  = load_local_module("macros", "macros.py")
    _squadre = load_local_module("squadre", "squadre.py")
    _correct_score_ev_sezione = load_local_module("correct_score_ev_sezione", "correct_score_ev_sezione.py")
    run_macro_stats      = get_callable(_macros, "run_macro_stats", label="macros")
    run_team_stats       = get_callable(_squadre, "run_team_stats", label="squadre")
    run_correct_score_ev = get_callable(_correct_score_ev_sezione, "run_correct_score_ev", label="correct_score_ev_sezione")
except Exception:
    LEGACY_OK = False

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
    Ordina stagioni estraendo l'ANNO PIÙ RECENTE presente nella stringa.
    Supporta: '2024-2025', '2024/25', '2024–25', '2025', '24/25', ecc.
    """
    if x is None:
        return -1
    s = str(x)
    nums = re.findall(r"\d{4}|\d{2}", s)
    if not nums:
        try: return int(s)
        except Exception: return -1
    vals = [(int(n) if len(n)==4 else 2000+int(n)) for n in nums]
    return max(vals)

def _concat_minutes(row: pd.Series, prefixes: list[str]) -> str:
    """
    Se nel dataset esistono colonne 'gh1..gh9'/'ga1..ga9' le concatena in stringa 'm1,m2,...'
    Utile a ricostruire 'minuti goal segnato home/away' se mancanti.
    """
    mins = []
    for p in prefixes:
        for i in range(1, 10):
            c = f"{p}{i}"
            if c in row and pd.notna(row[c]) and str(row[c]).strip() not in ("", "nan", "None"):
                try:
                    v = int(float(str(row[c]).replace(",", ".")))
                    if v > 0:
                        mins.append(v)
                except Exception:
                    continue
    mins.sort()
    return ",".join(str(m) for m in mins)

# -------------------------------------------------------
# CONFIG PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="ProTrader — Hub", page_icon="⚽", layout="wide")
st.sidebar.title("⚽ ProTrader — Hub")

# -------------------------------------------------------
# ORIGINE DATI
# -------------------------------------------------------
origine_dati = st.sidebar.radio("Origine dati", ["Supabase", "Upload Manuale"], key="origine_dati")
if origine_dati == "Supabase":
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase")
else:
    df, db_selected = load_data_from_file()

# -------------------------------------------------------
# MAPPING COLONNE E PULIZIA (esteso)
# -------------------------------------------------------
col_map = {
    # anagrafiche/contesto
    "country": "country",
    "sezonul": "Stagione",
    "datameci": "Data",
    "orameci": "Orario",
    "etapa": "Round",
    "txtechipa1": "Home",
    "txtechipa2": "Away",

    # punteggi
    "scor1": "Home Goal FT",
    "scor2": "Away Goal FT",
    "scorp1": "Home Goal 1T",
    "scorp2": "Away Goal 1T",

    # posizioni classifica
    "place1":  "Posizione Classifica Generale",
    "place1a": "Posizione Classifica Home",
    "place2":  "Posizione Classifica Away Generale",
    "place2d": "Posizione classifica away",

    # odds principali
    "cotaa":  "Odd home",
    "cotad":  "Odd Away",
    "cotae":  "Odd Draw",
    # over/under
    "cotao0": "Odd Over 0.5",
    "cotao1": "Odd Over 1.5",
    "cotao":  "Odd Over 2.5",
    "cotao3": "Odd Over 3.5",
    "cotao4": "Odd Over 4.5",
    "cotau0": "Odd Under 0.5",
    "cotau1": "Odd Under 1.5",
    "cotau":  "Odd Under 2.5",
    "cotau3": "Odd Under 3.5",
    "cotau4": "Odd Under 4.5",

    # btts
    "gg": "GG",
    "ng": "NG",

    # ELO / forma
    "elohomeo": "ELO Home",
    "eloawayo": "ELO Away",
    "formah": "Form Home",
    "formaa": "Form Away",

    # tiri totali e nello specchio
    "suth":  "Tiri Totali Home FT",
    "suth1": "Tiri Home 1T",
    "suth2": "Tiri Home 2T",
    "suta":  "Tiri Totali Away FT",
    "suta1": "Tiri Away 1T",
    "suta2": "Tiri Away 2T",
    "sutht":  "Tiri in Porta Home FT",
    "sutht1": "Tiri in Porta Home 1T",
    "sutht2": "Tiri in Porta Home 2T",
    "sutat":  "Tiri in Porta Away FT",
    "sutat1": "Tiri in Porta Away 1T",
    "sutat2": "Tiri in Porta Away 2T",

    # minuti goal — home (serie gh1..gh9) / away (ga1..ga9)
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

    # altri
    "stare": "Stare",
    "codechipa1": "CodeChipa1",
    "codechipa2": "CodeChipa2",
}
df = df.rename(columns=col_map)

# normalizzazione nomi colonne (spazi/CR/LF)
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.replace(r"[\n\r\t]", "", regex=True)
      .str.replace(r"\s+", " ", regex=True)
)

# etichetta "Label" (se manca) usando odds principali
if "Label" not in df.columns:
    if {"Odd home", "Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

# ricostruzione dei campi "minuti goal segnato ..." se non presenti
if "minuti goal segnato home" not in df.columns:
    if set([f"gh{i}" for i in range(1,10)]).issubset({c.lower(): c for c in df.columns.str.lower()}):
        df["minuti goal segnato home"] = df.apply(lambda r: _concat_minutes(r, ["gh"]), axis=1)
if "minuti goal segnato away" not in df.columns:
    if set([f"ga{i}" for i in range(1,10)]).issubset({c.lower(): c for c in df.columns.str.lower()}):
        df["minuti goal segnato away"] = df.apply(lambda r: _concat_minutes(r, ["ga"]), axis=1)

# tipi utili
for c in ["Home Goal FT","Away Goal FT","Home Goal 1T","Away Goal 1T"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# -------------------------------------------------------
# FILTRO STAGIONI (preset + personalizza) — ordinamento robusto
# -------------------------------------------------------
if "Stagione" in df.columns:
    stag_raw = df["Stagione"].dropna().astype(str).unique().tolist()
    stagioni_disponibili = sorted(stag_raw, key=_season_sort_key, reverse=True)

    opzione_range = st.sidebar.selectbox(
        "Intervallo stagioni",
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
            "Scegli stagioni",
            options=stagioni_disponibili,
            default=stagioni_disponibili[:5],
            key="multiselect_stagioni_personalizzate",
        )
    if stagioni_scelte:
        df = df[df["Stagione"].astype(str).isin(stagioni_scelte)]

# -------------------------------------------------------
# GUARD-RAIL
# -------------------------------------------------------
with st.expander("✅ Colonne presenti", expanded=False):
    st.write(list(df.columns))

if df.empty:
    st.error("⚠️ Nessun dato disponibile dopo i filtri applicati.")
    st.stop()
if "Home" not in df.columns or "Away" not in df.columns:
    st.error("⚠️ Colonne 'Home' e/o 'Away' mancanti nel dataset selezionato.")
    st.stop()

# rimuovi partite future se presente 'Data'
if "Data" in df.columns:
    df["Data"] = _safe_to_datetime(df["Data"])
    today = pd.Timestamp.today().normalize()
    df = df[(df["Data"].isna()) | (df["Data"] <= today)]

# -------------------------------------------------------
# KPI rapidi
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Partite DB", f"{len(df):,}")
if "country" in df.columns: c2.metric("Campionati", df["country"].nunique())
if "Home" in df.columns: c3.metric("Squadre", pd.concat([df["Home"], df["Away"]]).nunique())
if "Stagione" in df.columns: c4.metric("Stagioni", df["Stagione"].nunique())

st.caption(f"Campionato selezionato: **{db_selected}**")

# -------------------------------------------------------
# MENU: Hub snello + toggle legacy
# -------------------------------------------------------
st.sidebar.markdown("---")
show_legacy = st.sidebar.checkbox("Mostra strumenti avanzati (legacy)")

PAGINE_BASE = [
    "Pre-Match (Hub)",
    "Analisi Live da Minuto",
    "Partite del Giorno",
    "🧠 Reverse Engineering EV+",
]
PAGINE_LEGACY = [
    "Macro Stats per Campionato",
    "Statistiche per Squadre",
    "Correct Score EV",
] if (show_legacy and LEGACY_OK) else []

menu_option = st.sidebar.radio(
    "Naviga",
    PAGINE_BASE + (["— Strumenti legacy —"] + PAGINE_LEGACY if PAGINE_LEGACY else []),
    key="menu_principale",
)

# -------------------------------------------------------
# ROUTING
# -------------------------------------------------------
if menu_option == "Pre-Match (Hub)":
    run_pre_match(df, db_selected)

elif menu_option == "Analisi Live da Minuto":
    run_live_minuto_analysis(df)

elif menu_option == "Partite del Giorno":
    run_partite_del_giorno(df, db_selected)

elif menu_option == "🧠 Reverse Engineering EV+":
    run_reverse_engineering(df)

# Legacy opzionali
elif menu_option == "Macro Stats per Campionato" and LEGACY_OK:
    run_macro_stats(df, db_selected)

elif menu_option == "Statistiche per Squadre" and LEGACY_OK:
    run_team_stats(df, db_selected)

elif menu_option == "Correct Score EV" and LEGACY_OK:
    run_correct_score_ev(df, db_selected)
