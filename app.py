# app.py — ProTrader Hub (Supabase filtrato: scegli LEGA+STAGIONI prima di leggere il parquet)
from __future__ import annotations

import os
import sys
import importlib.util
import streamlit as st
import pandas as pd

# -------------------------------------------------------
# Loader robusto per moduli locali
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
# utils.py come "utils"
# -------------------------------------------------------
_app_utils = load_local_module("app_utils", "utils.py")
# Esponi come "utils" per compatibilità con altri moduli
sys.modules["utils"] = _app_utils

load_data_from_supabase = getattr(_app_utils, "load_data_from_supabase")
load_data_from_file     = getattr(_app_utils, "load_data_from_file")
label_match             = getattr(_app_utils, "label_match")

# -------------------------------------------------------
# Modulo principale (Pre-Match Hub)
# -------------------------------------------------------
_pre_match = load_local_module("pre_match", "pre_match.py")
run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")

# -------------------------------------------------------
# SUPPORTO
# -------------------------------------------------------
def _short_origin_label(s: str) -> str:
    if not s:
        return ""
    sl = s.lower()
    if sl.startswith("supabase"):
        return "Supabase"
    for sep in ("/", "\\"):
        if sep in s:
            s = s.split(sep)[-1]
    return s[:60]

# —— FILTRI GLOBALI condivisi tra moduli ——
GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"

def get_global_filters():
    return (
        st.session_state.get(GLOBAL_CHAMP_KEY),
        st.session_state.get(GLOBAL_SEASONS_KEY),
    )

def selection_badges():
    champ, seasons = get_global_filters()
    txt_champ = f"🏆 <b>{champ}</b>" if champ else "🏷️ nessun campionato selezionato"
    txt_seas  = ", ".join([str(s) for s in seasons]) if seasons else "tutte le stagioni"
    st.markdown(
        f"<div style='margin:.25rem 0 .75rem 0;display:flex;gap:.5rem;flex-wrap:wrap'>"
        f"<span style='border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>{txt_champ}</span>"
        f"<span style='border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>🗓️ <b>Stagioni:</b> {txt_seas}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

def _db_label_for_modules() -> str:
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    return str(champ) if champ else "Dataset"

# -------------------------------------------------------
# CONFIG PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="ProTrader — Hub", page_icon="⚽", layout="wide")
st.sidebar.title("⚽ ProTrader — Hub")

# -------------------------------------------------------
# ORIGINE DATI
#  - Supabase: ui_mode="full" → OBBLIGO scegliere Campionato/Stagioni PRIMA della lettura
#  - Upload:   resta minimal (file completo), poi si filtra a valle
# -------------------------------------------------------
origine_dati = st.sidebar.radio("Origine dati", ["Supabase", "Upload Manuale"], key="origine_dati")

if origine_dati == "Supabase":
    # UI integrata nei selettori di utils (lega + stagioni) e lettura FILTRATA server-side (cache 15')
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase", ui_mode="full", show_url_input=True)

    # Allinea i filtri GLOBALI con la scelta fatta nei selettori Supabase (stessi key di utils)
    chosen_league  = st.session_state.get("campionato_supabase")
    chosen_seasons = st.session_state.get("campionato_supabase__seasons", [])
    if chosen_league:
        st.session_state[GLOBAL_CHAMP_KEY] = str(chosen_league)
    st.session_state[GLOBAL_SEASONS_KEY] = list(chosen_seasons) if isinstance(chosen_seasons, (list, tuple)) else []

else:
    df, db_selected = load_data_from_file(ui_mode="minimal")
    # Se vuoi forzare qui selezione campionato/stagioni anche per upload, possiamo aggiungere un mini pannello locale.

# -------------------------------------------------------
# MAPPING COLONNE COMPLETO & LABEL di base
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
    "place1":  "Posizione Classifica Generale",
    "place1a": "Posizione Classifica Home",
    "place2":  "Posizione Classifica Away Generale",
    "place2d": "Posizione classifica away",
    "cotaa":  "Odd home",
    "cotad":  "Odd Away",
    "cotae":  "Odd Draw",
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
    "gg": "GG",
    "ng": "NG",
    "elohomeo": "ELO Home",
    "eloawayo": "ELO Away",
    "formah": "Form Home",
    "formaa": "Form Away",
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
df.columns = (
    df.columns.astype(str)
      .str.strip()
      .str.replace(r"[\n\r\t]", "", regex=True)
      .str.replace(r"\s+", " ", regex=True)
)

if "Label" not in df.columns:
    if {"Odd home", "Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

# -------------------------------------------------------
# HEADER & BADGES
# -------------------------------------------------------
st.title("📊 Pre-Match — Hub")
selection_badges()
st.caption(f"Origine dati: **{_short_origin_label(db_selected)}** — righe caricate: **{len(df):,}**")

# -------------------------------------------------------
# MENU (solo Hub)
# -------------------------------------------------------
st.sidebar.markdown("---")
menu_option = st.sidebar.radio("Naviga", ["Pre-Match (Hub)"], key="menu_principale")

# -------------------------------------------------------
# ROUTING
# -------------------------------------------------------
if menu_option == "Pre-Match (Hub)":
    # Il dataset è già filtrato in lettura (Supabase full) — passiamo dritti al modulo principale
    run_pre_match(df, _db_label_for_modules())
