# app.py — ProTrader Hub (Selezione globale in SIDEBAR con FORM + niente rerun su ogni input)
from __future__ import annotations

import os
import sys
import re
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np

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
# Moduli principali
# -------------------------------------------------------
_pre_match            = load_local_module("pre_match", "pre_match.py")
_analisi_live_minuto  = load_local_module("analisi_live_minuto", "analisi_live_minuto.py")
_partite_del_giorno   = load_local_module("partite_del_giorno", "partite_del_giorno.py")
_reverse_engineering  = load_local_module("reverse_engineering", "reverse_engineering.py")

run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")

# ✅ includo sia run_live_minuto_analysis che run_live_minute_analysis (fallback)
run_live_minuto_analysis = get_callable(
    _analisi_live_minuto,
    "run_live_minuto_analysis", "run_live_minute_analysis",
    "run_live_minuto", "run_live", "main",
    label="analisi_live_minuto",
)

run_partite_del_giorno  = get_callable(_partite_del_giorno, "run_partite_del_giorno", label="partite_del_giorno")
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
    mins = []
    for p in prefixes:
        for i in range(1, 10):
            c = f"{p}{i}"
            # supporta sia colonne lowercase che miste
            for col in (c, c.upper(), c.capitalize()):
                if col in row and pd.notna(row[col]) and str(row[col]).strip() not in ("", "nan", "None"):
                    try:
                        v = int(float(str(row[col]).replace(",", ".")))
                        if v > 0:
                            mins.append(v)
                    except Exception:
                        continue
    mins.sort()
    return ",".join(str(m) for m in mins)

# —— FILTRI GLOBALI (scelti in sidebar) ——
GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"

def get_global_filters():
    return (
        st.session_state.get(GLOBAL_CHAMP_KEY),
        st.session_state.get(GLOBAL_SEASONS_KEY),
    )

def apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    champ, seasons = get_global_filters()
    out = df.copy()
    if champ and "country" in out.columns:
        out = out[out["country"].astype(str) == str(champ)]
    if seasons and "Stagione" in out.columns:
        out = out[out["Stagione"].astype(str).isin([str(s) for s in seasons])]
    return out

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

# -------------------------------------------------------
# CONFIG PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="ProTrader — Hub", page_icon="⚽", layout="wide")
st.sidebar.title("⚽ ProTrader — Hub")

# -------------------------------------------------------
# ORIGINE DATI (sidebar)
# -------------------------------------------------------
origine_dati = st.sidebar.radio("Origine dati", ["Supabase", "Upload Manuale"], key="origine_dati")
if origine_dati == "Supabase":
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase", ui_mode="minimal")
else:
    df, db_selected = load_data_from_file(ui_mode="minimal")

# -------------------------------------------------------
# MAPPING COLONNE E PULIZIA (esteso, invariato)
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

# Etichette (una sola volta se possibile)
if "Label" not in df.columns:
    if {"Odd home", "Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

# Normalizza colonne minuti-gol se abbiamo gh*/ga* a disposizione
lower_cols = {c.lower(): c for c in df.columns}
has_all_gh = all(f"gh{i}" in lower_cols for i in range(1, 10))
has_all_ga = all(f"ga{i}" in lower_cols for i in range(1, 10))
if "minuti goal segnato home" not in df.columns and has_all_gh:
    df["minuti goal segnato home"] = df.apply(lambda r: _concat_minutes(r, ["gh"]), axis=1)
if "minuti goal segnato away" not in df.columns and has_all_ga:
    df["minuti goal segnato away"] = df.apply(lambda r: _concat_minutes(r, ["ga"]), axis=1)

# Cast numerici basilari
for c in ["Home Goal FT","Away Goal FT","Home Goal 1T","Away Goal 1T"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Limita lo storico a oggi (per KPI puliti) se c'è la colonna Data
if "Data" in df.columns:
    df["Data"] = _safe_to_datetime(df["Data"])
    today = pd.Timestamp.today().normalize()
    df = df[(df["Data"].isna()) | (df["Data"] <= today)]

# -------------------------------------------------------
# SIDEBAR: Selezione GLOBALE con FORM (nessun rerun finché non premi)
# -------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Selezione globale")

# Opzioni disponibili
champs = sorted(df["country"].dropna().astype(str).unique()) if "country" in df.columns else []
cur_champ, cur_seasons = get_global_filters()
if cur_champ is None and champs:
    st.session_state[GLOBAL_CHAMP_KEY] = champs[0]
    cur_champ = champs[0]

# La lista stagioni dipende dal campionato correntemente applicato
df_tmp = df.copy()
if cur_champ and "country" in df_tmp.columns:
    df_tmp = df_tmp[df_tmp["country"].astype(str) == str(cur_champ)]

seasons_all = []
if "Stagione" in df_tmp.columns:
    seasons_all = sorted(df_tmp["Stagione"].dropna().astype(str).unique(),
                         key=_season_sort_key, reverse=True)
    if cur_seasons is None:
        st.session_state[GLOBAL_SEASONS_KEY] = seasons_all
        cur_seasons = seasons_all

with st.sidebar.form("global_selection_form_sidebar", clear_on_submit=False):
    sel_champ = st.selectbox(
        "🏆 Campionato",
        options=champs,
        index=(champs.index(cur_champ) if cur_champ in champs else 0) if champs else 0,
        help="La selezione è globale (si applica a tutte le sezioni)."
    ) if champs else None

    mode = "Tutte"
    if seasons_all:
        mode = st.radio(
            "Intervallo stagioni",
            ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"],
            horizontal=True
        )

    sel_seasons = cur_seasons
    if seasons_all:
        if mode == "Tutte":
            sel_seasons = seasons_all
        elif mode == "Ultime 3":
            sel_seasons = seasons_all[:3]
        elif mode == "Ultime 5":
            sel_seasons = seasons_all[:5]
        elif mode == "Ultime 10":
            sel_seasons = seasons_all[:10]
        else:
            sel_seasons = st.multiselect(
                "Seleziona stagioni",
                options=seasons_all,
                default=cur_seasons if cur_seasons else seasons_all[:5]
            )

    applied = st.form_submit_button("✅ Applica")

if applied:
    if sel_champ is not None:
        st.session_state[GLOBAL_CHAMP_KEY] = sel_champ
        # Nota: la lista stagioni verrà ricalcolata al prossimo rerun in base al nuovo campionato
    st.session_state[GLOBAL_SEASONS_KEY] = sel_seasons

# -------------------------------------------------------
# KPI rapidi (sul dataset filtrato se già scelto)
# -------------------------------------------------------
df_for_kpi = apply_global_filters(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Partite DB", f"{len(df_for_kpi):,}")
if "country" in df_for_kpi.columns: c2.metric("Campionati", df_for_kpi["country"].nunique())
if "Home" in df_for_kpi.columns: c3.metric("Squadre", pd.concat([df_for_kpi["Home"], df_for_kpi["Away"]]).nunique())
if "Stagione" in df_for_kpi.columns: c4.metric("Stagioni", df_for_kpi["Stagione"].nunique())

db_short = _short_origin_label(str(db_selected))
if db_short:
    st.caption(f"Origine: **{db_short}**")

# -------------------------------------------------------
# MENU: Hub + toggle legacy
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
# Utility (badge selezione) e label per moduli
# -------------------------------------------------------
def selection_badges():
    champ, seasons = get_global_filters()
    txt_champ = f"🏆 <b>{champ}</b>" if champ else "🏷️ nessun campionato selezionato"
    txt_seas  = ", ".join([str(s) for s in seasons]) if seasons else "tutte le stagioni"
    st.markdown(
        f"<div style='margin:.25rem 0 .75rem 0;display:flex;gap:.5rem;flex-wrap:wrap'>"
        f"<span style='border:1px solid #e57e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>{txt_champ}</span>"
        f"<span style='border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>🗓️ <b>Stagioni:</b> {txt_seas}</span>"
        f"</div>",
        unsafe_allow_html=True
    )

def _db_label_for_modules() -> str:
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    if champ:
        return str(champ)
    return _short_origin_label(str(db_selected)) or "Dataset"

# -------------------------------------------------------
# ROUTING
# -------------------------------------------------------
if menu_option == "Pre-Match (Hub)":
    selection_badges()
    # Ora la selezione è in sidebar: passiamo semplicemente il df filtrato
    run_pre_match(apply_global_filters(df), _db_label_for_modules())

elif menu_option == "Analisi Live da Minuto":
    champ, _ = get_global_filters()
    if not champ:
        st.warning("Seleziona prima il **Campionato** nella sidebar.")
    selection_badges()
    run_live_minuto_analysis(apply_global_filters(df))

elif menu_option == "Partite del Giorno":
    champ, _ = get_global_filters()
    if not champ:
        st.warning("Seleziona prima il **Campionato** nella sidebar.")
    selection_badges()
    run_partite_del_giorno(apply_global_filters(df), _db_label_for_modules())

elif menu_option == "🧠 Reverse Engineering EV+":
    champ, _ = get_global_filters()
    if not champ:
        st.warning("Seleziona prima il **Campionato** nella sidebar.")
    selection_badges()
    run_reverse_engineering(apply_global_filters(df))

elif menu_option == "Macro Stats per Campionato" and LEGACY_OK:
    selection_badges()
    run_macro_stats(apply_global_filters(df), _db_label_for_modules())

elif menu_option == "Statistiche per Squadre" and LEGACY_OK:
    selection_badges()
    run_team_stats(apply_global_filters(df), _db_label_for_modules())

elif menu_option == "Correct Score EV" and LEGACY_OK:
    selection_badges()
    run_correct_score_ev(apply_global_filters(df), _db_label_for_modules())
