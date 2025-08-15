# app.py — ProTrader Hub (Selezione globale in SIDEBAR con FORM + intervalli stagioni smart)
from __future__ import annotations

import os
import sys
import re
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import datetime as _dt

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
# Moduli principali (solo Hub Pre-Match)
# -------------------------------------------------------
_pre_match = load_local_module("pre_match", "pre_match.py")
run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")

# -------------------------------------------------------
# SUPPORTO
# -------------------------------------------------------
def _safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")

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

def _db_label_for_modules() -> str:
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    if champ:
        return str(champ)
    # fallback all'origine dati
    return "Dataset"

# -------------------------------------------------------
# ✨ HELPER stagioni: parsing e scelta intervalli smart
# -------------------------------------------------------
def _parse_season_start_year(season_str: str) -> int | None:
    """
    Ritorna l'anno di inizio stagione (YYYY) per stringhe tipo:
    '2024-25', '2024/2025', '2024-2025', '23/24', '2024', ecc.
    """
    if season_str is None:
        return None
    s = str(season_str)
    nums = re.findall(r"\d{2,4}", s)
    if not nums:
        return None
    first = nums[0]
    if len(first) == 4:
        return int(first)
    if len(first) == 2:
        yy = int(first)
        return 2000 + yy if yy <= 50 else 1900 + yy
    return None

def _current_season_start_year(tz: str = "Europe/Rome") -> int:
    # Regola: stagione calcistica inizia a luglio (7). Se mese >= 7 → stagione in corso = anno corrente.
    now = pd.Timestamp.now(tz=tz)
    return int(now.year if now.month >= 7 else now.year - 1)

def _seasons_desc_for_champ(df_champ: pd.DataFrame) -> list[str]:
    """Restituisce le stagioni presenti nel dataset (colonna 'Stagione') ordinate per start-year decrescente."""
    if "Stagione" not in df_champ.columns:
        return []
    uniq = [str(x) for x in df_champ["Stagione"].dropna().unique()]
    uniq = list(set(uniq))
    uniq.sort(key=lambda s: (_parse_season_start_year(s) or -1, s), reverse=True)
    return uniq

def _map_startyear_to_seasons(seasons_list: list[str]) -> dict[int, list[str]]:
    m: dict[int, list[str]] = {}
    for s in seasons_list:
        y = _parse_season_start_year(s)
        if y is None:
            continue
        m.setdefault(y, []).append(s)
    # ordina ogni lista per estetica (decrescente al solito)
    for y in m:
        m[y].sort(key=lambda s: (_parse_season_start_year(s) or -1, s), reverse=True)
    return m

def _select_seasons_by_mode(seasons_list: list[str], mode: str) -> list[str]:
    """
    Costruisce la selezione stagioni (stringhe esistenti nel dataset) in base alla modalità:
    - 'Stagione Corrente' → solo la stagione con start-year corrente (se presente)
    - 'Ultime 3/5/10' → start-year: [cur, cur-1, ...], includendo tutte le stringhe dataset che matchano questi anni
    - 'Personalizza' → verrà gestito con multiselect (qui non usato)
    """
    if not seasons_list:
        return []
    cur = _current_season_start_year()
    m = _map_startyear_to_seasons(seasons_list)

    def take_last(n: int) -> list[str]:
        years = [cur - i for i in range(n)]
        out = []
        for y in years:
            out.extend(m.get(y, []))
        return out

    if mode == "Stagione Corrente":
        return m.get(cur, [])
    if mode == "Ultime 3":
        return take_last(3)
    if mode == "Ultime 5":
        return take_last(5)
    if mode == "Ultime 10":
        return take_last(10)
    return []  # Personalizza è gestito sotto

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
def _concat_minutes(row: pd.Series, prefixes: list[str]) -> str:
    mins = []
    for p in prefixes:
        for i in range(1, 10):
            c = f"{p}{i}"
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

# La lista stagioni dipende dal campionato correntemente applicato (valore attuale in sessione)
df_tmp = df.copy()
if cur_champ and "country" in df_tmp.columns:
    df_tmp = df_tmp[df_tmp["country"].astype(str) == str(cur_champ)]
seasons_all_desc = _seasons_desc_for_champ(df_tmp)  # ordinamento decrescente per start-year

with st.sidebar.form("global_selection_form_sidebar", clear_on_submit=False):
    sel_champ = st.selectbox(
        "🏆 Campionato",
        options=champs,
        index=(champs.index(cur_champ) if cur_champ in champs else 0) if champs else 0,
        help="La selezione è globale (si applica a tutte le sezioni)."
    ) if champs else None

    # Modalità intervallo
    modes = ["Stagione Corrente", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"]
    mode = st.radio("Intervallo stagioni", modes, horizontal=True, index=0)

    sel_seasons = cur_seasons
    if mode == "Personalizza":
        sel_seasons = st.multiselect(
            "Seleziona stagioni (più recente in alto)",
            options=seasons_all_desc,
            default=cur_seasons if cur_seasons else seasons_all_desc[:5]
        )
    else:
        sel_seasons = _select_seasons_by_mode(seasons_all_desc, mode)

    applied = st.form_submit_button("✅ Applica")

if applied:
    if sel_champ is not None:
        st.session_state[GLOBAL_CHAMP_KEY] = sel_champ
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
# MENU: Solo Hub Pre-Match
# -------------------------------------------------------
st.sidebar.markdown("---")
menu_option = st.sidebar.radio(
    "Naviga",
    ["Pre-Match (Hub)"],
    key="menu_principale",
)

# -------------------------------------------------------
# ROUTING
# -------------------------------------------------------
if menu_option == "Pre-Match (Hub)":
    selection_badges()
    run_pre_match(apply_global_filters(df), _db_label_for_modules())
