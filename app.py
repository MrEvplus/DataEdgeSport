# app.py — ProTrader Hub (pulito) + sezioni: Pre-Match | 📅 Upcoming (modulo separato)
from __future__ import annotations

import os, sys, re, importlib.util, unicodedata
from datetime import datetime
import streamlit as st
import pandas as pd

# ---------- loader moduli locali ----------
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
            return getattr(mod, n)
    raise AttributeError(f"Nessuna delle funzioni {names} trovata nel modulo '{mod.__name__}'.")

# ---------- import utils / pre_match / upcoming ----------
_app_utils = load_local_module("app_utils", "utils.py")
load_data_from_supabase = getattr(_app_utils, "load_data_from_supabase")
load_data_from_file     = getattr(_app_utils, "load_data_from_file")
label_match             = getattr(_app_utils, "label_match")

_pre_match = load_local_module("pre_match", "pre_match.py")
run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")

_minutes = load_local_module("minutes_mod", "minutes.py")
unify_goal_minute_columns = getattr(_minutes, "unify_goal_minute_columns")

_upcoming = load_local_module("upcoming", "upcoming.py")
render_upcoming = get_callable(_upcoming, "render_upcoming", label="upcoming")

# ---------- UI look ----------
def _inject_sidebar_css():
    st.markdown("""
    <style>
      [data-testid="stSidebar"] > div { padding-top: .6rem; }
      .sb-title { margin:.6rem 0 .25rem; font-weight:700; font-size:.92rem; color:#0f172a; }
      .sb-card { background:#f9fafb; border:1px solid #e5e7eb; border-radius:14px; padding:.6rem .65rem .7rem; margin-bottom:.6rem; }
      .tiny-badge { display:inline-block; padding:.1rem .5rem; border-radius:999px; background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; font-size:.78rem; }
    </style>
    """, unsafe_allow_html=True)

GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"

def selection_badges():
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    seasons = st.session_state.get(GLOBAL_SEASONS_KEY)
    txt_champ = f"🏆 <b>{champ}</b>" if champ else "🏷️ nessun campionato selezionato"
    txt_seas  = ", ".join([str(s) for s in seasons]) if seasons else "tutte le stagioni"
    st.markdown(
        f"<div style='margin:.25rem 0 .75rem 0;display:flex;gap:.5rem;flex-wrap:wrap'>"
        f"<span style='border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>{txt_champ}</span>"
        f"<span style='border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6'>🗓️ <b>Stagioni:</b> {txt_seas}</span>"
        f"</div>", unsafe_allow_html=True
    )

# ---- stagioni helpers ----
def _parse_season_start_year(season_str: str) -> int | None:
    if season_str is None: return None
    s = str(season_str); nums = re.findall(r"\d{2,4}", s)
    if not nums: return None
    first = nums[0]
    if len(first) == 4: return int(first)
    if len(first) == 2:
        yy = int(first); return 2000 + yy if yy <= 50 else 1900 + yy
    return None

def _current_season_start_year() -> int:
    now = pd.Timestamp.now(tz="Europe/Rome")
    return int(now.year if now.month >= 7 else now.year - 1)

def _seasons_desc_for(seasons_pool: list[str]) -> list[str]:
    uniq = sorted(set(str(x) for x in seasons_pool))
    uniq.sort(key=lambda s: (_parse_season_start_year(s) or -1, s), reverse=True)
    return uniq

def _map_startyear_to_seasons(seasons_list: list[str]) -> dict[int, list[str]]:
    m: dict[int, list[str]] = {}
    for s in seasons_list:
        y = _parse_season_start_year(s)
        if y is None: continue
        m.setdefault(y, []).append(s)
    for y in m:
        m[y].sort(key=lambda s: (_parse_season_start_year(s) or -1, s), reverse=True)
    return m

def _select_seasons_by_mode(seasons_all_desc: list[str], mode: str) -> list[str]:
    cur = _current_season_start_year()
    m = _map_startyear_to_seasons(seasons_all_desc)
    def take_last(n: int) -> list[str]:
        years = [cur - i for i in range(n)]
        out: list[str] = []
        for y in years: out.extend(m.get(y, []))
        return out
    return (
        m.get(cur, []) if mode == "Stagione corrente" else
        take_last(3)   if mode == "Ultime 3 stagioni" else
        take_last(5)   if mode == "Ultime 5 stagioni" else
        take_last(10)  if mode == "Ultime 10 stagioni" else
        []
    )

# ---------- pagina ----------
st.set_page_config(page_title="ProTrader — Hub", page_icon="⚽", layout="wide")
_inject_sidebar_css()

st.sidebar.markdown("<div class='sb-title'>Origine dati</div>", unsafe_allow_html=True)
origine_dati = st.sidebar.radio("", ["Supabase", "Upload Manuale"], key="origine_dati")

if origine_dati == "Supabase":
    st.sidebar.markdown("<div class='sb-card'>📦 Origine: Supabase Storage</div>", unsafe_allow_html=True)
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase", ui_mode="full", show_url_input=True)

    chosen_league  = st.session_state.get("campionato_supabase")
    chosen_seasons = st.session_state.get("campionato_supabase__seasons", [])
    if chosen_league: st.session_state[GLOBAL_CHAMP_KEY] = str(chosen_league)
    st.session_state[GLOBAL_SEASONS_KEY] = list(chosen_seasons) if isinstance(chosen_seasons, (list, tuple)) else []

    seasons_options = None
    for k in ("campionato_supabase__seasons_all","campionato_supabase__seasons_choices","supabase_seasons_choices"):
        if st.session_state.get(k): seasons_options = [str(x) for x in st.session_state[k]]; break
    if seasons_options is None:
        if "Stagione" in df.columns: seasons_options = sorted(df["Stagione"].dropna().astype(str).unique())
        elif "sezonul" in df.columns: seasons_options = sorted(df["sezonul"].dropna().astype(str).unique())
        else: seasons_options = []
    seasons_all_desc = _seasons_desc_for(seasons_options)

    st.sidebar.markdown("<div class='sb-title'>Intervallo stagioni</div>", unsafe_allow_html=True)
    mode = st.sidebar.radio("", ["Stagione corrente","Ultime 3 stagioni","Ultime 5 stagioni","Ultime 10 stagioni","Manuale"],
                            index=1 if len(seasons_all_desc)>=3 else 0, key="season_mode_supabase")
    if mode != "Manuale":
        sel_seasons = _select_seasons_by_mode(seasons_all_desc, mode)
        st.session_state[GLOBAL_SEASONS_KEY] = list(sel_seasons)
        wkey = "campionato_supabase__seasons"
        if wkey in st.session_state:
            cur_val = st.session_state[wkey]
            try:
                st.session_state[wkey] = tuple(sel_seasons) if isinstance(cur_val, tuple) else list(sel_seasons)
            except Exception:
                pass
        season_col = "Stagione" if "Stagione" in df.columns else ("sezonul" if "sezonul" in df.columns else None)
        if season_col and sel_seasons:
            df = df[df[season_col].astype(str).isin([str(s) for s in sel_seasons])]
        st.sidebar.caption(f"🎯 Preset → {', '.join(sel_seasons) if sel_seasons else '—'}")
    else:
        st.sidebar.caption("✍️ Manuale: usa il multiselect ‘Seleziona stagioni’ sopra.")

else:
    df, db_selected = load_data_from_file(ui_mode="minimal")
    st.sidebar.markdown("<div class='sb-card'>📄 Upload manuale del parquet locale</div>", unsafe_allow_html=True)
    champs = sorted(df["country"].dropna().astype(str).unique()) if "country" in df.columns else []
    if champs:
        st.sidebar.markdown("<div class='sb-title'>Campionato (upload)</div>", unsafe_allow_html=True)
        sel_champ = st.sidebar.selectbox("", champs, index=0)
        st.session_state[GLOBAL_CHAMP_KEY] = sel_champ
        df_ch = df[df["country"].astype(str) == str(sel_champ)]
        if "Stagione" in df_ch.columns: seasons_base = list(df_ch["Stagione"].dropna().astype(str).unique())
        elif "sezonul" in df_ch.columns: seasons_base = list(df_ch["sezonul"].dropna().astype(str).unique())
        else: seasons_base = []
        seasons_all_desc = _seasons_desc_for(seasons_base)

        st.sidebar.markdown("<div class='sb-title'>Intervallo stagioni</div>", unsafe_allow_html=True)
        mode = st.sidebar.radio("", ["Stagione corrente","Ultime 3 stagioni","Ultime 5 stagioni","Ultime 10 stagioni","Manuale"],
                                index=1 if len(seasons_all_desc)>=3 else 0, key="season_mode_upload")
        if mode != "Manuale":
            sel_seasons = _select_seasons_by_mode(seasons_all_desc, mode)
            st.session_state[GLOBAL_SEASONS_KEY] = list(sel_seasons)
            season_col = "Stagione" if "Stagione" in df.columns else ("sezonul" if "sezonul" in df.columns else None)
            if season_col and sel_seasons:
                df = df[(df["country"].astype(str)==str(sel_champ)) & (df[season_col].astype(str).isin([str(s) for s in sel_seasons]))]
            else:
                df = df[df["country"].astype(str)==str(sel_champ)]
            st.sidebar.caption(f"🎯 Preset → {', '.join(sel_seasons) if sel_seasons else '—'}")
        else:
            df = df[df["country"].astype(str)==str(sel_champ)]
            st.sidebar.caption("✍️ Manuale: gestisci i filtri stagioni nel modulo.")

# ---------- mapping colonne essenziali + label + minuti ----------
col_map = {
    "country": "country",
    "txtechipa1": "Home", "txtechipa2": "Away",
    "scor1": "Home Goal FT", "scor2": "Away Goal FT",
    "cotaa": "Odd home", "cotad": "Odd Away", "cotae": "Odd Draw",
    "sezonul": "Stagione", "datameci": "Data", "orameci": "Orario",
}
df = df.rename(columns=col_map)
df.columns = (df.columns.astype(str).str.strip()
              .str.replace(r"[\n\r\t]", "", regex=True)
              .str.replace(r"\s+", " ", regex=True))

if "Label" not in df.columns:
    if {"Odd home","Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

try:
    df = unify_goal_minute_columns(df)
except Exception as e:
    st.warning(f"Normalizzazione minuti-gol non applicata: {e}")

# ---------- header ----------
st.title("📊 Pre-Match — Hub")
selection_badges()
db_short = "Supabase" if str(db_selected).lower().startswith("supabase") else str(db_selected)
st.caption(f"Origine dati: **{db_short}** · <span class='tiny-badge'>Righe caricate: {len(df):,}</span>", unsafe_allow_html=True)

# ---------- menu ----------
st.sidebar.markdown("<div class='sb-title'>Naviga</div>", unsafe_allow_html=True)
menu = st.sidebar.radio("", ["Pre-Match (Hub)", "Upcoming"], key="menu_principale")
# Se Upcoming ha chiesto il redirect, forza il tab e riparti
if st.session_state.get("__goto_prematch__"):
    st.session_state.pop("__goto_prematch__", None)
    if menu != "Pre-Match (Hub)":
        st.session_state["menu_principale"] = "Pre-Match (Hub)"
        st.rerun()
    else:
        menu = "Pre-Match (Hub)"


# ---------- routing ----------
if menu == "Pre-Match (Hub)":
    run_pre_match(df, str(st.session_state.get(GLOBAL_CHAMP_KEY) or 'Dataset'))
else:
    # passa il DF corrente (per contesto), l’etichetta e la callback per aprire il pre-match
    render_upcoming(df.copy(), str(st.session_state.get(GLOBAL_CHAMP_KEY) or 'Dataset'), run_pre_match)
