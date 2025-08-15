# app.py — ProTrader Hub (Supabase + 📅 Upcoming con Auto-match LeagueRaw→CODICE + divisione)
from __future__ import annotations

import os
import sys
import re
import importlib.util
import unicodedata
from datetime import datetime
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
sys.modules["utils"] = _app_utils  # compat
load_data_from_supabase = getattr(_app_utils, "load_data_from_supabase")
load_data_from_file     = getattr(_app_utils, "load_data_from_file")
label_match             = getattr(_app_utils, "label_match")

# -------------------------------------------------------
# Modulo principale (Pre-Match Hub)
# -------------------------------------------------------
_pre_match = load_local_module("pre_match", "pre_match.py")
run_pre_match = get_callable(_pre_match, "run_pre_match", label="pre_match")

# Minuti-gol centralizzati
_minutes = load_local_module("minutes_mod", "minutes.py")
unify_goal_minute_columns = getattr(_minutes, "unify_goal_minute_columns")

# -------------------------------------------------------
# UI — stile sidebar
# -------------------------------------------------------
def _inject_sidebar_css():
    st.markdown("""
    <style>
    [data-testid="stSidebar"] > div { padding-top: .6rem; }
    .sb-header {
      background: linear-gradient(135deg, #0ea5e9 0%, #22c55e 100%);
      color: #fff; padding: .9rem .95rem; border-radius: 14px;
      box-shadow: 0 8px 24px rgba(2,6,23,.25);
      margin-bottom: .75rem; border: 1px solid rgba(255,255,255,.2);
    }
    .sb-header b { font-weight: 800; }
    .sb-sub { opacity:.85; font-size: .9rem; margin-top: .15rem; }
    .sb-title { margin: .6rem 0 .25rem 0; font-weight: 700; font-size: .92rem; color: #0f172a; }
    .sb-card {
      background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 14px;
      padding: .6rem .65rem .7rem; margin-bottom: .6rem;
    }
    [data-testid="stSidebar"] div[data-baseweb="select"] > div {
      background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px;
      box-shadow: 0 3px 10px rgba(2,6,23,.04);
      min-height: 44px;
    }
    [data-testid="stSidebar"] div[data-baseweb="tag"] {
      border-radius: 999px !important;
      background: #fee2e2 !important; color: #7f1d1d !important; border-color: #fecaca !important;
      font-weight: 600;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label {
      border: 1px solid #e5e7eb; border-radius: 12px; padding: .35rem .55rem;
      margin-bottom: .35rem; transition: all .12s ease-in-out;
      display:flex; align-items:center; gap:.4rem; background: #fff;
    }
    [data-testid="stSidebar"] [role="radiogroup"] label:hover {
      border-color: #cbd5e1; box-shadow: 0 4px 14px rgba(2,6,23,.06);
    }
    .tiny-badge {
      display:inline-block; padding:.1rem .5rem; border-radius: 999px;
      background:#eef2ff; color:#3730a3; border:1px solid #c7d2fe; font-size:.78rem;
    }
    </style>
    """, unsafe_allow_html=True)

def _sidebar_header():
    st.sidebar.markdown(
        "<div class='sb-header'>"
        "<div>⚽ <b>ProTrader — Hub</b></div>"
        "<div class='sb-sub'>Dataset filtrato in lettura: scegli prima Campionato & Stagioni</div>"
        "</div>",
        unsafe_allow_html=True
    )

# -------------------------------------------------------
# SUPPORTO
# -------------------------------------------------------
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

# ---- stagioni helpers (sempre all’indietro, switch a Luglio) ----
def _parse_season_start_year(season_str: str) -> int | None:
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
    now = pd.Timestamp.now(tz=tz)
    return int(now.year if now.month >= 7 else now.year - 1)

def _seasons_desc_for(seasons_pool: list[str]) -> list[str]:
    uniq = sorted(set(str(x) for x in seasons_pool))
    uniq.sort(key=lambda s: (_parse_season_start_year(s) or -1, s), reverse=True)
    return uniq

def _map_startyear_to_seasons(seasons_list: list[str]) -> dict[int, list[str]]:
    m: dict[int, list[str]] = {}
    for s in seasons_list:
        y = _parse_season_start_year(s)
        if y is None:
            continue
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
        for y in years:
            out.extend(m.get(y, []))
        return out

    if mode == "Stagione corrente":
        return m.get(cur, [])
    if mode == "Ultime 3 stagioni":
        return take_last(3)
    if mode == "Ultime 5 stagioni":
        return take_last(5)
    if mode == "Ultime 10 stagioni":
        return take_last(10)
    return []  # Manuale

# -------------------------------------------------------
# CONFIG PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="ProTrader — Hub", page_icon="⚽", layout="wide")
_inject_sidebar_css()
_sidebar_header()

# -------------------------------------------------------
# ORIGINE DATI + caricamento filtrato Supabase
# -------------------------------------------------------
st.sidebar.markdown("<div class='sb-title'>Origine dati</div>", unsafe_allow_html=True)
origine_dati = st.sidebar.radio("", ["Supabase", "Upload Manuale"], key="origine_dati")

if origine_dati == "Supabase":
    st.sidebar.markdown("<div class='sb-card'>📦 <b>Origine:</b> Supabase Storage (Parquet via DuckDB)</div>", unsafe_allow_html=True)
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase", ui_mode="full", show_url_input=True)
    chosen_league  = st.session_state.get("campionato_supabase")
    chosen_seasons = st.session_state.get("campionato_supabase__seasons", [])
    if chosen_league:
        st.session_state[GLOBAL_CHAMP_KEY] = str(chosen_league)
    st.session_state[GLOBAL_SEASONS_KEY] = list(chosen_seasons) if isinstance(chosen_seasons, (list, tuple)) else []

    seasons_options = None
    for k in ("campionato_supabase__seasons_all", "campionato_supabase__seasons_choices", "supabase_seasons_choices"):
        if k in st.session_state and st.session_state[k]:
            seasons_options = [str(x) for x in st.session_state[k]]
            break
    if seasons_options is None:
        if "Stagione" in df.columns:
            seasons_options = sorted(df["Stagione"].dropna().astype(str).unique())
        elif "sezonul" in df.columns:
            seasons_options = sorted(df["sezonul"].dropna().astype(str).unique())
        else:
            seasons_options = []

    seasons_all_desc = _seasons_desc_for(seasons_options)
    st.sidebar.markdown("<div class='sb-title'>Intervallo stagioni</div>", unsafe_allow_html=True)
    mode = st.sidebar.radio(
        "",
        ["Stagione corrente", "Ultime 3 stagioni", "Ultime 5 stagioni", "Ultime 10 stagioni", "Manuale"],
        index=1 if len(seasons_all_desc) >= 3 else 0,
        key="season_mode_supabase",
    )
    if mode != "Manuale":
        sel_seasons = _select_seasons_by_mode(seasons_all_desc, mode)
        st.session_state[GLOBAL_SEASONS_KEY] = list(sel_seasons)
        wkey = "campionato_supabase__seasons"
        if wkey in st.session_state:
            cur_val = st.session_state[wkey]
            try:
                if isinstance(cur_val, tuple):
                    st.session_state[wkey] = tuple(sel_seasons)
                elif isinstance(cur_val, list):
                    st.session_state[wkey] = list(sel_seasons)
            except Exception:
                pass
        season_col = "Stagione" if "Stagione" in df.columns else ("sezonul" if "sezonul" in df.columns else None)
        if season_col and sel_seasons:
            df = df[df[season_col].astype(str).isin([str(s) for s in sel_seasons])]
        st.sidebar.caption(f"🎯 Preset applicato → {', '.join(sel_seasons) if sel_seasons else 'nessuna stagione trovata'}")
    else:
        st.sidebar.caption("✍️ Manuale: usa il multiselect ‘Seleziona stagioni’ sopra.")
else:
    df, db_selected = load_data_from_file(ui_mode="minimal")
    st.sidebar.markdown("<div class='sb-card'>📄 Upload manuale del parquet locale</div>", unsafe_allow_html=True)
    champ_list = sorted(df["country"].dropna().astype(str).unique()) if "country" in df.columns else []
    if champ_list:
        st.sidebar.markdown("<div class='sb-title'>Campionato (upload)</div>", unsafe_allow_html=True)
        sel_champ = st.sidebar.selectbox("", champ_list, index=0)
        st.session_state[GLOBAL_CHAMP_KEY] = sel_champ
        df_ch = df[df["country"].astype(str) == str(sel_champ)]
        if "Stagione" in df_ch.columns:
            seasons_base = list(df_ch["Stagione"].dropna().astype(str).unique())
        elif "sezonul" in df_ch.columns:
            seasons_base = list(df_ch["sezonul"].dropna().astype(str).unique())
        else:
            seasons_base = []
        seasons_all_desc = _seasons_desc_for(seasons_base)
        st.sidebar.markdown("<div class='sb-title'>Intervallo stagioni</div>", unsafe_allow_html=True)
        mode = st.sidebar.radio(
            "",
            ["Stagione corrente", "Ultime 3 stagioni", "Ultime 5 stagioni", "Ultime 10 stagioni", "Manuale"],
            index=1 if len(seasons_all_desc) >= 3 else 0,
            key="season_mode_upload",
        )
        if mode != "Manuale":
            sel_seasons = _select_seasons_by_mode(seasons_all_desc, mode)
            st.session_state[GLOBAL_SEASONS_KEY] = list(sel_seasons)
            season_col = "Stagione" if "Stagione" in df.columns else ("sezonul" if "sezonul" in df.columns else None)
            if season_col and sel_seasons:
                df = df[(df["country"].astype(str) == str(sel_champ)) & (df[season_col].astype(str).isin([str(s) for s in sel_seasons]))]
            else:
                df = df[df["country"].astype(str) == str(sel_champ)]
            st.sidebar.caption(f"🎯 Preset applicato → {', '.join(sel_seasons) if sel_seasons else 'nessuna stagione trovata'}")
        else:
            df = df[df["country"].astype(str) == str(sel_champ)]
            st.sidebar.caption("✍️ Manuale: gestisci i filtri stagioni nel modulo.")

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

# Etichetta "Label"
if "Label" not in df.columns:
    if {"Odd home", "Odd Away"}.issubset(df.columns):
        df["Label"] = df.apply(label_match, axis=1)
    else:
        df["Label"] = "Others"

# Minuti-gol normalizzati
try:
    df = unify_goal_minute_columns(df)
except Exception as e:
    st.warning(f"Normalizzazione minuti-gol non applicata: {e}")

# -------------------------------------------------------
# 📅 UPCOMING — helpers (file + mapping + parsing)
# -------------------------------------------------------
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _guess_col(cols, candidates):
    cset = [_norm(c) for c in candidates]
    for c in cols:
        nc = _norm(c)
        if nc in cset:
            return c
    for c in cols:
        nc = _norm(c)
        if any(k in nc for k in cset):
            return c
    return None

def _coerce_odd(x):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return None

def _read_upcoming_file(uploaded_file) -> pd.DataFrame:
    name = (uploaded_file.name or "").lower()
    if name.endswith(".csv"):
        dfu = pd.read_csv(uploaded_file)
    elif name.endswith(".parquet"):
        dfu = pd.read_parquet(uploaded_file)
    else:
        try:
            dfu = pd.read_excel(uploaded_file)
        except Exception:
            dfu = pd.read_excel(uploaded_file, engine="xlrd")
    return dfu

def _auto_map_upcoming(df_in: pd.DataFrame) -> dict:
    cols = list(df_in.columns)
    M = {}
    M["league"] = _guess_col(cols, ["league","league raw","leagueraw","campionato","compet","country","liga","camp","competition"])
    M["date"]   = _guess_col(cols, ["date","data","matchdate"])
    M["time"]   = _guess_col(cols, ["time","ora","orario","hour"])
    M["home"]   = _guess_col(cols, ["home","team1","txtechipa1","echipa1","gazde","casa"])
    M["away"]   = _guess_col(cols, ["away","team2","txtechipa2","echipa2","ospiti","trasferta"])
    M["odd1"]   = _guess_col(cols, ["1","oddhome","homeodds","cotaa","odds1","cota1","home"])
    M["oddx"]   = _guess_col(cols, ["x","draw","odddraw","cotae","oddsx"])
    M["odd2"]   = _guess_col(cols, ["2","oddaway","awayodds","cotad","odds2","away"])
    M["ov15"]   = _guess_col(cols, ["over15","over 1.5","o1.5","cotao1"])
    M["ov25"]   = _guess_col(cols, ["over25","over 2.5","o2.5","cotao","cotao2"])
    M["ov35"]   = _guess_col(cols, ["over35","over 3.5","o3.5","cotao3"])
    M["btts"]   = _guess_col(cols, ["btts","both teams to score","gg","gg/no"])
    return M

def _to_datetime_safe(datestr, timestr):
    try:
        if pd.isna(datestr) and pd.isna(timestr):
            return None
        if isinstance(datestr, (pd.Timestamp, datetime)):
            dt = pd.to_datetime(datestr)
        else:
            dt = pd.to_datetime(str(datestr), errors="coerce", dayfirst=True, utc=False)
        if pd.isna(dt):
            return None
        t = str(timestr or "").strip()
        if t and t not in ("", "nan"):
            try:
                tt = pd.to_datetime(t).time()
            except Exception:
                t = t.replace(".", ":")
                if re.fullmatch(r"\d{4}", t):
                    t = t[:2] + ":" + t[2:]
                tt = pd.to_datetime(t, errors="coerce").time()
            if tt:
                dt = pd.to_datetime(f"{dt.date()} {tt}")
        return dt
    except Exception:
        return None

# -------------------------------------------------------
# 🔎 LEAGUE RAW → PREFISSI + AUTO-MATCH CODICE (divisione)
# -------------------------------------------------------
# Sinonimi (name→prefissi); verranno filtrati sui prefissi realmente presenti nel DB
COUNTRY_PREFIX_MAP = {
    "italy": ["ITA"], "england": ["ENG"], "spain": ["SPA","ESP"], "germany": ["GER","DEU"],
    "france": ["FRA"], "portugal": ["POR"], "netherlands": ["NED","NET","HOL"],
    "belgium": ["BEL"], "turkey": ["TUR"], "poland": ["POL"], "hungary": ["HUN"],
    "romania": ["ROU","ROM"], "greece": ["GRE"], "austria": ["AUT"], "switzerland": ["SUI","SWI","CHE"],
    "czechrepublic": ["CZE"], "czechia": ["CZE"], "slovakia": ["SVK"], "slovenia": ["SVN","SLO"],
    "croatia": ["CRO"], "serbia": ["SRB"], "russia": ["RUS"], "ukraine": ["UKR"],
    "sweden": ["SWE"], "norway": ["NOR"], "denmark": ["DEN","DNK"],
    "scotland": ["SCO"], "wales": ["WAL"], "ireland": ["IRL","ROI"], "northernireland": ["NIR"],
    "bulgaria": ["BUL"], "belarus": ["BLR"], "finland": ["FIN"], "iceland": ["ISL"]
}

def _clean_code(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())

@st.cache_data(show_spinner=False)
def _existing_country_codes(df_global: pd.DataFrame) -> list[str]:
    if "country" not in df_global.columns:
        return []
    codes = [ _clean_code(x) for x in df_global["country"].dropna().astype(str).unique() ]
    # rimuovi vuoti
    return [c for c in codes if c]

@st.cache_data(show_spinner=False)
def _teams_by_country(df_global: pd.DataFrame) -> dict[str,set]:
    tbc: dict[str,set] = {}
    if "country" not in df_global.columns:
        return tbc
    home_col = "Home" if "Home" in df_global.columns else None
    away_col = "Away" if "Away" in df_global.columns else None
    if not home_col and not away_col:
        return tbc
    for _, row in df_global.iterrows():
        c = row.get("country")
        if pd.isna(c): 
            continue
        code = _clean_code(c)
        if not code:
            continue
        tbc.setdefault(code, set())
        for col in (home_col, away_col):
            if not col:
                continue
            nm = row.get(col)
            if pd.isna(nm):
                continue
            nm = str(nm).strip()
            if nm:
                tbc[code].add(nm)
    return tbc

_NEUTRAL_TOKENS = {
    "fc","cf","sc","afc","calcio","ac","as","ssd","uc","usd","asd","polisportiva","sporting",
    "fk","bk","if","sv","sd","cd","de","clube","club","athletic","atletico","spd","ssa",
    "u","utd","united","city","town","u19","u21","ii","iii","b"
}
def _name_tokens(name: str) -> set[str]:
    s = _norm(name)
    raw = re.findall(r"[a-z0-9]+", s)
    toks = {t for t in raw if len(t) >= 2 and not t.isdigit()}
    toks = {t for t in toks if t not in _NEUTRAL_TOKENS}
    return toks

def _token_score(a: str, b: str) -> float:
    A, B = _name_tokens(a), _name_tokens(b)
    if not A or not B:
        na, nb = _norm(a), _norm(b)
        if not na or not nb:
            return 0.0
        if na in nb or nb in na:
            return 0.6
        return 0.0
    inter = len(A & B)
    denom = max(len(A), len(B))
    return inter/denom if denom else 0.0

def _best_match_in_code(name: str, teams_in_code: set[str]) -> tuple[float,str|None]:
    if not teams_in_code:
        return 0.0, None
    nn = _norm(name)
    for t in teams_in_code:
        if _norm(t) == nn:
            return 1.0, t
    best_s, best_t = 0.0, None
    for t in teams_in_code:
        s = _token_score(name, t)
        if s > best_s:
            best_s, best_t = s, t
            if best_s >= 0.99:
                break
    return best_s, best_t

def _candidate_prefixes_from_leagueraw(leagueraw: str, existing_codes: list[str]) -> list[str]:
    """Converte 'Netherlands'→['NED'] ecc., poi filtra prefissi presenti nel DB."""
    key = _norm(leagueraw)
    # prova exact key
    pref = COUNTRY_PREFIX_MAP.get(key, [])
    # fallback: togli spazi / parole tipo 'the'
    if not pref and key.startswith("the"):
        pref = COUNTRY_PREFIX_MAP.get(key[3:], [])
    if not pref and "republic" in key:
        pref = COUNTRY_PREFIX_MAP.get(key.replace("republic",""), []) or COUNTRY_PREFIX_MAP.get(key.replace("republic","").strip(), [])
    if not pref:
        # heuristic: prime 3 lettere uppercase
        if len(leagueraw) >= 3:
            pref = [leagueraw.strip().upper()[:3]]
    # filtra su prefissi realmente presenti
    ex_pref = { re.match(r"^[A-Z]{3}", c).group(0) for c in existing_codes if re.match(r"^[A-Z]{3}", c) }
    pref = [p for p in pref if p in ex_pref]
    return pref

def _auto_detect_code_by_leagueraw_and_teams(df_global: pd.DataFrame, leagueraw: str, home: str, away: str):
    """
    Sceglie il codice campionato (es. SPA1/SPA2) coerente con LeagueRaw e dove compaiono entrambe le squadre.
    """
    existing_codes = _existing_country_codes(df_global)
    if not existing_codes:
        return None, {"reason": "no-codes"}
    prefixes = _candidate_prefixes_from_leagueraw(leagueraw or "", existing_codes)
    if not prefixes:
        return None, {"reason": "no-prefix-match"}
    tbc = _teams_by_country(df_global)
    candidates = [code for code in existing_codes if any(code.startswith(p) for p in prefixes)]
    ranked = []
    for code in candidates:
        teams = tbc.get(code, set())
        sH, mH = _best_match_in_code(home, teams)
        sA, mA = _best_match_in_code(away, teams)
        min_s, max_s = min(sH, sA), max(sH, sA)
        ranked.append((code, min_s, max_s, sH, sA, mH, mA))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    THRESH = 0.60  # più permissivo perché prefisso già filtra molto
    if ranked and ranked[0][1] >= THRESH:
        code, min_s, max_s, sH, sA, mH, mA = ranked[0]
        return code, {
            "home_score": sH, "home_match": mH,
            "away_score": sA, "away_match": mA,
            "min_score": min_s, "max_score": max_s,
            "method": "prefix+token-match",
            "ordered_top3": ranked[:3],
        }
    return None, {"reason": "low-score-prefix", "top": ranked[:3]}

def _resolve_dataset_country(df_global: pd.DataFrame, league_str: str) -> str:
    """Fallback molto generico (non usa divisione)."""
    if "country" not in df_global.columns or df_global.empty:
        return str(league_str or "")
    options = sorted(set(df_global["country"].dropna().astype(str)))
    s = _norm(league_str)
    for o in options:
        if _norm(o) == s:
            return o
    for o in options:
        if s and (_norm(o).find(s) >= 0 or s.find(_norm(o)) >= 0):
            return o
    return options[0] if options else str(league_str or "")

def _auto_detect_country_by_teams(df_global: pd.DataFrame, home: str, away: str, hint: str | None = None):
    """Fallback senza usare LeagueRaw (scorre tutti i campionati del DB)."""
    tbc = _teams_by_country(df_global)
    if not tbc:
        return None, {"reason": "no-index"}
    ranked = []
    for code, teams in tbc.items():
        sH, mH = _best_match_in_code(home, teams)
        sA, mA = _best_match_in_code(away, teams)
        min_s, max_s = min(sH, sA), max(sH, sA)
        ranked.append((code, min_s, max_s, sH, sA, mH, mA))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    THRESH = 0.78
    if ranked and ranked[0][1] >= THRESH:
        code, min_s, max_s, sH, sA, mH, mA = ranked[0]
        return code, {
            "home_score": sH, "home_match": mH,
            "away_score": sA, "away_match": mA,
            "min_score": min_s, "max_score": max_s,
            "method": "token-match",
            "ordered_top3": ranked[:3],
        }
    # tie-break con hint (se il prefisso del hint è presente)
    if ranked and hint:
        pref = _candidate_prefixes_from_leagueraw(hint, _existing_country_codes(df_global))
        for code, min_s, _, sH, sA, mH, mA in ranked[:3]:
            if pref and any(code.startswith(p) for p in pref) and min_s >= 0.60:
                return code, {
                    "home_score": sH, "home_match": mH,
                    "away_score": sA, "away_match": mA,
                    "min_score": min_s, "max_score": max(sH, sA),
                    "method": "token+hint",
                    "ordered_top3": ranked[:3],
                }
    return None, {"reason": "low-score", "top": ranked[:3]}

# -------------------------------------------------------
# 📅 UPCOMING — UI
# -------------------------------------------------------
def render_upcoming(df_global: pd.DataFrame, db_selected_label: str):
    st.title("📅 Upcoming — partite del giorno (quote auto-import)")

    up = st.file_uploader("Carica il file quotidiano (Excel/CSV)", type=["xls","xlsx","csv","parquet"], key="upcoming_upl")
    if up is None:
        st.info("Carica il file della giornata per vedere la lista dei match.")
        return

    try:
        dfu = _read_upcoming_file(up)
    except Exception as e:
        st.error(f"Impossibile leggere il file: {e}")
        return

    if dfu is None or dfu.empty:
        st.warning("Il file è vuoto.")
        return

    # Mappatura colonne (auto + override)
    auto = _auto_map_upcoming(dfu)
    with st.expander("🔧 Mappatura colonne (controlla/correggi se necessario)", expanded=True):
        cols = list(dfu.columns)
        if not cols:
            st.warning("Nessuna colonna trovata nel file.")
            return
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            col_league = st.selectbox("Lega/Campionato (LeagueRaw)", options=cols, index=cols.index(auto["league"]) if auto.get("league") in cols else 0)
            col_date   = st.selectbox("Data", options=cols, index=cols.index(auto["date"])   if auto.get("date")   in cols else 0)
            col_time   = st.selectbox("Ora",  options=cols, index=cols.index(auto["time"])   if auto.get("time")   in cols else 0)
        with c2:
            col_home   = st.selectbox("Squadra Casa", options=cols, index=cols.index(auto["home"]) if auto.get("home") in cols else 0)
            col_away   = st.selectbox("Squadra Ospite", options=cols, index=cols.index(auto["away"]) if auto.get("away") in cols else 0)
        with c3:
            col_1      = st.selectbox("Quota 1", options=cols, index=cols.index(auto["odd1"]) if auto.get("odd1") in cols else 0)
            col_x      = st.selectbox("Quota X", options=cols, index=cols.index(auto["oddx"]) if auto.get("oddx") in cols else 0)
            col_2      = st.selectbox("Quota 2", options=cols, index=cols.index(auto["odd2"]) if auto.get("odd2") in cols else 0)
        with c4:
            col_ov15   = st.selectbox("Over 1.5 (facolt.)", options=["(nessuna)"]+cols, index=(cols.index(auto["ov15"])+1) if auto.get("ov15") in cols else 0)
            col_ov25   = st.selectbox("Over 2.5 (facolt.)", options=["(nessuna)"]+cols, index=(cols.index(auto["ov25"])+1) if auto.get("ov25") in cols else 0)
            col_ov35   = st.selectbox("Over 3.5 (facolt.)", options=["(nessuna)"]+cols, index=(cols.index(auto["ov35"])+1) if auto.get("ov35") in cols else 0)
            col_btts   = st.selectbox("BTTS (facolt.)",     options=["(nessuna)"]+cols, index=(cols.index(auto["btts"])+1) if auto.get("btts") in cols else 0)

    # Normalizza in uno schema unico
    def _pick(row, colname):
        return row[colname] if (colname and colname in row) else None

    records = []
    for _, r in dfu.iterrows():
        league = _pick(r, col_league)
        dt = _to_datetime_safe(_pick(r, col_date), _pick(r, col_time))
        home = str(_pick(r, col_home) or "").strip()
        away = str(_pick(r, col_away) or "").strip()
        o1   = _coerce_odd(_pick(r, col_1))
        ox   = _coerce_odd(_pick(r, col_x))
        o2   = _coerce_odd(_pick(r, col_2))
        ov15 = None if col_ov15 == "(nessuna)" else _coerce_odd(_pick(r, col_ov15))
        ov25 = None if col_ov25 == "(nessuna)" else _coerce_odd(_pick(r, col_ov25))
        ov35 = None if col_ov35 == "(nessuna)" else _coerce_odd(_pick(r, col_ov35))
        btts = None if col_btts == "(nessuna)" else _coerce_odd(_pick(r, col_btts))
        if not home or not away:
            continue
        records.append({
            "LeagueRaw": league,
            "Datetime": dt,
            "Home": home,
            "Away": away,
            "Odd home": o1,
            "Odd Draw": ox,
            "Odd Away": o2,
            "Over 1.5": ov15,
            "Over 2.5": ov25,
            "Over 3.5": ov35,
            "BTTS": btts,
        })

    if not records:
        st.warning("Nessun match valido trovato nel file.")
        return

    df_up = pd.DataFrame(records)
    if "Datetime" in df_up.columns:
        df_up = df_up.sort_values("Datetime", na_position="last")

    st.subheader("Lista partite")
    show = df_up.copy()
    show["Data/Ora"] = show["Datetime"].astype(str).str.slice(0,16)
    st.dataframe(
        show[["Data/Ora","LeagueRaw","Home","Away","Odd home","Odd Draw","Odd Away","Over 1.5","Over 2.5","Over 3.5","BTTS"]],
        use_container_width=True,
        height=min(420, 44*(len(show)+1))
    )

    # Selezione match
    label_opts = []
    for i, r in df_up.iterrows():
        dts = r["Datetime"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["Datetime"]) else "—"
        leg = str(r["LeagueRaw"] or "").strip()
        label_opts.append(f"[{dts}] {r['Home']} vs {r['Away']}  —  {leg}")

    sel = st.selectbox("Seleziona un match per aprire l'analisi", options=label_opts, index=0 if label_opts else None, key="upcoming_pick")
    if not sel:
        return

    idx = label_opts.index(sel)
    row = df_up.iloc[idx]

    # 1) PROVA con prefisso da LeagueRaw + token-match squadre (seleziona anche la divisione)
    auto_code, det1 = _auto_detect_code_by_leagueraw_and_teams(df_global, str(row["LeagueRaw"]), row["Home"], row["Away"])

    # 2) Fallback: token-match su TUTTI i campionati
    auto_code2, det2 = (None, None)
    if not auto_code:
        auto_code2, det2 = _auto_detect_country_by_teams(df_global, row["Home"], row["Away"], hint=str(row["LeagueRaw"]))

    # 3) Ultimo fallback: risoluzione bland (quasi mai necessaria)
    league_guess = auto_code or auto_code2 or _resolve_dataset_country(df_global, str(row["LeagueRaw"]))

    # UI scelta campionato (default = auto rilevato)
    countries = sorted(df_global["country"].dropna().astype(str).unique()) if "country" in df_global.columns else []
    st.caption("Associa il match al campionato presente nel tuo dataset (colonna ‘country’: es. SPA1/SPA2).")
    default_idx = (countries.index(league_guess) if (countries and league_guess in countries) else 0)
    target_country = st.selectbox(
        "Campionato dataset",
        options=countries or [league_guess],
        index=default_idx,
        key="upcoming_dataset_country"
    )

    # Info trasparente sull'auto-match
    with st.expander("ℹ️ Dettagli auto-match campionato", expanded=False):
        if auto_code:
            st.write(f"**Rilevato (LeagueRaw→prefisso):** {auto_code}")
            if isinstance(det1, dict):
                try:
                    st.write(f"- Home match: `{det1.get('home_match')}` (score={det1.get('home_score'):.2f})")
                    st.write(f"- Away match: `{det1.get('away_match')}` (score={det1.get('away_score'):.2f})")
                    st.write(f"- Metodo: {det1.get('method')}  ·  min_score={det1.get('min_score'):.2f}")
                except Exception:
                    st.write(det1)
        elif auto_code2:
            st.write(f"**Rilevato (solo token-match):** {auto_code2}")
            if isinstance(det2, dict):
                try:
                    st.write(f"- Home match: `{det2.get('home_match')}` (score={det2.get('home_score'):.2f})")
                    st.write(f"- Away match: `{det2.get('away_match')}` (score={det2.get('away_score'):.2f})")
                    st.write(f"- Metodo: {det2.get('method')}  ·  min_score={det2.get('min_score'):.2f}")
                except Exception:
                    st.write(det2)
        else:
            st.write("Nessun match affidabile; fallback su risoluzione generale del nome campionato.")

    # ▶ Apri Pre-Match con quote importate
    if st.button("🚀 Apri in Pre-Match (quote autocaricate)", type="primary", use_container_width=True, key="open_prematch"):
        st.session_state[GLOBAL_CHAMP_KEY] = str(target_country)
        st.session_state["prematch:squadra_casa"]   = str(row["Home"])
        st.session_state["prematch:squadra_ospite"] = str(row["Away"])
        if row["Odd home"] is not None: st.session_state["prematch:quota_home"] = float(row["Odd home"])
        if row["Odd Draw"] is not None: st.session_state["prematch:quota_draw"] = float(row["Odd Draw"])
        if row["Odd Away"] is not None: st.session_state["prematch:quota_away"] = float(row["Odd Away"])
        if row.get("Over 1.5") is not None: st.session_state["prematch:shared:q_ov15"] = float(row["Over 1.5"])
        if row.get("Over 2.5") is not None: st.session_state["prematch:shared:q_ov25"] = float(row["Over 2.5"])
        if row.get("Over 3.5") is not None: st.session_state["prematch:shared:q_ov35"] = float(row["Over 3.5"])
        if row.get("BTTS")     is not None: st.session_state["prematch:shared:q_btts"] = float(row["BTTS"])
        st.success("Impostazioni caricate: campionato, squadre e quote.")
        try:
            run_pre_match(df_global, _db_label_for_modules())
        except Exception as e:
            st.error(f"Errore nell'apertura del Pre-Match: {e}")

# -------------------------------------------------------
# HEADER & BADGES
# -------------------------------------------------------
st.title("📊 Pre-Match — Hub")
selection_badges()
db_short = _short_origin_label(str(db_selected))
st.caption(f"Origine dati: **{db_short}** · <span class='tiny-badge'>Righe caricate: {len(df):,}</span>", unsafe_allow_html=True)

# -------------------------------------------------------
# MENU (Hub)
# -------------------------------------------------------
st.sidebar.markdown("<div class='sb-title'>Naviga</div>", unsafe_allow_html=True)
menu_option = st.sidebar.radio("", ["Pre-Match (Hub)", "Upcoming"], key="menu_principale")

# -------------------------------------------------------
# ROUTING
# -------------------------------------------------------
if menu_option == "Pre-Match (Hub)":
    run_pre_match(df, _db_label_for_modules())
elif menu_option == "Upcoming":
    render_upcoming(df.copy(), _db_label_for_modules())

