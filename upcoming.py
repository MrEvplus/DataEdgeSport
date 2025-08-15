# upcoming.py — sezione "📅 Upcoming" (upload file giornata + auto-match LeagueRaw→CODICE + divisione, bypass filtro campionato)
from __future__ import annotations

import os, sys, re, importlib.util, unicodedata
from datetime import datetime
import streamlit as st
import pandas as pd

GLOBAL_CHAMP_KEY = "global_country"   # stessa chiave usata in app.py
GLOBAL_SEASONS_KEY = "global_seasons"

# ---- loader opzionale di utilità del progetto (se disponibili) ----
def _load_utils():
    try:
        base_dir = os.path.dirname(__file__)
        path = os.path.join(base_dir, "utils.py")
        spec = importlib.util.spec_from_file_location("up_utils", path)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["up_utils"] = mod
            spec.loader.exec_module(mod)  # type: ignore
            return (
                getattr(mod, "get_global_source_df", None),
                getattr(mod, "_read_parquet_filtered", None),
            )
    except Exception:
        pass
    return None, None

_get_global_source_df, _read_parquet_filtered = _load_utils()

# ---- normalizzazione stringhe ----
def _norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "", s.lower())

def _clean_code(s: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(s).upper())

# ---- lettura file giornata + mapping ----
def _guess_col(cols, candidates):
    cset = [_norm(c) for c in candidates]
    for c in cols:
        if _norm(c) in cset:
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
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet"):
        return pd.read_parquet(uploaded_file)
    try:
        return pd.read_excel(uploaded_file)
    except Exception:
        return pd.read_excel(uploaded_file, engine="xlrd")

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
                if re.fullmatch(r"\d{4}", t):  # 1730 -> 17:30
                    t = t[:2] + ":" + t[2:]
                tt = pd.to_datetime(t, errors="coerce").time()
            if tt:
                dt = pd.to_datetime(f"{dt.date()} {tt}")
        return dt
    except Exception:
        return None

# ---- dataset “all leagues” per Upcoming (bypass campionato) ----
# mapping minimo per allineare eventuali upload locali grezzi
COL_MAP_FOR_UPLOAD = {
    "country": "country",
    "txtechipa1": "Home", "txtechipa2": "Away",
    "echipa1": "Home", "echipa2": "Away",
    "gazde": "Home", "ospiti": "Away",
    "cotaa": "Odd home", "cotae": "Odd Draw", "cotad": "Odd Away",
}

def _get_df_all_leagues(df_current: pd.DataFrame) -> pd.DataFrame:
    """
    Ritorna il DF sorgente SENZA filtro campionato (mantiene il filtro stagioni, se disponibile).
    - Supabase: usa utils._read_parquet_filtered(url, league="", seasons)
    - Upload: rilegge l'upload originale (se disponibile) e rinomina le colonne essenziali
    - Fallback: df_current
    """
    if callable(_get_global_source_df):
        try:
            df_src, info = _get_global_source_df()
            seasons = tuple(st.session_state.get(GLOBAL_SEASONS_KEY) or [])
            origin  = (info or {}).get("origin")
            url     = (info or {}).get("url", "")
            if origin == "supabase" and url and callable(_read_parquet_filtered):
                try:
                    return _read_parquet_filtered(url, league="", seasons=seasons)
                except Exception as e:
                    st.warning(f"Bypass Supabase fallito: {e}")
            if isinstance(df_src, pd.DataFrame) and not df_src.empty:
                return df_src
        except Exception as e:
            st.warning(f"Bypass indisponibile: {e}")

    # Upload: prova a ricaricare l'ultimo file caricato in sidebar (se la tua utils lo espone)
    up = st.session_state.get("file_uploader_upload")
    if up is not None:
        try:
            name = (up.name or "").lower()
            if name.endswith(".csv"):
                df0 = pd.read_csv(up)
            elif name.endswith(".parquet"):
                df0 = pd.read_parquet(up)
            else:
                try:
                    df0 = pd.read_excel(up)
                except Exception:
                    df0 = pd.read_excel(up, engine="xlrd")
            df0 = df0.rename(columns=COL_MAP_FOR_UPLOAD)
            df0.columns = (df0.columns.astype(str).str.strip()
                           .str.replace(r"[\n\r\t]", "", regex=True)
                           .str.replace(r"\s+", " ", regex=True))
            return df0
        except Exception:
            pass

    return df_current

# ---- matching squadre/campionati ----
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

_NEUTRAL_TOKENS = {
    "fc","cf","sc","afc","calcio","ac","as","ssd","uc","usd","asd","polisportiva","sporting",
    "fk","bk","if","sv","sd","cd","de","clube","club","athletic","atletico","spd","ssa",
    "u","utd","united","city","town","u19","u21","ii","iii","b"
}
def _name_tokens(name: str) -> set[str]:
    s = _norm(name)
    raw = re.findall(r"[a-z0-9]+", s)
    toks = {t for t in raw if len(t) >= 2 and not t.isdigit()}
    return {t for t in toks if t not in _NEUTRAL_TOKENS}

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

@st.cache_data(show_spinner=False)
def _existing_country_codes(df_global: pd.DataFrame) -> list[str]:
    if "country" not in df_global.columns:
        return []
    codes = [_clean_code(x) for x in df_global["country"].dropna().astype(str).unique()]
    return [c for c in codes if c]

@st.cache_data(show_spinner=False)
def _teams_by_code(df_global: pd.DataFrame) -> dict[str,set]:
    tbc: dict[str,set] = {}
    if "country" not in df_global.columns:
        return tbc
    hc = "Home" if "Home" in df_global.columns else None
    ac = "Away" if "Away" in df_global.columns else None
    if not hc and not ac:
        return tbc
    for _, row in df_global.iterrows():
        c = row.get("country")
        if pd.isna(c): 
            continue
        code = _clean_code(c)
        if not code:
            continue
        tbc.setdefault(code, set())
        for col in (hc, ac):
            if not col:
                continue
            nm = row.get(col)
            if pd.isna(nm):
                continue
            nm = str(nm).strip()
            if nm:
                tbc[code].add(nm)
    return tbc

def _candidate_prefixes_from_leagueraw(leagueraw: str, existing_codes: list[str]) -> list[str]:
    key = _norm(leagueraw)
    pref = COUNTRY_PREFIX_MAP.get(key, [])
    if not pref and key.startswith("the"):
        pref = COUNTRY_PREFIX_MAP.get(key[3:], [])
    if not pref and "republic" in key:
        pref = COUNTRY_PREFIX_MAP.get(key.replace("republic",""), []) or COUNTRY_PREFIX_MAP.get(key.replace("republic","").strip(), [])
    if not pref and len(leagueraw) >= 3:
        pref = [leagueraw.strip().upper()[:3]]
    ex_pref = { re.match(r"^[A-Z]{3}", c).group(0) for c in existing_codes if re.match(r"^[A-Z]{3}", c) }
    return [p for p in pref if p in ex_pref]

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

def _auto_detect_code_by_leagueraw_and_teams(df_global: pd.DataFrame, leagueraw: str, home: str, away: str):
    existing_codes = _existing_country_codes(df_global)
    if not existing_codes:
        return None, {"reason": "no-codes"}
    prefixes = _candidate_prefixes_from_leagueraw(leagueraw or "", existing_codes)
    if not prefixes:
        return None, {"reason": "no-prefix-match"}
    tbc = _teams_by_code(df_global)
    candidates = [code for code in existing_codes if any(code.startswith(p) for p in prefixes)]
    ranked = []
    for code in candidates:
        teams = tbc.get(code, set())
        sH, mH = _best_match_in_code(home, teams)
        sA, mA = _best_match_in_code(away, teams)
        min_s, max_s = min(sH, sA), max(sH, sA)
        ranked.append((code, min_s, max_s, sH, sA, mH, mA))
    ranked.sort(key=lambda x: (x[1], x[2]), reverse=True)
    THRESH = 0.60  # prefisso restringe già molto
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

def _auto_detect_country_by_teams(df_global: pd.DataFrame, home: str, away: str, hint: str | None = None):
    """Fallback senza LeagueRaw: scorre tutti i campionati del DB."""
    tbc = _teams_by_code(df_global)
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
            "min_score": min_s, "max_score": max(sH, sA),
            "method": "token-match",
            "ordered_top3": ranked[:3],
        }
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

def _resolve_dataset_country(df_global: pd.DataFrame, league_str: str) -> str:
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

# ---- UI principale ----
def render_upcoming(df_current: pd.DataFrame, db_selected_label: str, run_pre_match_cb):
    st.title("📅 Upcoming — partite del giorno (quote auto-import)")

    # DF “all leagues” per il match (bypass campionato)
    df_all = _get_df_all_leagues(df_current)

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

    auto = _auto_map_upcoming(dfu)
    with st.expander("🔧 Mappatura colonne (controlla/correggi se necessario)", expanded=True):
        cols = list(dfu.columns)
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

    df_up = pd.DataFrame(records).sort_values("Datetime", na_position="last")
    st.subheader("Lista partite")
    show = df_up.copy()
    show["Data/Ora"] = show["Datetime"].astype(str).str.slice(0,16)
    st.dataframe(
        show[["Data/Ora","LeagueRaw","Home","Away","Odd home","Odd Draw","Odd Away","Over 1.5","Over 2.5","Over 3.5","BTTS"]],
        use_container_width=True,
        height=min(420, 44*(len(show)+1))
    )

    labels = []
    for _, r in df_up.iterrows():
        dts = r["Datetime"].strftime("%Y-%m-%d %H:%M") if pd.notna(r["Datetime"]) else "—"
        labels.append(f"[{dts}] {r['Home']} vs {r['Away']}  —  {str(r['LeagueRaw'] or '').strip()}")
    sel = st.selectbox("Seleziona un match per aprire l'analisi", options=labels, index=0 if labels else None, key="upcoming_pick")
    if not sel:
        return
    idx = labels.index(sel)
    row = df_up.iloc[idx]

    # --- AUTO-DETECT codice campionato ---
    auto_code, det1 = _auto_detect_code_by_leagueraw_and_teams(df_all, str(row["LeagueRaw"]), row["Home"], row["Away"])
    auto_code2, det2 = (None, None)
    if not auto_code:
        auto_code2, det2 = _auto_detect_country_by_teams(df_all, row["Home"], row["Away"], hint=str(row["LeagueRaw"]))
    league_guess = auto_code or auto_code2 or _resolve_dataset_country(df_all, str(row["LeagueRaw"]))

    countries = sorted(df_all["country"].dropna().astype(str).unique()) if "country" in df_all.columns else []
    st.caption("Associa il match al campionato presente nel tuo dataset (es. SPA1/ENG2...).")
    default_idx = (countries.index(league_guess) if (countries and league_guess in countries) else 0)
    target_country = st.selectbox("Campionato dataset", options=countries or [league_guess], index=default_idx)

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
            st.write("Nessun match affidabile; seleziona manualmente il codice campionato.")

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
            run_pre_match_cb(df_all, db_selected_label or "Dataset")
        except Exception as e:
            st.error(f"Errore nell'apertura del Pre-Match: {e}")
