# pages/1_Admin_Carica_Upcoming.py
from __future__ import annotations
import os, io, re, unicodedata, math
from datetime import datetime
import pandas as pd
import streamlit as st

# supabase
from supabase import create_client, Client

st.set_page_config(page_title="Admin — Carica Upcoming", page_icon="🗂", layout="wide")

# --------------------------------------------------------------------
# Credenziali
# --------------------------------------------------------------------
SB_URL = st.secrets.get("SUPABASE_URL", "")
SB_ANON = st.secrets.get("SUPABASE_ANON_KEY", "")
SB_SERVICE = st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "")  # per scrivere la tabella
if not SB_URL or not (SB_ANON or SB_SERVICE):
    st.error("Config mancante: aggiungi SUPABASE_URL e una chiave (ANON o SERVICE_ROLE) in st.secrets.")
    st.stop()

# client per letture Storage (anon ok) e per scritture (service se presente)
sb_read: Client = create_client(SB_URL, SB_ANON or SB_SERVICE)
sb_write: Client = create_client(SB_URL, SB_SERVICE or SB_ANON)

# --------------------------------------------------------------------
# Utils di normalizzazione / matching (replica del comportamento buono)
# --------------------------------------------------------------------
NEUTRAL = {"fc","cf","sc","afc","ac","as","ssd","uc","usd","asd","club","clube",
           "sv","bk","fk","if","u23","u21","u19","ii","iii","b","res","reserve","am","amat"}

def _norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+","", s.lower())

def _tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", _norm(name)) if len(t)>=2 and t not in NEUTRAL and not t.isdigit()}

def _tok_score(a,b):
    A,B=_tokens(a),_tokens(b)
    if not A or not B: return 0.0
    return len(A&B)/max(len(A),len(B))

def _coerce_f(x):
    try: return float(str(x).replace(",", "."))
    except: return math.nan

COUNTRY_PREFIX = {
    "italy":["ITA"],"england":["ENG"],"spain":["SPA","ESP"],"germany":["GER","DEU"],
    "france":["FRA"],"portugal":["POR"],"netherlands":["NED","HOL","NET"],
    "belgium":["BEL"],"turkey":["TUR"],"poland":["POL"],"denmark":["DEN","DNK"],
    "sweden":["SWE"],"norway":["NOR"],"austria":["AUT"],"switzerland":["SUI","CHE"],
    "romania":["ROU","ROM"],"greece":["GRE"],"scotland":["SCO"],"wales":["WAL"],
    "ireland":["IRL","ROI"],"hungary":["HUN"],"czechia":["CZE"],"slovakia":["SVK"],
    "slovenia":["SVN","SLO"],"croatia":["CRO"],"serbia":["SRB"],"russia":["RUS"],
    "ukraine":["UKR"],"argentina":["ARG"],"brazil":["BRA"],"mexico":["MEX"],
}
COMP_HINTS = {"ligue":"france","laliga":"spain","primera":"spain","bundesliga":"germany",
              "seriea":"italy","serieb":"italy","jupiler":"belgium","eredivisie":"netherlands","superlig":"turkey"}

def _country_key_from_leagueraw(leagueraw: str) -> str|None:
    s = _norm(leagueraw)
    if not s: return None
    for k in COUNTRY_PREFIX:
        if k in s or s.startswith(k): return k
    for kw,key in COMP_HINTS.items():
        if kw in s: return key
    return None

def best_league_code(team_idx: dict[str,set[str]], leagueraw: str, home: str, away: str) -> str|None:
    pref_key = _country_key_from_leagueraw(leagueraw or "")
    prefixes = COUNTRY_PREFIX.get(pref_key, [])
    candidates = [code for code in team_idx.keys() if (not prefixes or any(code.startswith(p) for p in prefixes))]
    ranked=[]
    for code in candidates:
        teams = team_idx[code]
        sH = max((_tok_score(home,t) for t in teams), default=0.0)
        sA = max((_tok_score(away,t) for t in teams), default=0.0)
        ranked.append((code, min(sH,sA), max(sH,sA)))
    ranked.sort(key=lambda x:(x[1],x[2]), reverse=True)
    return ranked[0][0] if ranked and ranked[0][1] >= 0.55 else None

# --------------------------------------------------------------------
# Sorgente storico per l'indice squadre (riusa il parquet/caricamento dell'app)
# --------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_global_df() -> pd.DataFrame:
    """
    Prova a riusare le utilità dell'Hub (utils.py). In alternativa chiede un file parquet locale.
    """
    # 1) prova con utils.py dell'app
    try:
        import importlib.util, sys, os
        base = os.path.dirname(os.path.dirname(__file__))
        p = os.path.join(base, "utils.py")
        spec = importlib.util.spec_from_file_location("u_mod", p)
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            sys.modules["u_mod"] = mod
            spec.loader.exec_module(mod)  # type: ignore
            get_global_source_df = getattr(mod, "get_global_source_df", None)
            _read_parquet_filtered = getattr(mod, "_read_parquet_filtered", None)
            if callable(get_global_source_df):
                df0, info = get_global_source_df()
                # se c'e' la signed url del parquet, leggiamo tutto senza filtro di campionato
                url = (info or {}).get("url", "")
                seasons = tuple(st.session_state.get("global_seasons") or [])
                if url and callable(_read_parquet_filtered):
                    try:
                        df_all = _read_parquet_filtered(url, league="", seasons=seasons)
                        return df_all
                    except Exception:
                        pass
                return df0 if isinstance(df0, pd.DataFrame) else pd.DataFrame()
    except Exception:
        pass
    # 2) fallback: chiedi parquet locale all'utente (minime colonne: Home, Away, country)
    st.warning("Sorgente globale non disponibile. Carica un parquet locale con Home, Away, country.")
    up = st.file_uploader("Carica parquet storico (min: Home, Away, country)", type=["parquet"], key="parquet_hist")
    if not up:
        return pd.DataFrame()
    return pd.read_parquet(up)

def make_team_index(df_hist: pd.DataFrame) -> dict[str,set[str]]:
    idx: dict[str,set[str]] = {}
    if df_hist.empty:
        return idx
    keep_cols = [c for c in df_hist.columns if c in ("Home","Away","country")]
    df = df_hist[keep_cols].dropna(subset=["Home","Away","country"])
    for _,r in df.iterrows():
        code = str(r["country"])
        idx.setdefault(code,set()).update([str(r["Home"]), str(r["Away"])])
    return idx

# --------------------------------------------------------------------
# Normalizzazione dell'XLS con i nomi colonne che mi hai dato
# --------------------------------------------------------------------
def normalize_upcoming(df_raw: pd.DataFrame, default_date: pd.Timestamp) -> pd.DataFrame:
    cols = {c.lower(): c for c in df_raw.columns}

    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    # mapping specifico (con fallback)
    c_league = pick("league","leagueraw","country","compet")
    c_home   = pick("txtechipa1","home")
    c_away   = pick("txtechipa2","away")

    c_1 = pick("cotaa","odd1","1"); c_x = pick("cotae","oddx","x"); c_2 = pick("cotad","odd2","2")
    c_ov15 = pick("cotao1","over 1.5","over15","o1.5")
    c_ov25 = pick("cotao","over 2.5","over25","o2.5")
    c_ov35 = pick("cotao3","over 3.5","over35","o3.5")
    c_btts = pick("gg","btts")

    c_date = pick("datameci","date","data","matchdate")
    c_time = pick("orameci","time","ora","orario","hour")
    c_dt   = pick("data/ora","dataora","datetime","kickoff","date/ora")

    def _parse_time_like(x):
        s = str(x or "").strip()
        if not s or s.lower()=="nan": return None
        s = s.replace(".", ":")
        if re.fullmatch(r"\d{4}", s): s = s[:2]+":"+s[2:]
        return s

    def build_dt(row):
        if c_dt:
            s = str(row.get(c_dt) or "").strip()
            if not s: return None
            if re.fullmatch(r"\d{1,2}[:.]\d{2}", s) or re.fullmatch(r"\d{4}", s):
                hhmm = _parse_time_like(s)
                return pd.to_datetime(f"{default_date.date()} {hhmm}", dayfirst=True, errors="coerce")
            return pd.to_datetime(s, dayfirst=True, errors="coerce")
        d = row.get(c_date); t = row.get(c_time)
        if (d is None or str(d).strip() in ("","nan")) and t is not None:
            hhmm = _parse_time_like(t)
            return pd.to_datetime(f"{default_date.date()} {hhmm}", dayfirst=True, errors="coerce")
        if d is not None and (t is None or str(t).strip() in ("","nan")):
            return pd.to_datetime(str(d), dayfirst=True, errors="coerce")
        if d is not None and t is not None:
            hhmm = _parse_time_like(t)
            if not hhmm: return pd.to_datetime(str(d), dayfirst=True, errors="coerce")
            return pd.to_datetime(f"{d} {hhmm}", dayfirst=True, errors="coerce")
        return None

    out=[]
    for _,r in df_raw.iterrows():
        dt = build_dt(r)
        if pd.isna(dt): continue
        home = str(r.get(c_home) or "").strip(); away = str(r.get(c_away) or "").strip()
        if not home or not away: continue
        out.append({
            "kickoff_ts": pd.to_datetime(dt).tz_localize("Europe/Rome").tz_convert("UTC"),
            "league_raw": r.get(c_league),
            "home": home, "away": away,
            "odd1": _coerce_f(r.get(c_1)) if c_1 else math.nan,
            "oddx": _coerce_f(r.get(c_x)) if c_x else math.nan,
            "odd2": _coerce_f(r.get(c_2)) if c_2 else math.nan,
            "ov15": _coerce_f(r.get(c_ov15)) if c_ov15 else math.nan,
            "ov25": _coerce_f(r.get(c_ov25)) if c_ov25 else math.nan,
            "ov35": _coerce_f(r.get(c_ov35)) if c_ov35 else math.nan,
            "btts": _coerce_f(r.get(c_btts)) if c_btts else math.nan,
        })
    return pd.DataFrame(out)

# lettura robusta file
def read_upcoming_file(uploaded) -> pd.DataFrame:
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx"):
        import openpyxl  # ensure installed
        return pd.read_excel(uploaded, engine="openpyxl")
    if name.endswith(".xls"):
        import xlrd      # ensure installed
        return pd.read_excel(uploaded, engine="xlrd")
    # fallback
    return pd.read_excel(uploaded)

# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
st.title("Admin — Carica Upcoming (web)")

col_top1, col_top2 = st.columns([2,1])
with col_top1:
    upl = st.file_uploader("Carica file giornaliero (XLS/XLSX/CSV)", type=["xls","xlsx","csv"])
with col_top2:
    base_date = st.date_input("Data di riferimento (se nel file manca la data)", value=pd.Timestamp.now(tz="Europe/Rome").date())

if not upl:
    st.info("Carica il file per iniziare.")
    st.stop()

# leggi e normalizza
try:
    raw = read_upcoming_file(upl)
except Exception as e:
    st.error(f"Errore lettura file: {e}")
    st.stop()

df_up = normalize_upcoming(raw, pd.Timestamp(base_date).tz_localize("Europe/Rome"))
if df_up.empty:
    st.warning("Nessuna riga valida normalizzata.")
    st.stop()

st.subheader("Anteprima normalizzata")
prev = df_up.copy()
prev["kickoff_local"] = pd.to_datetime(prev["kickoff_ts"]).dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(prev[["kickoff_local","league_raw","home","away","odd1","oddx","odd2","ov25","btts"]], use_container_width=True, height=320)

st.divider()

# carica storico per indice squadre
st.subheader("Matching campionato")
df_hist = load_global_df()
if df_hist.empty:
    st.stop()

idx = make_team_index(df_hist)
df_up["league_code"] = [
    best_league_code(idx, str(r.league_raw), r.home, r.away) for r in df_up.itertuples(index=False)
]

st.dataframe(
    df_up.assign(kickoff_local=pd.to_datetime(df_up["kickoff_ts"]).dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d %H:%M"))[
        ["kickoff_local","league_raw","league_code","home","away","odd1","oddx","odd2"]
    ].sort_values(["kickoff_local","league_code"]),
    use_container_width=True, height=360
)

st.success(f"Match trovati: {int(df_up['league_code'].notna().sum())} / {len(df_up)}")

st.divider()

# --------------------------------------------------------------------
# Azioni: salva su tabella oppure su Storage come parquet
# --------------------------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Salva su tabella `public.upcoming`**")
    if not SB_SERVICE:
        st.caption("Serve SUPABASE_SERVICE_ROLE_KEY in st.secrets per scrivere la tabella.")
    if st.button("Upsert in tabella", type="primary", disabled=not SB_SERVICE):
        rows = df_up.to_dict(orient="records")
        try:
            # spezza in batch
            for i in range(0, len(rows), 500):
                sb_write.table("upcoming").upsert(rows[i:i+500], on_conflict="kickoff_ts,home,away").execute()
            st.success(f"Inserite/aggiornate {len(rows)} righe in public.upcoming")
        except Exception as e:
            st.error(f"Errore upsert: {e}")

with c2:
    st.markdown("**Salva su Storage come Parquet per giorno**")
    bucket = st.text_input("Bucket Storage", value="upcoming-parquet")
    prefix = st.text_input("Cartella", value="parquet")
    if st.button("Pubblica Parquet"):
        try:
            g = df_up.copy()
            g["kickoff_ts"] = pd.to_datetime(g["kickoff_ts"], utc=True)
            g["kickoff_date"] = g["kickoff_ts"].dt.tz_convert("Europe/Rome").dt.date.astype(str)
            count = 0
            for day, grp in g.groupby("kickoff_date", sort=True):
                buf = io.BytesIO()
                grp.drop(columns=["kickoff_date"]).to_parquet(buf, index=False)
                buf.seek(0)
                path = f"{prefix}/upcoming_{day}.parquet"
                sb_write.storage.from_(bucket).upload(path, buf.read(), {"content-type":"application/octet-stream", "x-upsert":"true"})
                count += len(grp)
            st.success(f"Pubblicate {count} righe in {bucket}/{prefix}")
        except Exception as e:
            st.error(f"Errore upload parquet: {e}")
