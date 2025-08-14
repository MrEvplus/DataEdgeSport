# pre_match.py — versione PRO consolidata
# - Macro KPI Plus con tabelle professionali
# - EV/ROI con glossari e stile
# - Statistiche Squadre: filtro solo per la sezione (default stagione corrente)
# - Live: passaggio contesto + badge
# - Fix robustezza: _first_present, _calc_market_roi
# - Calibrazione 1X2: chart Altair robusto senza transform_fold

from __future__ import annotations

import re
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ========= Moduli esterni opzionali =========
try:
    from squadre import compute_team_macro_stats, render_team_stats_tab
except Exception:
    try:
        from team_stats import compute_team_macro_stats, render_team_stats_tab  # type: ignore
    except Exception:
        compute_team_macro_stats = None  # type: ignore
        def render_team_stats_tab(*args, **kwargs):
            st.info("Modulo 'render_team_stats_tab' non disponibile in questo ambiente.")

try:
    from correct_score import run_correct_score_panel
except Exception:
    try:
        from correct_score_ev_sezione import run_correct_score_ev as run_correct_score_panel  # type: ignore
    except Exception:
        run_correct_score_panel = None  # type: ignore

try:
    from analisi_live_minuto import run_live_minuto_analysis as _run_live
except Exception:
    try:
        from analisi_live_minuto import run_live_minute_analysis as _run_live  # type: ignore
    except Exception:
        _run_live = None  # type: ignore

try:
    from utils import label_match
except Exception:
    # fallback ultra semplice
    def label_match(row):
        try:
            h = float(row.get("Odd home", 2.0))
            a = float(row.get("Odd Away", 2.0))
        except Exception:
            return "Others"
        if h < 1.9 and a > 3.2:
            return "H_MediumFav 1.5-2"
        if a < 1.9 and h > 3.2:
            return "A_MediumFav 1.5-2"
        return "Others"


# ========= Config HUB =========
USE_GLOBAL_FILTERS = True
GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"   # lista stagioni (HUB)


# ========= Altair Theme =========
def _alt_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 12, "titleFontSize": 12, "gridColor": "#223", "labelColor": "#e5e7eb", "titleColor": "#cbd5e1"},
            "legend": {"labelFontSize": 12, "titleFontSize": 12, "labelColor": "#e5e7eb", "titleColor": "#cbd5e1"},
            "title": {"fontSize": 14, "color": "#e5e7eb"},
            "background": "transparent",
        }
    }
try:
    alt.themes.register("app_theme", _alt_theme)
    alt.themes.enable("app_theme")
except Exception:
    pass


# ========= Helper generali =========
def _k(name: str) -> str:
    return f"prematch:{name}"

def _ensure_str(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="string")
    try:
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.astype("string")
    except Exception:
        pass
    return s.astype("string")

def _coerce_float(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float")
    return pd.to_numeric(_ensure_str(s).str.replace(",", ".", regex=False), errors="coerce")

def _first_present(cols, columns: pd.Index) -> str | None:
    """Ritorna il primo nome colonna presente; accetta anche None/non-iterabili."""
    if not cols:
        return None
    try:
        iterable = list(cols)
    except Exception:
        return None
    for c in iterable:
        if c in columns:
            return c
    return None

def _label_from_odds(home_odd: float, away_odd: float) -> str:
    return label_match({"Odd home": home_odd, "Odd Away": away_odd})

def _label_type(label: str | None) -> str:
    if not label:
        return "Both"
    if label.startswith("H_"):
        return "Home"
    if label.startswith("A_"):
        return "Away"
    return "Both"

def _format_value(val: float | int | None, is_roi: bool = False) -> str:
    if val is None or pd.isna(val):
        val = 0.0
    suffix = "%" if is_roi else ""
    if val > 0:
        return f"🟢 +{float(val):.2f}{suffix}"
    if val < 0:
        return f"🔴 {float(val):.2f}{suffix}"
    return f"0.00{suffix}"

def _download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv", key=_k(f"dl:{filename}"))

def _get_qparams() -> dict:
    try:
        return dict(st.query_params)
    except Exception:
        return st.experimental_get_query_params()

def _set_qparams(**kwargs):
    qp = {k: str(v) for k, v in kwargs.items() if v is not None}
    try:
        st.query_params.update(qp)
    except Exception:
        st.experimental_set_query_params(**qp)


# ========= Stagioni helpers =========
def _season_sort_key(s: str) -> int:
    if not isinstance(s, str):
        s = str(s)
    yrs = [int(x) for x in re.findall(r"\d{4}", s)]
    return max(yrs) if yrs else -1

def _seasons_desc(unique_seasons: list) -> list[str]:
    arr = [str(x) for x in unique_seasons if pd.notna(x)]
    return sorted(arr, key=_season_sort_key, reverse=True)

def _pick_current_season(seasons_desc: list[str]) -> list[str]:
    """Stagione in corso: 1 Lug -> 30 Giu (EU style)."""
    if not seasons_desc:
        return []
    today = date.today()
    if today.month >= 7:
        sy, ey = today.year, today.year + 1
    else:
        sy, ey = today.year - 1, today.year
    ey2 = str(ey)[-2:]
    candidates = [f"{sy}/{ey}", f"{sy}-{ey}", f"{sy}/{ey2}", f"{sy}-{ey2}", f"{sy}–{ey}", f"{sy}–{ey2}", str(sy), str(ey)]
    for cand in candidates:
        for s in seasons_desc:
            if s.strip() == cand:
                return [s]
    for s in seasons_desc:
        txt = s.strip()
        if str(sy) in txt or str(ey) in txt:
            return [s]
    return seasons_desc[:1]


# ========= Quote condivise (sincronizzate tra tab) =========
_SHARED_PREFIX = "prematch:shared:"

def _shared_key(name: str) -> str:
    return f"{_SHARED_PREFIX}{name}"

def _init_shared_quotes():
    defaults = {"ov15": 2.00, "ov25": 2.00, "ov35": 2.00, "btts": 2.00}
    for k, v in defaults.items():
        st.session_state.setdefault(_shared_key(k), v)

def _shared_number_input(label: str, shared_name: str, local_key: str,
                         min_value: float = 1.01, step: float = 0.01):
    _init_shared_quotes()
    if local_key not in st.session_state:
        st.session_state[local_key] = float(st.session_state[_shared_key(shared_name)])

    def _on_change():
        st.session_state[_shared_key(shared_name)] = float(st.session_state[local_key])

    return st.number_input(label, min_value=min_value, step=step, key=local_key, on_change=_on_change)

def _get_shared_quotes() -> dict:
    _init_shared_quotes()
    return {
        "ov15": float(st.session_state[_shared_key("ov15")]),
        "ov25": float(st.session_state[_shared_key("ov25")]),
        "ov35": float(st.session_state[_shared_key("ov35")]),
        "btts": float(st.session_state[_shared_key("btts")]),
    }


# ========= League data by Label + cache =========
@st.cache_data(show_spinner=False, ttl=900)
def _league_data_by_label_cached(df_light: pd.DataFrame, label: str | None) -> dict | None:
    df = df_light.copy()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    hft = pd.to_numeric(df["Home Goal FT"], errors="coerce")
    aft = pd.to_numeric(df["Away Goal FT"], errors="coerce")
    df["__res__"] = np.where(hft > aft, "Home Win", np.where(hft < aft, "Away Win", "Draw"))

    group = df.groupby("Label").agg(
        Matches=("Home", "count"),
        HomeWin_pct=("__res__", lambda x: (x == "Home Win").mean() * 100),
        Draw_pct=("__res__", lambda x: (x == "Draw").mean() * 100),
        AwayWin_pct=("__res__", lambda x: (x == "Away Win").mean() * 100),
    ).reset_index()

    if label:
        row = group[group["Label"] == label]
    else:
        row = group
    return row.iloc[0].to_dict() if not row.empty else None


# ========= Back/Lay 1x2 + cache =========
def _calc_back_lay_1x2(df: pd.DataFrame, commission: float = 0.0):
    if df.empty:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, zero, zero, 0

    for c in ("Odd home", "Odd Draw", "Odd Away"):
        if c in df.columns:
            df[c] = _coerce_float(df[c])

    profits_back = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    profits_lay  = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    valid_rows = []

    for _, row in df.iterrows():
        hg = pd.to_numeric(row.get("Home Goal FT"), errors="coerce")
        ag = pd.to_numeric(row.get("Away Goal FT"), errors="coerce")
        if pd.isna(hg) or pd.isna(ag):
            continue
        valid_rows.append((row, int(hg), int(ag)))

    if not valid_rows:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, zero, zero, 0

    for row, hg, ag in valid_rows:
        result = "HOME" if hg > ag else "AWAY" if hg < ag else "DRAW"
        prices = {
            "HOME": float(row.get("Odd home", np.nan)),
            "DRAW": float(row.get("Odd Draw", np.nan)),
            "AWAY": float(row.get("Odd Away", np.nan)),
        }
        for k, v in prices.items():
            if not (v and v > 1.0 and np.isfinite(v)):
                prices[k] = 2.0

        for outcome in ("HOME", "DRAW", "AWAY"):
            p = prices[outcome]
            if result == outcome:
                profits_back[outcome] += (p - 1) * (1 - commission)
            else:
                profits_back[outcome] -= 1
            stake = 1.0 / (p - 1.0)
            if result != outcome:
                profits_lay[outcome] += stake
            else:
                profits_lay[outcome] -= 1.0

    denom = len(valid_rows)
    rois_back = {k: round((v / denom) * 100, 2) for k, v in profits_back.items()}
    rois_lay  = {k: round((v / denom) * 100, 2) for k, v in profits_lay.items()}
    return profits_back, rois_back, profits_lay, rois_lay, denom

@st.cache_data(show_spinner=False, ttl=900)
def _calc_back_lay_1x2_cached(df_light: pd.DataFrame, commission: float = 0.0):
    return _calc_back_lay_1x2(df_light.copy(), commission)


# ========= ROI Over/BTTS + cache (robusto) =========
def _calc_market_roi(df: pd.DataFrame, market: str, price_cols,
                     line: float | None, commission: float, manual_price: float | None = None):
    df = df.dropna(subset=["Home Goal FT", "Away Goal FT"])
    if df.empty:
        return {"Mercato": market, "Quota Media": np.nan, "Esiti %": "0.0%",
                "ROI Back %": "0.0%", "ROI Lay %": "0.0%", "Match Analizzati": 0}

    col = _first_present(price_cols, df.columns)
    odds = _coerce_float(df[col]) if col else pd.Series([np.nan] * len(df), index=df.index)

    if manual_price and (odds.isna() | (odds < 1.01)).all():
        odds = pd.Series([manual_price] * len(df), index=df.index)
    else:
        if manual_price is not None:
            odds = odds.where(odds >= 1.01, manual_price)

    hits = 0; back_profit = 0.0; lay_profit = 0.0; total = 0
    qsum = 0.0; qcount = 0

    for i, row in df.iterrows():
        o = float(odds.loc[i]) if i in odds.index else float("nan")
        if not (o and o >= 1.01 and np.isfinite(o)):
            continue

        hg = pd.to_numeric(row["Home Goal FT"], errors="coerce")
        ag = pd.to_numeric(row["Away Goal FT"], errors="coerce")
        if pd.isna(hg) or pd.isna(ag):
            continue

        total += 1
        qsum += o; qcount += 1

        goals = int(hg) + int(ag)
        if market == "BTTS":
            goal_both = (hg > 0) and (ag > 0)
            if goal_both:
                hits += 1; back_profit += (o - 1) * (1 - commission); lay_profit -= 1
            else:
                lay_profit += 1 / (o - 1); back_profit -= 1
        else:
            assert line is not None
            if goals > line:
                hits += 1; back_profit += (o - 1) * (1 - commission); lay_profit -= 1
            else:
                lay_profit += 1 / (o - 1); back_profit -= 1

    avg_quote = round(qsum / qcount, 2) if qcount > 0 else np.nan
    pct = round((hits / total) * 100, 2) if total > 0 else 0.0
    roi_back = round((back_profit / total) * 100, 2) if total > 0 else 0.0
    roi_lay  = round((lay_profit  / total) * 100, 2) if total > 0 else 0.0

    return {
        "Mercato": market, "Quota Media": avg_quote, "Esiti %": f"{pct}%",
        "ROI Back %": f"{roi_back}%", "ROI Lay %": f"{roi_lay}%", "Match Analizzati": total
    }

@st.cache_data(show_spinner=False, ttl=900)
def _calc_market_roi_cached(df_light: pd.DataFrame, market: str, price_cols: tuple[str, ...],
                            line: float | None, commission: float, manual_price: float | None):
    return _calc_market_roi(df_light.copy(), list(price_cols), line, commission, manual_price)


# ========= Probabilità storiche per EV =========
def _market_prob(df: pd.DataFrame, market: str, line: float | None) -> float:
    if df.empty:
        return 0.0
    hg = pd.to_numeric(df["Home Goal FT"], errors="coerce").fillna(0)
    ag = pd.to_numeric(df["Away Goal FT"], errors="coerce").fillna(0)
    goals = hg + ag
    if market == "BTTS":
        ok = ((hg > 0) & (ag > 0)).mean()
    else:
        ok = (goals > float(line)).mean() if line is not None else 0.0
    return round(float(ok) * 100, 2)

def _quality_label(n: int) -> str:
    if n >= 50: return "ALTO"
    if n >= 20: return "MEDIO"
    return "BASSO"


# ========= EV storico – tabella + Best EV =========
def _build_ev_table(df_home_ctx: pd.DataFrame, df_away_ctx: pd.DataFrame, df_h2h: pd.DataFrame,
                    squadra_casa: str, squadra_ospite: str,
                    quota_ov15: float, quota_ov25: float, quota_ov35: float, quota_btts: float):
    markets = [("Over 1.5", 1.5, quota_ov15), ("Over 2.5", 2.5, quota_ov25), ("Over 3.5", 3.5, quota_ov35), ("BTTS", None, quota_btts)]
    rows, candidates_for_best = [], []

    for name, line, q in markets:
        p_home = _market_prob(df_home_ctx, name, line)
        p_away = _market_prob(df_away_ctx, name, line)
        p_blnd = round((p_home + p_away) / 2, 2) if (p_home > 0 or p_away > 0) else 0.0
        p_h2h  = _market_prob(df_h2h, name, line)

        ev_home = round(q * (p_home / 100) - 1, 2)
        ev_away = round(q * (p_away / 100) - 1, 2)
        ev_blnd = round(q * (p_blnd / 100) - 1, 2)
        ev_h2h  = round(q * (p_h2h / 100) - 1, 2)

        n_h = len(df_home_ctx); n_a = len(df_away_ctx); n_h2h = len(df_h2h)
        qual_blnd = _quality_label(n_h + n_a)
        qual_h2h  = _quality_label(n_h2h)

        rows.append({
            "Mercato": name, "Quota": q,
            f"{squadra_casa} @Casa %": p_home, f"EV {squadra_casa}": ev_home,
            f"{squadra_ospite} @Trasferta %": p_away, f"EV {squadra_ospite}": ev_away,
            "Blended %": p_blnd, "EV Blended": ev_blnd, "Qualità Blended": qual_blnd,
            "Head-to-Head %": p_h2h, "EV H2H": ev_h2h, "Qualità H2H": qual_h2h,
            "Match H": n_h, "Match A": n_a, "Match H2H": n_h2h,
        })

        candidates_for_best.extend([
            {"scope": "Blended", "mercato": name, "quota": q, "prob": p_blnd, "ev": ev_blnd, "campione": n_h + n_a, "qualita": qual_blnd},
            {"scope": "Head-to-Head", "mercato": name, "quota": q, "prob": p_h2h, "ev": ev_h2h, "campione": n_h2h, "qualita": qual_h2h},
        ])

    df_ev = pd.DataFrame(rows)

    best = None
    for c in sorted(candidates_for_best, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c; break
    return df_ev, best

@st.cache_data(show_spinner=False, ttl=900)
def _build_ev_table_cached(home_ctx_light: pd.DataFrame, away_ctx_light: pd.DataFrame, h2h_light: pd.DataFrame,
                           squadra_casa: str, squadra_ospite: str,
                           quota_ov15: float, quota_ov25: float, quota_ov35: float, quota_btts: float):
    return _build_ev_table(home_ctx_light.copy(), away_ctx_light.copy(), h2h_light.copy(),
                           squadra_casa, squadra_ospite, quota_ov15, quota_ov25, quota_ov35, quota_btts)


# ================================================================
# ===============  MACRO KPI PLUS (tabelle pro)  =================
# ================================================================
_MIN_RE = re.compile(r"\d+")

def _mins_list(cell):
    if cell is None:
        return []
    s = str(cell)
    vals = []
    for m in _MIN_RE.findall(s):
        try:
            v = int(m)
            if 0 < v <= 130:
                vals.append(v)
        except Exception:
            pass
    return sorted(vals)

def _first_goal_side_min(row):
    h_raw = row.get("minuti goal segnato home", row.get("Minuti Goal Home"))
    a_raw = row.get("minuti goal segnato away", row.get("Minuti Goal Away"))
    if (h_raw is None or str(h_raw).strip() == "") and any(c in row for c in [f"gh{i}" for i in range(1,10)]):
        hs = [row[c] for c in [f"gh{i}" for i in range(1,10)] if c in row and pd.notna(row[c])]
        h_raw = ";".join(str(x) for x in hs) if hs else ""
    if (a_raw is None or str(a_raw).strip() == "") and any(c in row for c in [f"ga{i}" for i in range(1,10)]):
        aa = [row[c] for c in [f"ga{i}" for i in range(1,10)] if c in row and pd.notna(row[c])]
        a_raw = ";".join(str(x) for x in aa) if aa else ""
    hm, am = _mins_list(h_raw), _mins_list(a_raw)
    h1 = hm[0] if hm else None; a1 = am[0] if am else None
    if h1 is None and a1 is None: return (None, None)
    if h1 is None: return ("Away", a1)
    if a1 is None: return ("Home", h1)
    if h1 < a1: return ("Home", h1)
    if a1 < h1: return ("Away", a1)
    return (None, None)

def _team_df(df, team, side):
    return df[df["Home"].astype(str).eq(str(team))] if side=="Home" else df[df["Away"].astype(str).eq(str(team))]

def _w_d_l(df, side):
    if df.empty: return (0.0,0.0,0.0)
    if side == "Home":
        w = (df["Home Goal FT"] > df["Away Goal FT"]).mean()
        d = (df["Home Goal FT"] == df["Away Goal FT"]).mean()
        l = (df["Home Goal FT"] < df["Away Goal FT"]).mean()
    else:
        w = (df["Away Goal FT"] > df["Home Goal FT"]).mean()
        d = (df["Away Goal FT"] == df["Home Goal FT"]).mean()
        l = (df["Away Goal FT"] < df["Home Goal FT"]).mean()
    w, d, l = [0 if np.isnan(x) else float(x) for x in (w,d,l)]
    s = w+d+l
    return (w/s, d/s, l/s) if s>0 else (0.0,0.0,0.0)

def _gf_ga(df, side):
    if df.empty: return (0.0, 0.0)
    return (df["Home Goal FT"].mean(), df["Away Goal FT"].mean()) if side=="Home" else (df["Away Goal FT"].mean(), df["Home Goal FT"].mean())

def _shots(df: pd.DataFrame, side: str):
    """
    Ritorna:
      - shots_for / shots_against   (somme sui match)
      - sot_for / sot_against       (somme sui match)
      - pace_avg                    (media per match di tiri totali = for+against)
    Supporta sia le colonne short-code (suth, suta, sutht, su tat)
    sia eventuali etichette 'umane' se presenti.
    """
    # Possibili alias (primo che troviamo viene usato)
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    h_tot  = pick("suth",  "Tiri Totali Home FT")
    a_tot  = pick("suta",  "Tiri Totali Away FT")
    h_sot  = pick("sutht", "Tiri in Porta Home FT")
    a_sot  = pick("sutat", "Tiri in Porta Away FT")

    # Se mancano colonne, creiamo serie vuote (NaN) così non esplodono i calcoli
    def _series_or_nan(col):
        if col is None or col not in df.columns:
            return pd.Series([np.nan] * len(df), index=df.index, dtype="float")
        # coercion robusta: gestisce stringhe, virgole, ecc.
        s = pd.to_numeric(df[col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
        return s

    # Home / Away totali e SOT
    H_tot  = _series_or_nan(h_tot)
    A_tot  = _series_or_nan(a_tot)
    H_sotS = _series_or_nan(h_sot)
    A_sotS = _series_or_nan(a_sot)

    # Dal punto di vista della squadra (side)
    if side == "Home":
        sf   = H_tot
        sa   = A_tot
        sotf = H_sotS
        sota = A_sotS
    else:
        sf   = A_tot
        sa   = H_tot
        sotf = A_sotS
        sota = H_sotS

    pace_avg = np.nanmean(sf + sa)  # media tiri totali per match
    return {
        "shots_for":    float(np.nansum(sf)),
        "shots_against":float(np.nansum(sa)),
        "sot_for":      float(np.nansum(sotf)),
        "sot_against":  float(np.nansum(sota)),
        "pace_avg":     float(pace_avg) if np.isfinite(pace_avg) else np.nan,
    }

def _btts_over(df):
    if df.empty: return (0.0, 0.0)
    btts = ((df["Home Goal FT"]>0) & (df["Away Goal FT"]>0)).mean()
    over25 = ((df["Home Goal FT"] + df["Away Goal FT"]) > 2.5).mean()
    return (0 if np.isnan(btts) else float(btts), 0 if np.isnan(over25) else float(over25))

def _reliability_stars(n):
    if n >= 45: return "★★★★★"
    if n >= 30: return "★★★★☆"
    if n >= 20: return "★★★☆☆"
    if n >= 10: return "★★☆☆☆"
    if n >= 5:  return "★☆☆☆☆"
    return "—"

def _momentum_block(df, team, side, last_n=8):
    df_side = _team_df(df, team, side)
    n_all = len(df_side)
    if n_all == 0: return None
    w0, d0, l0 = _w_d_l(df_side, side); gf0, ga0 = _gf_ga(df_side, side)
    if "Data" in df_side.columns: df_side = df_side.sort_values("Data")
    df_last = df_side.tail(last_n)
    w1, d1, l1 = _w_d_l(df_last, side); gf1, ga1 = _gf_ga(df_last, side)
    delta = {"Δ Win%": (w1 - w0) * 100, "Δ Draw%": (d1 - d0) * 100, "Δ Loss%": (l1 - l0) * 100, "Δ GF": gf1 - gf0, "Δ GA": ga1 - ga0}
    elo_col = "ELO Home" if side=="Home" else "ELO Away"
    form_col = "Form Home" if side=="Home" else "Form Away"
    elo_delta = None; form_delta = None
    if elo_col in df_side.columns:
        e0 = df_side[elo_col].mean(); e1 = df_last[elo_col].mean()
        if not (np.isnan(e0) or np.isnan(e1)): elo_delta = e1 - e0
    if form_col in df_side.columns:
        f0 = df_side[form_col].mean(); f1 = df_last[form_col].mean()
        if not (np.isnan(f0) or np.isnan(f1)): form_delta = f1 - f0
    rel = _reliability_stars(len(df_last))
    return {"n_all": n_all, "n_last": len(df_last), "delta": delta, "elo_delta": elo_delta, "form_delta": form_delta, "rel": rel}

def _first_goal_tables(df, team_home, team_away):
    if df.empty: return None
    work = df.copy()
    if "Data" in work.columns: work = work.sort_values("Data")
    who, minute = [], []
    for _, r in work.iterrows():
        s, m = _first_goal_side_min(r); who.append(s); minute.append(m)
    work["first_side"] = who; work["first_min"]  = minute

    def _outcome(row):
        if row["Home Goal FT"] > row["Away Goal FT"]: return "1"
        if row["Home Goal FT"] < row["Away Goal FT"]: return "2"
        return "X"
    work["FT"] = work.apply(_outcome, axis=1)

    t_home = work[work["first_side"] == "Home"]; t_away = work[work["first_side"] == "Away"]

    def _pivot(a):
        if a.empty: return pd.DataFrame({"Esito": ["1","X","2"], "Freq %": [0,0,0]})
        p = (a["FT"].value_counts(normalize=True) * 100).reindex(["1","X","2"]).fillna(0).reset_index()
        p.columns = ["Esito","Freq %"]; return p

    tab_home = _pivot(t_home); tab_away = _pivot(t_away)

    bins = [(0,15),(16,30),(31,45),(46,60),(61,75),(76,90)]
    labels = [f"{a}-{b}" for a,b in bins]
    counts = {lab:0 for lab in labels}
    mm = [m for m in work["first_min"].tolist() if isinstance(m, (int,float))]
    for m in mm:
        for (a,b),lab in zip(bins, labels):
            if a < m <= b:
                counts[lab] += 1; break
    hist = pd.DataFrame({"Finestra": labels, "Occorrenze": [counts[l] for l in labels]})
    return {"home_first": tab_home, "away_first": tab_away, "hist": hist}

def _style_rhythm_block(df: pd.DataFrame, team: str, side: str):
    if side == "Home":
        df_side = df[df["Home"].astype(str) == str(team)]
        gf_col, ga_col = "Home Goal FT", "Away Goal FT"
    else:
        df_side = df[df["Away"].astype(str) == str(team)]
        gf_col, ga_col = "Away Goal FT", "Home Goal FT"

    if df_side.empty:
        return None

    # Coercion numerica sui gol
    gf = pd.to_numeric(df_side[gf_col], errors="coerce").fillna(0)
    ga = pd.to_numeric(df_side[ga_col], errors="coerce").fillna(0)

    shots = _shots(df_side, side)

    # Conversione: gol / SOT (solo se SOT>0)
    sot_for = shots["sot_for"]
    conv = np.nan
    if sot_for and sot_for > 0:
        conv = float(gf.sum()) / float(sot_for)

    # Save% approx: 1 - gol subiti / SOT contro (solo se SOT contro >0)
    sot_ag = shots["sot_against"]
    savep = np.nan
    if sot_ag and sot_ag > 0:
        savep = 1.0 - (float(ga.sum()) / float(sot_ag))

    n = int(len(df_side))
    # Pace per match: media tiri totali (for+against)
    pace_pm = shots["pace_avg"]

    # BTTS / Over 2.5 (percentuali)
    btts = ((pd.to_numeric(df_side["Home Goal FT"], errors="coerce") > 0) &
            (pd.to_numeric(df_side["Away Goal FT"], errors="coerce") > 0)).mean()
    over25 = ((pd.to_numeric(df_side["Home Goal FT"], errors="coerce") +
               pd.to_numeric(df_side["Away Goal FT"], errors="coerce")) > 2.5).mean()

    return {
        "pace": pace_pm,                     # tiri totali per match
        "conv": conv,                        # frazione (es. 0.25 → 25%)
        "save": savep,                       # frazione (es. 0.70 → 70%)
        "btts": float(0 if np.isnan(btts) else btts),
        "over25": float(0 if np.isnan(over25) else over25),
        "n": n,
    }


def _calibration_one(df, market: str, k_bins=5):
    col_map = {"Home": "Odd home", "Draw": "Odd Draw", "Away": "Odd Away"}
    col = col_map.get(market)
    if not col or col not in df.columns:
        return pd.DataFrame(columns=["Bin Quota","Implied %","Observed %","Gap %","N","Brier"])
    d = df[[col, "Home Goal FT", "Away Goal FT"]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=["Bin Quota","Implied %","Observed %","Gap %","N","Brier"])
    if market == "Home": d["y"] = (d["Home Goal FT"] > d["Away Goal FT"]).astype(int)
    elif market == "Away": d["y"] = (d["Away Goal FT"] > d["Home Goal FT"]).astype(int)
    else: d["y"] = (d["Home Goal FT"] == d["Away Goal FT"]).astype(int)
    d["p"] = 1.0 / d[col].astype(float); d = d[(d["p"]>0) & (d["p"]<=1.0)]
    qs = np.linspace(0, 1, k_bins+1); edges = np.unique(np.round(d[col].quantile(qs).values, 4))
    if len(edges) < 3: edges = np.array([d[col].min(), d[col].median(), d[col].max()])
    d["bin"] = pd.cut(d[col], bins=edges, include_lowest=True, duplicates="drop")
    res = []
    for b, g in d.groupby("bin"):
        if g.empty: continue
        implied = g["p"].mean(); obs = g["y"].mean()
        brier = np.mean((g["p"] - g["y"])**2)
        res.append({"Bin Quota": str(b),"Implied %": round(implied*100,1),"Observed %": round(obs*100,1),
                    "Gap %": round((obs-implied)*100,1), "N": int(len(g)), "Brier": round(brier,4)})
    return pd.DataFrame(res)

# ========= helpers visuali per tabelle pro =========
def _df_style_positive_negative(df: pd.DataFrame, pos_good_cols: list[str] = None, neg_good_cols: list[str] = None):
    pos_good_cols = pos_good_cols or []; neg_good_cols = neg_good_cols or []
    def _col(v, good="pos"):
        if pd.isna(v): return ""
        try: x = float(v)
        except Exception: return ""
        if good == "pos":
            if x > 0: return "background-color:#073b1d;color:#c7ffe4;"
            if x < 0: return "background-color:#4c0a0a;color:#ffd9d9;"
        else:
            if x < 0: return "background-color:#073b1d;color:#c7ffe4;"
            if x > 0: return "background-color:#4c0a0a;color:#ffd9d9;"
        return "background-color:#1f2937;color:#e5e7eb;"
    sty = df.style
    for c in pos_good_cols:
        if c in df.columns: sty = sty.applymap(lambda v: _col(v, "pos"), subset=[c])
    for c in neg_good_cols:
        if c in df.columns: sty = sty.applymap(lambda v: _col(v, "neg"), subset=[c])
    return sty.set_properties(**{"font-size":"12px"})

# ========= Stili/Glossari per EV & ROI =========
def _style_ev(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    def _col_ev(v):
        try: f = float(v)
        except Exception: return ""
        if f > 0:  return "background-color:#073b1d;color:#c7ffe4;"
        if f < 0:  return "background-color:#4c0a0a;color:#ffd9d9;"
        return "background-color:#1f2937;color:#e5e7eb;"
    ev_cols = [c for c in df.columns if c.lower().startswith("ev")]
    sty = df.style
    for c in ev_cols:
        sty = sty.applymap(_col_ev, subset=[c])
    return sty.set_properties(**{"font-size":"12px"})

def _render_glossary_ev():
    with st.expander("ℹ️ Glossario — EV storico", expanded=False):
        st.markdown("""
- **Prob %**: stima storica della probabilità dell’esito (Home@Casa, Away@Trasferta, Blended, Head-to-Head).
- **EV = quota × p − 1**: valore atteso unitario. EV>0 = vantaggio.
- **Qualità**: solidità del campione (H/A = somma match di contesto; H2H = scontri diretti).
- **Best EV**: opportunità migliore tra *Blended* e *Head-to-Head* con EV>0.
        """)

def _render_glossary_roi():
    with st.expander("ℹ️ Glossario — ROI mercati", expanded=False):
        st.markdown("""
- **Quota Media**: media delle quote usate (o la quota manuale se i prezzi storici non erano disponibili).
- **Esiti %**: percentuale di volte in cui l’evento si è verificato.
- **ROI Back/Lay %**: profitto medio per scommessa (commissione exchange applicata).
- **Scope**:
  - *Campionato/Label*: tutte le partite del campionato (o solo del label).
  - *Squadra @Casa/Trasferta*: partite della squadra nel relativo contesto.
        """)

# ========= Macro KPI Plus (tabelle pro) =========
def render_macro_kpi_plus(df_ctx: pd.DataFrame, home_team: str, away_team: str):
    st.markdown("### 🔬 Approfondimenti Macro KPI (complementari a 1X2)")

    # 1) Momentum & Affidabilità
    st.subheader("1) 📈 Momentum & Affidabilità")
    last_n = st.slider("Ultime N gare per Momentum", 6, 12, 8, key="mom_lastn")

    mH = _momentum_block(df_ctx, home_team, "Home", last_n=last_n)
    mA = _momentum_block(df_ctx, away_team, "Away", last_n=last_n)
    c1, c2 = st.columns(2)

    def _render_mom(block, title):
        if block is None:
            st.info(f"Nessun dato per **{title}**.")
            return

        dlt = block.get("delta", {}) or {}
        base = {
            "Δ Win%":  float(dlt.get("Δ Win%", 0.0)),
            "Δ Draw%": float(dlt.get("Δ Draw%", 0.0)),
            "Δ Loss%": float(dlt.get("Δ Loss%", 0.0)),
            "Δ GF":    float(dlt.get("Δ GF", 0.0)),
            "Δ GA":    float(dlt.get("Δ GA", 0.0)),
        }
        if block.get("elo_delta") is not None:
            base["Δ ELO"] = float(block["elo_delta"])
        if block.get("form_delta") is not None:
            base["Δ Form"] = float(block["form_delta"])

        base["N (ultime)"] = int(block.get("n_last", 0))
        base["N (tot)"]   = int(block.get("n_all", 0))
        base["Affidabilità"] = block.get("rel", "—")

        dfm = pd.DataFrame([base])

        # --- Stile + FORMATTING numerico con segno e decimali (fix "+.1f") ---
        sty = _df_style_positive_negative(
            dfm,
            pos_good_cols=["Δ Win%","Δ GF","Δ ELO","Δ Form"],
            neg_good_cols=["Δ Loss%","Δ GA"]
        ).format({
            "Δ Win%":  "{:+.1f}",
            "Δ Draw%": "{:+.1f}",
            "Δ Loss%": "{:+.1f}",
            "Δ GF":    "{:+.2f}",
            "Δ GA":    "{:+.2f}",
            "Δ ELO":   "{:+.1f}",
            "Δ Form":  "{:+.2f}",
            "N (ultime)": "{:.0f}",
            "N (tot)":    "{:.0f}",
        })
        # ----------------------------------------------------------------------

        st.caption(
            f"**{title}** — campione: {base['N (tot)']} • recenti: {base['N (ultime)']} • affidabilità: {base['Affidabilità']}"
        )
        st.dataframe(
            sty,
            use_container_width=True,
            height=88,
            column_config={
                "Δ Win%":   st.column_config.NumberColumn(format="+.1f"),
                "Δ Draw%":  st.column_config.NumberColumn(format="+.1f"),
                "Δ Loss%":  st.column_config.NumberColumn(format="+.1f"),
                "Δ GF":     st.column_config.NumberColumn(format="+.2f"),
                "Δ GA":     st.column_config.NumberColumn(format="+.2f"),
                "Δ ELO":    st.column_config.NumberColumn(format="+.1f"),
                "Δ Form":   st.column_config.NumberColumn(format="+.2f"),
                "N (ultime)": st.column_config.NumberColumn(format="%.0f"),
                "N (tot)":    st.column_config.NumberColumn(format="%.0f"),
            },
        )

    with c1: _render_mom(mH, f"{home_team} @casa")
    with c2: _render_mom(mA, f"{away_team} @trasferta")

    st.divider()

    # 2) Primo Gol → Esito
    st.subheader("2) ⏱️ Primo Gol → Esito (Pressione & Rimonte)")
    fg = _first_goal_tables(df_ctx, home_team, away_team)
    if fg is None:
        st.info("Dati minuti gol non disponibili.")
    else:
        c1, c2 = st.columns(2)
        def _render_fg(df_, title):
            df_ = df_.copy()
            df_["Freq %"] = df_["Freq %"].astype(float)
            st.caption(title)
            st.dataframe(
                df_, use_container_width=True, height=150,
                column_config={
                    "Esito": st.column_config.TextColumn(),
                    "Freq %": st.column_config.ProgressColumn(format="%.1f%%", min_value=0.0, max_value=100.0),
                },
            )
        with c1: _render_fg(fg["home_first"], f"Se segna prima **{home_team}** (Home-first)")
        with c2: _render_fg(fg["away_first"], f"Se segna prima **{away_team}** (Away-first)")

        base = alt.Chart(fg["hist"]).mark_bar().encode(
            x=alt.X("Finestra:N", title="Finestra minuto"),
            y=alt.Y("Occorrenze:Q", title="Occorrenze"),
            tooltip=["Finestra","Occorrenze"]
        ).properties(height=160, width="container")
        st.altair_chart(base, use_container_width=True)

    st.divider()

    # 3) Stile & Ritmo
    st.subheader("3) ⚙️ Stile & Ritmo (pace / precisione SOT→Gol / save% / BTTS / Over2.5)")
    c1, c2 = st.columns(2)
    def _render_style(block, title):
        if block is None:
            st.info(f"Nessun dato per **{title}**.")
            return
        dfk = pd.DataFrame([{
            "Pace (tiri tot/match)": round(block["pace"], 2) if pd.notna(block["pace"]) else np.nan,
            "Precisione SOT→Gol %": round(block["conv"]*100, 1) if block.get("conv") is not None and not np.isnan(block["conv"]) else np.nan,
            "Save % (approx)":      round(block["save"]*100, 1) if block.get("save") is not None and not np.isnan(block["save"]) else np.nan,
            "BTTS %":               round(block["btts"]*100, 1),
            "Over 2.5 %":           round(block["over25"]*100, 1),
            "Campione":             int(block["n"])
        }])
        st.caption(title)
        st.dataframe(
            dfk, use_container_width=True, height=80,
            column_config={
                "Pace (tiri tot/match)": st.column_config.NumberColumn(format="%.2f"),
                "Precisione SOT→Gol %": st.column_config.NumberColumn(format="%.1f"),
                "Save % (approx)":      st.column_config.NumberColumn(format="%.1f"),
                "BTTS %":               st.column_config.NumberColumn(format="%.1f"),
                "Over 2.5 %":           st.column_config.NumberColumn(format="%.1f"),
                "Campione":             st.column_config.NumberColumn(format="%.0f"),
            },
        )
    with c1: _render_style(_style_rhythm_block(df_ctx, home_team, "Home"), f"**{home_team} @casa**")
    with c2: _render_style(_style_rhythm_block(df_ctx, away_team, "Away"), f"**{away_team} @trasferta**")

    st.divider()

    # 4) Calibrazione 1X2
    st.subheader("4) 🎯 Calibration 1X2 (Quote → Outcome)")
    with st.expander("Mostra tabelle calibrazione per Home / Draw / Away", expanded=True):
        tabs = st.tabs(["Home", "Draw", "Away"])
        def _render_calib(market: str, t_idx: int):
            dfc = _calibration_one(df_ctx, market)
            with tabs[t_idx]:
                st.caption(f"Calibrazione {market}: **implied vs observed**, gap e Brier score.")
                if dfc.empty:
                    st.info("Dati insufficienti.")
                else:
                    st.dataframe(
                        dfc, use_container_width=True, height=220,
                        column_config={
                            "Implied %": st.column_config.NumberColumn(format="%.1f"),
                            "Observed %": st.column_config.NumberColumn(format="%.1f"),
                            "Gap %":      st.column_config.NumberColumn(format="%+.1f"),
                            "N":          st.column_config.NumberColumn(format="%.0f"),
                            "Brier":      st.column_config.NumberColumn(format="%.4f"),
                        },
                    )
                    # Chart robusto senza transform_fold
                    df_long = dfc.melt(
                        id_vars=["Bin Quota"],
                        value_vars=["Implied %", "Observed %"],
                        var_name="Serie",
                        value_name="Valore",
                    )
                    df_long["Valore"] = pd.to_numeric(df_long["Valore"], errors="coerce")
                    df_long = df_long.dropna(subset=["Valore"])
                    ch = alt.Chart(df_long).mark_line(point=True).encode(
                        x=alt.X("Bin Quota:N", title="Bin di quota"),
                        y=alt.Y("Valore:Q", title="%"),
                        color=alt.Color("Serie:N", legend=alt.Legend(orient="bottom")),
                        tooltip=["Bin Quota", "Serie", alt.Tooltip("Valore:Q", format=".1f")],
                    ).properties(height=180, width="container")
                    st.altair_chart(ch, use_container_width=True)
        _render_calib("Home", 0)
        _render_calib("Draw", 1)
        _render_calib("Away", 2)


# ========= ENTRY POINT =========
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title("📊 Pre-Match – Analisi per Trader Professionisti")

    qin = _get_qparams()
    _init_shared_quotes()

    if "country" not in df.columns:
        st.error("Dataset senza colonna 'country'."); st.stop()

    df = df.copy()
    df["country"] = _ensure_str(df["country"]).str.strip().str.upper()

    # Campionato & Stagioni: SOLO dall’Hub
    if USE_GLOBAL_FILTERS:
        league = (st.session_state.get(GLOBAL_CHAMP_KEY) or db_selected or "").strip().upper()
        seasons_from_hub = st.session_state.get(GLOBAL_SEASONS_KEY)
        df = df[df["country"] == league] if league else df
        if seasons_from_hub and "Stagione" in df.columns:
            df = df[df["Stagione"].astype(str).isin([str(s) for s in seasons_from_hub])]
        txt_seas = ", ".join([str(s) for s in seasons_from_hub]) if seasons_from_hub else "tutte"
        st.caption(f"Contesto: **{league}** • **Stagioni**: {txt_seas}")
        df_league_all = df.copy()
        seasons_selected = seasons_from_hub or None
    else:
        league = (db_selected or "").strip().upper()
        df = df[df["country"] == league]
        if df.empty: st.warning(f"Nessun dato per '{league}'."); st.stop()
        df_league_all = df.copy()
        seasons_selected = None
        if "Stagione" in df.columns:
            seasons_desc = _seasons_desc(df["Stagione"].dropna().unique().tolist())
            latest = seasons_desc[0] if seasons_desc else None
            with st.expander("⚙️ Filtro Stagioni", expanded=True):
                st.markdown("Intervallo rapido parte dalla **stagione più recente**.")
                colA, colB = st.columns([2, 1])
                with colA:
                    seasons_selected = st.multiselect("Seleziona stagioni (Manuale)", options=seasons_desc, default=[latest] if latest else [], key=_k("stagioni_manual"))
                with colB:
                    preset = st.selectbox("Intervallo rapido", ["Tutte","Stagione in corso","Ultime 10","Ultime 5","Ultime 3","Ultime 2","Ultime 1"], index=0, key=_k("stagioni_preset"))
                    if preset == "Stagione in corso":
                        seasons_selected = _pick_current_season(seasons_desc)
                    elif preset != "Tutte" and seasons_desc:
                        try: n = int(preset.split()[-1])
                        except Exception: n = 1
                        seasons_selected = seasons_desc[:n]
                if seasons_selected:
                    st.caption(f"Stagioni attive: **{', '.join(seasons_selected)}**")
                    df = df[df["Stagione"].astype(str).isin(seasons_selected)]
                else:
                    st.caption("Filtro stagioni: **Tutte**"); seasons_selected = None

    df["Home"] = _ensure_str(df["Home"]).str.strip()
    df["Away"] = _ensure_str(df["Away"]).str.strip()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # Ricerca rapida + assegnazione
    all_teams = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))
    if st.session_state.get(_k("swap_trigger")):
        st.session_state.pop(_k("squadra_casa"), None)
        st.session_state.pop(_k("squadra_ospite"), None)
        st.session_state[_k("swap_trigger")] = False

    with st.expander("🔎 Ricerca rapida squadre (assegna come Casa/Ospite)", expanded=True):
        col_search, col_role, col_btn = st.columns([2, 1, 1])
        with col_search:
            team_filter = st.text_input("Cerca squadra", value=qin.get("q", ""), key=_k("search_team"))
            if team_filter:
                flt = str(team_filter).strip().lower()
                filtered = [t for t in all_teams if flt in t.lower()]
                if not filtered:
                    st.info("Nessuna squadra trovata. Mostro tutte."); filtered = all_teams
            else:
                filtered = all_teams
            selected_from_search = st.selectbox("Risultati ricerca", options=filtered, key=_k("search_pick"))
        with col_role:
            role = st.radio("Assegna a", options=["Casa", "Ospite"], horizontal=True, key=_k("search_role"))
        with col_btn:
            if st.button("Assegna", key=_k("assign_btn"), use_container_width=True):
                try:
                    if role == "Casa":
                        other = st.session_state.get(_k("squadra_ospite"))
                        if other == selected_from_search:
                            st.warning("⚠️ La stessa squadra è già selezionata come Ospite.")
                        else:
                            st.session_state[_k("squadra_casa")] = selected_from_search
                    else:
                        other = st.session_state.get(_k("squadra_casa"))
                        if other == selected_from_search:
                            st.warning("⚠️ La stessa squadra è già selezionata come Casa.")
                        else:
                            st.session_state[_k("squadra_ospite")] = selected_from_search
                    _set_qparams(q=team_filter or "")
                    try: st.rerun()
                    except Exception: st.experimental_rerun()
                except Exception:
                    st.warning("Impossibile assegnare la squadra. Riprova.")

    # Selettori Squadre finali + Swap
    default_home = 0
    default_away = 1 if len(all_teams) > 1 else 0
    col_sel1, col_swap, col_sel2 = st.columns([2, 0.5, 2])
    with col_sel1:
        squadra_casa = st.selectbox("Seleziona Squadra Casa", options=all_teams, index=default_home, key=_k("squadra_casa"))
    with col_swap:
        st.write("")
        if st.button("🔁 Inverti", use_container_width=True, key=_k("swap")):
            current_home = st.session_state.get(_k("squadra_casa"), all_teams[default_home])
            current_away = st.session_state.get(_k("squadra_ospite"), all_teams[default_away])
            if current_home == current_away:
                st.warning("⚠️ Casa e Ospite sono uguali.")
            else:
                _set_qparams(home=current_away, away=current_home)
                st.session_state[_k("swap_trigger")] = True
                try: st.rerun()
                except Exception: st.experimental_rerun()
    with col_sel2:
        squadra_ospite = st.selectbox("Seleziona Squadra Ospite", options=all_teams, index=default_away, key=_k("squadra_ospite"))

    if squadra_casa == squadra_ospite:
        st.error("❌ Casa e Ospite non possono essere la stessa squadra."); return

    # Quote 1X2 + Over/BTTS condivise
    st.markdown("Le quote 1X2 **classificano** il match (label). Le **quote Over/BTTS** qui sotto sono **condivise** in tutti i tab.")
    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Quota Vincente Casa", min_value=1.01, step=0.01, value=2.00, key=_k("quota_home"))
        st.caption(f"Prob. Casa ({squadra_casa}): **{round(100/odd_home, 2)}%**")
    with c2:
        odd_draw = st.number_input("Quota Pareggio", min_value=1.01, step=0.01, value=3.20, key=_k("quota_draw"))
        st.caption(f"Prob. Pareggio: **{round(100/odd_draw, 2)}%**")
    with c3:
        odd_away = st.number_input("Quota Vincente Ospite", min_value=1.01, step=0.01, value=3.80, key=_k("quota_away"))
        st.caption(f"Prob. Ospite ({squadra_ospite}): **{round(100/odd_away, 2)}%**")

    st.markdown("### Quote Over/BTTS (condivise)")
    q1, q2, q3, q4 = st.columns(4)
    with q1: _shared_number_input("Quota Over 1.5", "ov15", _k("shared:q_ov15"))
    with q2: _shared_number_input("Quota Over 2.5", "ov25", _k("shared:q_ov25"))
    with q3: _shared_number_input("Quota Over 3.5", "ov35", _k("shared:q_ov35"))
    with q4: _shared_number_input("Quota BTTS",     "btts", _k("shared:q_btts"))

    label = _label_from_odds(float(odd_home), float(odd_away))
    label_t = _label_type(label)

    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown(f"**Campionato:** `{league}`")
    header_cols[1].markdown(f"**Label:** `{label}`")
    header_cols[2].markdown(f"**Home:** `{squadra_casa}`")
    header_cols[3].markdown(f"**Away:** `{squadra_ospite}`")

    if label == "Others" or label not in set(df["Label"]):
        st.info("⚠️ Nessuna partita trovata per questo label: uso l'intero campionato.")
        label = None

    # Tabs
    tab_1x2, tab_roi, tab_ev, tab_stats, tab_cs, tab_live = st.tabs(
        ["1X2", "ROI mercati", "EV storico squadre", "Statistiche squadre", "Correct Score", "Live da Minuto"]
    )

    # === TAB 1: 1X2 + Macro KPI (+ Plus) ===
    with tab_1x2:
        st.markdown("**Cosa vedi**: confronto 1X2 tra campionato/label e le due squadre nel loro contesto (Home@Casa, Away@Trasferta). **Win%** = frequenze storiche; **Back/Lay ROI** = profitti medi per scommessa.")

        df_league_scope = df[df["Label"] == label] if label else df
        cols_1x2 = ["Home","Away","Home Goal FT","Away Goal FT","Odd home","Odd Draw","Odd Away","Label"]
        df_league_light = df_league_scope[cols_1x2].copy()
        profits_back, rois_back, profits_lay, rois_lay, matches_league = _calc_back_lay_1x2_cached(df_league_light)

        league_stats = _league_data_by_label_cached(
            df[["Home","Away","Home Goal FT","Away Goal FT","Label"]].copy(), label
        )

        row_league = {
            "LABEL": "League (Label)" if label else "League (All)",
            "MATCHES": matches_league,
            "BACK WIN% HOME": round(league_stats["HomeWin_pct"], 2) if league_stats else 0,
            "BACK WIN% DRAW": round(league_stats["Draw_pct"], 2) if league_stats else 0,
            "BACK WIN% AWAY": round(league_stats["AwayWin_pct"], 2) if league_stats else 0,
        }
        for outcome in ("HOME", "DRAW", "AWAY"):
            row_league[f"BACK PTS {outcome}"] = _format_value(profits_back[outcome])
            row_league[f"BACK ROI% {outcome}"] = _format_value(rois_back[outcome], is_roi=True)
            row_league[f"Lay pts {outcome}"] = _format_value(profits_lay[outcome])
            row_league[f"lay ROI% {outcome}"] = _format_value(rois_lay[outcome], is_roi=True)

        rows = [row_league]

        # Casa
        if label and label_t in ("Home", "Both"):
            df_home = df[(df["Label"] == label) & (df["Home"] == squadra_casa)]
            if df_home.empty:
                df_home = df[df["Home"] == squadra_casa]
                st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_casa}.")
        else:
            df_home = df[df["Home"] == squadra_casa]
        df_home_light = df_home[cols_1x2].copy()
        profits_back, rois_back, profits_lay, rois_lay, matches_home = _calc_back_lay_1x2_cached(df_home_light)
        row_home = {"LABEL": squadra_casa, "MATCHES": matches_home}
        if matches_home > 0:
            wins = int((pd.to_numeric(df_home["Home Goal FT"], errors="coerce") > pd.to_numeric(df_home["Away Goal FT"], errors="coerce")).sum())
            draws = int((pd.to_numeric(df_home["Home Goal FT"], errors="coerce") == pd.to_numeric(df_home["Away Goal FT"], errors="coerce")).sum())
            losses = int((pd.to_numeric(df_home["Home Goal FT"], errors="coerce") < pd.to_numeric(df_home["Away Goal FT"], errors="coerce")).sum())
            row_home["BACK WIN% HOME"] = round((wins / matches_home) * 100, 2)
            row_home["BACK WIN% DRAW"] = round((draws / matches_home) * 100, 2)
            row_home["BACK WIN% AWAY"] = round((losses / matches_home) * 100, 2)
        else:
            row_home["BACK WIN% HOME"] = row_home["BACK WIN% DRAW"] = row_home["BACK WIN% AWAY"] = 0.0
        for outcome in ("HOME", "DRAW", "AWAY"):
            row_home[f"BACK PTS {outcome}"] = _format_value(profits_back[outcome])
            row_home[f"BACK ROI% {outcome}"] = _format_value(rois_back[outcome], is_roi=True)
            row_home[f"Lay pts {outcome}"] = _format_value(profits_lay[outcome])
            row_home[f"lay ROI% {outcome}"] = _format_value(rois_lay[outcome], is_roi=True)
        rows.append(row_home)

        # Ospite
        if label and label_t in ("Away", "Both"):
            df_away = df[(df["Label"] == label) & (df["Away"] == squadra_ospite)]
            if df_away.empty:
                df_away = df[df["Away"] == squadra_ospite]
                st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_ospite}.")
        else:
            df_away = df[df["Away"] == squadra_ospite]
        df_away_light = df_away[cols_1x2].copy()
        profits_back, rois_back, profits_lay, rois_lay, matches_away = _calc_back_lay_1x2_cached(df_away_light)
        row_away = {"LABEL": squadra_ospite, "MATCHES": matches_away}
        if matches_away > 0:
            wins = int((pd.to_numeric(df_away["Away Goal FT"], errors="coerce") > pd.to_numeric(df_away["Home Goal FT"], errors="coerce")).sum())
            draws = int((pd.to_numeric(df_away["Away Goal FT"], errors="coerce") == pd.to_numeric(df_away["Home Goal FT"], errors="coerce")).sum())
            losses = int((pd.to_numeric(df_away["Away Goal FT"], errors="coerce") < pd.to_numeric(df_away["Home Goal FT"], errors="coerce")).sum())
            row_away["BACK WIN% HOME"] = round((losses / matches_away) * 100, 2)
            row_away["BACK WIN% DRAW"] = round((draws / matches_away) * 100, 2)
            row_away["BACK WIN% AWAY"] = round((wins / matches_away) * 100, 2)
        else:
            row_away["BACK WIN% HOME"] = row_away["BACK WIN% DRAW"] = row_away["BACK WIN% AWAY"] = 0.0
        for outcome in ("HOME", "DRAW", "AWAY"):
            row_away[f"BACK PTS {outcome}"] = _format_value(profits_back[outcome])
            row_away[f"BACK ROI% {outcome}"] = _format_value(rois_back[outcome], is_roi=True)
            row_away[f"Lay pts {outcome}"] = _format_value(profits_lay[outcome])
            row_away[f"lay ROI% {outcome}"] = _format_value(rois_lay[outcome], is_roi=True)
        rows.append(row_away)

        df_long = pd.DataFrame([
            {
                "LABEL": row["LABEL"] if i == 0 or row["LABEL"] != rows[i-1]["LABEL"] else "",
                "SEGNO": outcome,
                "Matches": row.get("MATCHES", 0),
                "Win %": row.get(f"BACK WIN% {outcome}", 0),
                "Back Pts": row.get(f"BACK PTS {outcome}", _format_value(0)),
                "Back ROI %": row.get(f"BACK ROI% {outcome}", _format_value(0, is_roi=True)),
                "Lay Pts": row.get(f"Lay pts {outcome}", _format_value(0)),
                "Lay ROI %": row.get(f"lay ROI% {outcome}", _format_value(0, is_roi=True)),
            }
            for i, row in enumerate(rows)
            for outcome in ("HOME", "DRAW", "AWAY")
        ])

        st.dataframe(
            df_long, use_container_width=True, height=420,
            column_config={"Matches": st.column_config.NumberColumn(format="%.0f"),
                           "Win %": st.column_config.NumberColumn(format="%.2f")}
        )
        _download_df_button(df_long, "1x2_overview.csv", "⬇️ Scarica 1X2 CSV")

        st.divider()
        st.subheader("📌 Macro KPI Squadre")
        st.markdown("Media goal fatti/subiti, esito 1X2 e BTTS% per **Home@Casa** e **Away@Trasferta**.")

        @st.cache_data(show_spinner=False, ttl=900)
        def _macro_stats_cached(df_light: pd.DataFrame, team: str, side: str):
            if compute_team_macro_stats is None:
                d = df_light.copy()
                if side == "Home":
                    d = d[d["Home"] == team]
                    stats = {
                        "Matches Played": len(d),
                        "Win %": (d["Home Goal FT"] > d["Away Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Draw %": (d["Home Goal FT"] == d["Away Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Loss %": (d["Home Goal FT"] < d["Away Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Avg Goals Scored": d["Home Goal FT"].mean() if len(d) else 0.0,
                        "Avg Goals Conceded": d["Away Goal FT"].mean() if len(d) else 0.0,
                        "BTTS %": ((d["Home Goal FT"]>0)&(d["Away Goal FT"]>0)).mean()*100 if len(d) else 0.0,
                    }
                else:
                    d = d[d["Away"] == team]
                    stats = {
                        "Matches Played": len(d),
                        "Win %": (d["Away Goal FT"] > d["Home Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Draw %": (d["Away Goal FT"] == d["Home Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Loss %": (d["Away Goal FT"] < d["Home Goal FT"]).mean()*100 if len(d) else 0.0,
                        "Avg Goals Scored": d["Away Goal FT"].mean() if len(d) else 0.0,
                        "Avg Goals Conceded": d["Home Goal FT"].mean() if len(d) else 0.0,
                        "BTTS %": ((d["Home Goal FT"]>0)&(d["Away Goal FT"]>0)).mean()*100 if len(d) else 0.0,
                    }
                return stats
            return compute_team_macro_stats(df_light.copy(), team, side)  # type: ignore

        df_stats_light = df[["Home","Away","Home Goal FT","Away Goal FT"]].copy()
        stats_home = _macro_stats_cached(df_stats_light, squadra_casa, "Home")
        stats_away = _macro_stats_cached(df_stats_light, squadra_ospite, "Away")
        if not stats_home or not stats_away:
            st.info("⚠️ Una delle due squadre non ha match disponibili.")
        else:
            df_comp = pd.DataFrame({squadra_casa: stats_home, squadra_ospite: stats_away})
            st.dataframe(df_comp, use_container_width=True, height=320)
            _download_df_button(df_comp.reset_index(), "macro_kpi.csv", "⬇️ Scarica Macro KPI CSV")

        st.divider()
        # KPI PLUS — tabelle professionali
        render_macro_kpi_plus(df_league_all, squadra_casa, squadra_ospite)

    # === TAB 2: ROI mercati ===
    with tab_roi:
        st.markdown("**Cosa vedi**: ROI storico per mercati **Over/BTTS**. Scope *Campionato/Label* e *Squadra @Casa/Trasferta*.")
        commission = 0.045
        df_ev_scope = df[df["Label"] == label].copy() if label else df.copy()
        df_ev_scope = df_ev_scope.dropna(subset=["Home Goal FT", "Away Goal FT"])
        OVER15_COLS = ("cotao1", "Odd Over 1.5", "odd over 1,5", "Over 1.5")
        OVER25_COLS = ("cotao",  "Odd Over 2.5", "odd over 2,5", "Over 2.5")
        OVER35_COLS = ("cotao3", "Odd Over 3.5", "odd over 3,5", "Over 3.5")
        BTTS_YES_COLS = ("gg", "GG", "odd goal", "BTTS Yes", "Odd BTTS Yes")
        shared = _get_shared_quotes()
        q_ov15, q_ov25, q_ov35, q_btts = shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"]

        def _light(df_in: pd.DataFrame, extra_cols: tuple[str, ...]) -> pd.DataFrame:
            base = ["Home Goal FT","Away Goal FT"]; cols = list({*base, *extra_cols})
            return df_in[[c for c in cols if c in df_in.columns]].copy()

        df_roi_league = pd.DataFrame([
            _calc_market_roi_cached(_light(df_ev_scope, OVER15_COLS), "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15),
            _calc_market_roi_cached(_light(df_ev_scope, OVER25_COLS), "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25),
            _calc_market_roi_cached(_light(df_ev_scope, OVER35_COLS), "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35),
            _calc_market_roi_cached(_light(df_ev_scope, BTTS_YES_COLS), "BTTS",     BTTS_YES_COLS, None, commission, q_btts),
        ])
        df_roi_league.insert(0, "Scope", "Campionato/Label")

        df_home_ctx = df[(df["Home"] == squadra_casa)].dropna(subset=["Home Goal FT", "Away Goal FT"]).copy()
        df_roi_home = pd.DataFrame([
            _calc_market_roi_cached(_light(df_home_ctx, OVER15_COLS), "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15),
            _calc_market_roi_cached(_light(df_home_ctx, OVER25_COLS), "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25),
            _calc_market_roi_cached(_light(df_home_ctx, OVER35_COLS), "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35),
            _calc_market_roi_cached(_light(df_home_ctx, BTTS_YES_COLS), "BTTS",     BTTS_YES_COLS, None, commission, q_btts),
        ])
        df_roi_home.insert(0, "Scope", f"{squadra_casa} @Casa")

        df_away_ctx = df[(df["Away"] == squadra_ospite)].dropna(subset=["Home Goal FT", "Away Goal FT"]).copy()
        df_roi_away = pd.DataFrame([
            _calc_market_roi_cached(_light(df_away_ctx, OVER15_COLS), "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15),
            _calc_market_roi_cached(_light(df_away_ctx, OVER25_COLS), "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25),
            _calc_market_roi_cached(_light(df_away_ctx, OVER35_COLS), "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35),
            _calc_market_roi_cached(_light(df_away_ctx, BTTS_YES_COLS), "BTTS",     BTTS_YES_COLS, None, commission, q_btts),
        ])
        df_roi_away.insert(0, "Scope", f"{squadra_ospite} @Trasferta")

        df_roi_all = pd.concat([df_roi_league, df_roi_home, df_roi_away], ignore_index=True)
        df_roi_all["Mercato"] = df_roi_all["Mercato"].astype(str).str.replace(r"\s+", " ", regex=True)

        st.dataframe(
            df_roi_all, use_container_width=True, height=420,
            column_config={
                "Quota Media": st.column_config.NumberColumn(format="%.2f"),
                "Esiti %": st.column_config.TextColumn(),
                "ROI Back %": st.column_config.TextColumn(),
                "ROI Lay %": st.column_config.TextColumn(),
                "Match Analizzati": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        _render_glossary_roi()
        _download_df_button(df_roi_all, "roi_markets_all_scopes.csv", "⬇️ Scarica ROI mercati (tutti gli scope)")

    # === TAB 3: EV storico squadre ===
    with tab_ev:
        st.markdown("**Cosa vedi**: stime di Probabilità (%) per Home@Casa, Away@Trasferta, Blended, Head-to-Head. EV = quota × p − 1.")
        use_label = st.checkbox("Usa il filtro Label (se disponibile)", value=True if label else False, key=_k("use_label_ev_squadre"))
        last_n = st.slider("Limita agli ultimi N match (0 = tutti)", 0, 50, 0, key=_k("last_n_ev"))

        def _limit_last_n(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
            if n and n > 0 and "Data" in df_in.columns:
                s = pd.to_datetime(df_in["Data"], errors="coerce")
                tmp = df_in.copy(); tmp["_data_"] = s
                tmp = tmp.sort_values("_data_", ascending=False).drop(columns=["_data_"])
                return tmp.head(n)
            return df_in

        df_home_ctx = df[(df["Home"] == squadra_casa)].copy()
        df_away_ctx = df[(df["Away"] == squadra_ospite)].copy()
        if use_label and label:
            df_home_ctx = df_home_ctx[df_home_ctx["Label"] == label]
            df_away_ctx = df_away_ctx[df_away_ctx["Label"] == label]

        df_home_ctx = _limit_last_n(df_home_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)
        df_away_ctx = _limit_last_n(df_away_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)

        df_h2h = df[((df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)) | ((df["Home"] == squadra_ospite) & (df["Away"] == squadra_casa))].copy()
        if use_label and label:
            df_h2h = df_h2h[df_h2h["Label"] == label]
        df_h2h = _limit_last_n(df_h2h.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)

        shared = _get_shared_quotes()
        quota_ov15, quota_ov25, quota_ov35, quota_btts = shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"]

        base_cols = ["Home Goal FT","Away Goal FT"]
        home_ctx_light = df_home_ctx[base_cols].copy()
        away_ctx_light = df_away_ctx[base_cols].copy()
        h2h_light       = df_h2h[base_cols].copy()

        df_ev_squadre, best = _build_ev_table_cached(
            home_ctx_light, away_ctx_light, h2h_light, squadra_casa, squadra_ospite,
            quota_ov15, quota_ov25, quota_ov35, quota_btts
        )

        st.markdown("### 🏅 Best EV (storico)")
        if best and best["ev"] > 0:
            bg = "#052e16"
            st.markdown(
                f"""
                <div style="border:1px solid #16a34a;border-radius:10px;padding:14px;background:{bg};color:#e5fff0;">
                    <div style="font-size:14px;opacity:.9;">Miglior opportunità (storico)</div>
                    <div style="display:flex;gap:20px;align-items:baseline;flex-wrap:wrap;">
                        <div style="font-size:28px;font-weight:700;">EV {best['ev']:+.2f}</div>
                        <div style="font-size:16px;">Mercato: <b>{best['mercato']}</b></div>
                        <div style="font-size:16px;">Scope: <b>{best['scope']}</b></div>
                        <div style="font-size:16px;">Prob: <b>{best['prob']:.1f}%</b></div>
                        <div style="font-size:16px;">Quota: <b>{best['quota']:.2f}</b></div>
                        <div style="font-size:16px;">Campione: <b>{best['campione']}</b> ({best['qualita']})</div>
                    </div>
                    <div style="font-size:12px;opacity:.8;margin-top:6px;">Nota: EV calcolato su storico; valida sempre la dimensione campione.</div>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            st.info("Nessun EV positivo tra Blended e H2H con le quote inserite.")

        st.subheader("📋 EV storico per mercato e scope")
        styled = _style_ev(df_ev_squadre.copy())
        st.dataframe(
            styled, use_container_width=True, height=380,
            column_config={
                "Quota": st.column_config.NumberColumn(format="%.2f"),
                f"{squadra_casa} @Casa %": st.column_config.NumberColumn(format="%.2f"),
                f"{squadra_ospite} @Trasferta %": st.column_config.NumberColumn(format="%.2f"),
                "Blended %": st.column_config.NumberColumn(format="%.2f"),
                "Head-to-Head %": st.column_config.NumberColumn(format="%.2f"),
                "EV Blended": st.column_config.NumberColumn(format="%.2f"),
                "EV H2H": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        _render_glossary_ev()
        _download_df_button(df_ev_squadre, "ev_storico_squadre.csv", "⬇️ Scarica EV Storico CSV")

    # === TAB 4: Statistiche squadre ===
    with tab_stats:
        st.subheader("Statistiche squadre")
        # Filtro stagioni SOLO per questa sezione (default stagione corrente)
        if "Stagione" in df_league_all.columns:
            seasons_desc = _seasons_desc(df_league_all["Stagione"].dropna().unique().tolist())
        else:
            seasons_desc = []
        with st.expander("⚙️ Filtro stagioni (solo per questa sezione)", expanded=True):
            default_curr = _pick_current_season(seasons_desc) if seasons_desc else []
            selected_stats_seasons = st.multiselect(
                "Scegli le stagioni da includere (se vuoto = tutte)",
                options=seasons_desc,
                default=default_curr,
                key=_k("stats_seasons_filter"),
            )
        df_stats_scope = df_league_all.copy()
        if selected_stats_seasons:
            df_stats_scope = df_stats_scope[df_stats_scope["Stagione"].astype(str).isin([str(s) for s in selected_stats_seasons])]
        render_team_stats_tab(df_stats_scope, league, squadra_casa, squadra_ospite)

    # === TAB 5: Correct Score ===
    with tab_cs:
        st.subheader("Correct Score – Poisson + Dixon-Coles")
        st.caption("Stima λ con shrink verso media di lega + forma recente. Heatmap, top score e EV sui punteggi corretti.")
        if run_correct_score_panel is None:
            st.info("Modulo Correct Score non disponibile in questo ambiente.")
        else:
            run_correct_score_panel(
                df=df_league_all, league_code=league, home_team=squadra_casa, away_team=squadra_ospite,
                seasons=seasons_selected or None, default_rho=-0.05, default_kappa=3.0,
                default_recent_weight=0.25, default_recent_n=6, default_max_goals=6,
            )

    # === TAB 6: Live da Minuto ===
    with tab_live:
        st.markdown("**Analisi Live** — precompilata con i dati selezionati sopra.")
        st.session_state["campionato_corrente"] = league
        st.session_state["home_live"] = squadra_casa
        st.session_state["away_live"] = squadra_ospite
        st.session_state["odd_h"] = float(odd_home)
        st.session_state["odd_d"] = float(odd_draw)
        st.session_state["odd_a"] = float(odd_away)

        cL1, cL2 = st.columns([1,1])
        with cL1:
            live_minute = st.slider("⏲️ Minuto (prefill)", 1, 120, 45, key=_k("live:min"))
            st.session_state["minlive"] = int(live_minute)
        with cL2:
            live_score = st.text_input("📟 Risultato live (prefill)", value="0-0", key=_k("live:score"))
            st.session_state["scorelive"] = str(live_score).strip()

        st.caption("Suggerimento: puoi modificare anche i controlli interni del modulo Live.")
        if _run_live is None:
            st.info("Modulo Live non disponibile in questo ambiente.")
        else:
            _orig_set_cfg = getattr(st, "set_page_config", None)
            try:
                def _noop(*args, **kwargs): return None
                st.set_page_config = _noop
                _run_live(df_league_all)
            except Exception as e:
                st.warning(f"Modulo Live caricato con avviso: {e}")
            finally:
                if _orig_set_cfg:
                    st.set_page_config = _orig_set_cfg

    shared = _get_shared_quotes()
    _set_qparams(
        league=league, home=squadra_casa, away=squadra_ospite,
        q=st.session_state.get(_k("search_team"), "") or "",
        qh=odd_home, qd=odd_draw, qa=odd_away,
        ov15=shared["ov15"], ov25=shared["ov25"], ov35=shared["ov35"], btts=shared["btts"],
    )
