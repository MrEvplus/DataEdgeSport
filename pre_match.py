# pre_match.py
from __future__ import annotations

import re
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from squadre import compute_team_macro_stats
from utils import label_match

# ==========================
# Altair Theme (globale)
# ==========================
def _alt_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 12, "titleFontSize": 12},
            "legend": {"labelFontSize": 12, "titleFontSize": 12},
            "title": {"fontSize": 14},
        }
    }

try:
    alt.themes.register("app_theme", _alt_theme)
    alt.themes.enable("app_theme")
except Exception:
    pass

# ==========================
# Helper generali
# ==========================
def _k(name: str) -> str:
    return f"prematch:{name}"

def _ensure_str(s: pd.Series) -> pd.Series:
    """String robusta anche per dtype Categorical."""
    if s is None:
        return pd.Series(dtype="string")
    try:
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.astype("string")
    except Exception:
        pass
    return s.astype("string")

def _coerce_float(s: pd.Series) -> pd.Series:
    """Converte Serie a float accettando virgole e stringhe."""
    if s is None:
        return pd.Series(dtype="float")
    return pd.to_numeric(_ensure_str(s).str.replace(",", ".", regex=False), errors="coerce")

def _first_present(cols: list[str], columns: pd.Index) -> str | None:
    """Ritorna il primo nome colonna presente tra quelli proposti."""
    for c in cols:
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

# Query params (compat)
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

# ==========================
# Cache helpers
# ==========================
@st.cache_data(show_spinner=False)
def _subset_league(df: pd.DataFrame, league: str) -> pd.DataFrame:
    d = df.copy()
    d["country"] = _ensure_str(d["country"]).str.strip().str.upper()
    return d[d["country"] == (league or "").strip().str.upper()]

@st.cache_data(show_spinner=False)
def _with_label(df: pd.DataFrame) -> pd.DataFrame:
    if "Label" in df.columns:
        return df
    tmp = df.copy()
    tmp["Label"] = tmp.apply(label_match, axis=1)
    return tmp

@st.cache_data(show_spinner=False)
def _h2h(df: pd.DataFrame, h: str, a: str, label: str | None) -> pd.DataFrame:
    x = df[
        ((df["Home"] == h) & (df["Away"] == a)) |
        ((df["Home"] == a) & (df["Away"] == h))
    ]
    if label:
        x = x[x["Label"] == label]
    return x

# ==========================
# Filtro stagioni (ordinamento dalla più recente)
# ==========================
def _season_sort_key(s: str) -> int:
    # Estrae l'anno più grande nella stringa (es. "2024/2025" -> 2025)
    if not isinstance(s, str):
        s = str(s)
    yrs = [int(x) for x in re.findall(r"\d{4}", s)]
    return max(yrs) if yrs else -1

def _seasons_desc(unique_seasons: list) -> list[str]:
    arr = [str(x) for x in unique_seasons if pd.notna(x)]
    return sorted(arr, key=_season_sort_key, reverse=True)

# ==========================
# League data by Label
# ==========================
def _league_data_by_label(df: pd.DataFrame, label: str) -> dict | None:
    if "Label" not in df.columns:
        df = df.copy()
        df["Label"] = df.apply(label_match, axis=1)

    if "match_result" not in df.columns:
        df["match_result"] = df.apply(
            lambda r: "Home Win" if r["Home Goal FT"] > r["Away Goal FT"]
            else "Away Win" if r["Home Goal FT"] < r["Away Goal FT"]
            else "Draw",
            axis=1
        )

    group = df.groupby("Label").agg(
        Matches=("Home", "count"),
        HomeWin_pct=("match_result", lambda x: (x == "Home Win").mean() * 100),
        Draw_pct=("match_result", lambda x: (x == "Draw").mean() * 100),
        AwayWin_pct=("match_result", lambda x: (x == "Away Win").mean() * 100),
    ).reset_index()

    row = group[group["Label"] == label]
    return row.iloc[0].to_dict() if not row.empty else None

# ==========================
# Back/Lay 1X2 su dataset
# ==========================
def _calc_back_lay_1x2(df: pd.DataFrame, commission: float = 0.0):
    """Profitti e ROI% per BACK e LAY (liability=1) su HOME/DRAW/AWAY."""
    if df.empty:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, zero, zero, 0

    for c in ("Odd home", "Odd Draw", "Odd Away"):
        if c in df.columns:
            df[c] = _coerce_float(df[c])

    profits_back = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    profits_lay = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    matches = len(df)

    for _, row in df.iterrows():
        hg, ag = row["Home Goal FT"], row["Away Goal FT"]
        result = "HOME" if hg > ag else "AWAY" if hg < ag else "DRAW"

        prices = {
            "HOME": float(row.get("Odd home", np.nan)),
            "DRAW": float(row.get("Odd Draw", np.nan)),
            "AWAY": float(row.get("Odd Away", np.nan)),
        }
        for k, v in prices.items():
            if not (v and v > 1.0 and np.isfinite(v)):
                prices[k] = 2.0  # default sicuro

        for outcome in ("HOME", "DRAW", "AWAY"):
            p = prices[outcome]
            # BACK
            if result == outcome:
                profits_back[outcome] += (p - 1) * (1 - commission)
            else:
                profits_back[outcome] -= 1
            # LAY (liability=1)
            stake = 1.0 / (p - 1.0)
            if result != outcome:
                profits_lay[outcome] += stake
            else:
                profits_lay[outcome] -= 1.0

    rois_back = {k: round((v / matches) * 100, 2) for k, v in profits_back.items()}
    rois_lay = {k: round((v / matches) * 100, 2) for k, v in profits_lay.items()}
    return profits_back, rois_back, profits_lay, rois_lay, matches

# ==========================
# ROI Over/Under/BTTS
# ==========================
def _calc_market_roi(df: pd.DataFrame, market: str, price_cols: list[str],
                     line: float | None, commission: float, manual_price: float | None = None):
    """
    Calcola ROI per un singolo mercato:
    - market: "Over 1.5" | "Over 2.5" | "Over 3.5" | "BTTS"
    - price_cols: sinonimi accettati per la colonna quota
    - line: soglia goal (None per BTTS)
    - commission: es. 0.045
    - manual_price: quota di fallback se la colonna manca o è < 1.01
    """
    if df.empty:
        return {
            "Mercato": market, "Quota Media": np.nan, "Esiti %": "0.0%",
            "ROI Back %": "0.0%", "ROI Lay %": "0.0%", "Match Analizzati": 0
        }

    col = _first_present(price_cols, df.columns)
    odds = _coerce_float(df[col]) if col else pd.Series([np.nan] * len(df), index=df.index)
    if manual_price and (odds.isna() | (odds < 1.01)).all():
        odds = pd.Series([manual_price] * len(df), index=df.index)
    else:
        if manual_price is not None:
            odds = odds.where(odds >= 1.01, manual_price)

    hits = 0
    back_profit = 0.0
    lay_profit = 0.0
    total = 0
    qsum = 0.0
    qcount = 0

    for i, row in df.iterrows():
        o = float(odds.loc[i]) if i in odds.index else float("nan")
        if not (o and o >= 1.01 and np.isfinite(o)):
            continue

        total += 1
        qsum += o; qcount += 1

        goals = int(row["Home Goal FT"]) + int(row["Away Goal FT"])

        if market == "BTTS":
            goal_both = (row["Home Goal FT"] > 0) and (row["Away Goal FT"] > 0)
            if goal_both:
                hits += 1
                back_profit += (o - 1) * (1 - commission)
                lay_profit -= 1
            else:
                lay_profit += 1 / (o - 1)
                back_profit -= 1
        else:
            assert line is not None
            if goals > line:
                hits += 1
                back_profit += (o - 1) * (1 - commission)
                lay_profit -= 1
            else:
                lay_profit += 1 / (o - 1)
                back_profit -= 1

    avg_quote = round(qsum / qcount, 2) if qcount > 0 else np.nan
    pct = round((hits / total) * 100, 2) if total > 0 else 0.0
    roi_back = round((back_profit / total) * 100, 2) if total > 0 else 0.0
    roi_lay = round((lay_profit / total) * 100, 2) if total > 0 else 0.0

    return {
        "Mercato": market,
        "Quota Media": avg_quote,
        "Esiti %": f"{pct}%",
        "ROI Back %": f"{roi_back}%",
        "ROI Lay %": f"{roi_lay}%",
        "Match Analizzati": total
    }

# ==========================
# Probabilità storiche per EV
# ==========================
def _market_prob(df: pd.DataFrame, market: str, line: float | None) -> float:
    """Ritorna la probabilità (0-100) che il mercato si verifichi su df."""
    if df.empty:
        return 0.0
    goals = pd.to_numeric(df["Home Goal FT"], errors="coerce").fillna(0) + \
            pd.to_numeric(df["Away Goal FT"], errors="coerce").fillna(0)
    if market == "BTTS":
        ok = ((df["Home Goal FT"] > 0) & (df["Away Goal FT"] > 0)).mean()
    else:
        ok = (goals > float(line)).mean() if line is not None else 0.0
    return round(float(ok) * 100, 2)

def _quality_label(n: int) -> str:
    if n >= 50:
        return "ALTO"
    if n >= 20:
        return "MEDIO"
    return "BASSO"

# ==========================
# Sezione EV storico (costruzione tabella + Best EV)
# ==========================
def _build_ev_table(df_home_ctx: pd.DataFrame, df_away_ctx: pd.DataFrame, df_h2h: pd.DataFrame,
                    squadra_casa: str, squadra_ospite: str,
                    quota_ov15: float, quota_ov25: float, quota_ov35: float, quota_btts: float):
    markets = [
        ("Over 1.5", 1.5, quota_ov15),
        ("Over 2.5", 2.5, quota_ov25),
        ("Over 3.5", 3.5, quota_ov35),
        ("BTTS", None, quota_btts),
    ]
    rows = []
    candidates_for_best = []  # solo Blended e H2H

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
            "Mercato": name,
            "Quota": q,
            f"{squadra_casa} @Casa %": p_home,
            f"EV {squadra_casa}": ev_home,
            f"{squadra_ospite} @Trasferta %": p_away,
            f"EV {squadra_ospite}": ev_away,
            "Blended %": p_blnd,
            "EV Blended": ev_blnd,
            "Qualità Blended": qual_blnd,
            "Head-to-Head %": p_h2h,
            "EV H2H": ev_h2h,
            "Qualità H2H": qual_h2h,
            "Match H": n_h,
            "Match A": n_a,
            "Match H2H": n_h2h,
        })

        candidates_for_best.append({
            "scope": "Blended",
            "mercato": name,
            "quota": q,
            "prob": p_blnd,
            "ev": ev_blnd,
            "campione": n_h + n_a,
            "qualita": qual_blnd
        })
        candidates_for_best.append({
            "scope": "Head-to-Head",
            "mercato": name,
            "quota": q,
            "prob": p_h2h,
            "ev": ev_h2h,
            "campione": n_h2h,
            "qualita": qual_h2h
        })

    df_ev = pd.DataFrame(rows)

    # Best EV: scegliamo il miglior EV positivo (priorità Blended > H2H a parità di EV)
    best = None
    for c in sorted(candidates_for_best, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c
            break
    return df_ev, best

# ==========================
# ENTRY POINT (PRE-MATCH SOLTANTO)
# ==========================
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title("📊 Pre-Match – Analisi per Trader Professionisti")

    qin = _get_qparams()

    # Normalizzazione e filtro league
    if "country" not in df.columns:
        st.error("Dataset senza colonna 'country'.")
        st.stop()

    df = df.copy()
    df["country"] = _ensure_str(df["country"]).str.strip().str.upper()
    league = (db_selected or "").strip().upper()
    df = df[df["country"] == league]
    if df.empty:
        st.warning(f"Nessun dato per il campionato '{league}'.")
        st.stop()

    df = _with_label(df)
    df["Home"] = _ensure_str(df["Home"]).str.strip()
    df["Away"] = _ensure_str(df["Away"]).str.strip()

    # ======== Filtro Stagioni (pro) ========
    seasons_selected = None
    if "Stagione" in df.columns:
        seasons_desc = _seasons_desc(df["Stagione"].dropna().unique().tolist())
        latest = seasons_desc[0] if seasons_desc else None

        with st.expander("⚙️ Filtro Stagioni", expanded=True):
            colA, colB = st.columns([2, 1])
            with colA:
                seasons_selected = st.multiselect(
                    "Seleziona stagioni (Manuale)",
                    options=seasons_desc,
                    default=[latest] if latest else [],
                    key=_k("stagioni_manual"),
                    help="Puoi scegliere una o più stagioni. L'elenco è ordinato dalla più recente."
                )
            with colB:
                preset = st.selectbox(
                    "Intervallo rapido",
                    options=["—", "Ultime 1", "Ultime 2", "Ultime 3", "Ultime 5"],
                    index=0,
                    key=_k("stagioni_preset"),
                    help="Applica un intervallo dalla stagione più recente verso il passato."
                )
                if preset != "—" and seasons_desc:
                    n = int(preset.split()[-1])
                    seasons_selected = seasons_desc[:n]

            if seasons_selected:
                st.caption(f"Stagioni attive: **{', '.join(seasons_selected)}**")
                df = df[df["Stagione"].astype(str).isin(seasons_selected)]
            else:
                st.caption("Nessun filtro stagioni attivo (tutte le stagioni disponibili).")

    # ======== Selettori Squadre ========
    all_teams = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))
    team_filter = st.text_input("🔎 Cerca squadra", value=qin.get("q", ""), key=_k("search_team"))
    if team_filter:
        flt = str(team_filter).strip().lower()
        teams = [t for t in all_teams if flt in t.lower()]
        if not teams:
            st.info("Nessuna squadra trovata col filtro. Mostro tutte.")
            teams = all_teams
    else:
        teams = all_teams

    qp_home = qin.get("home") if isinstance(qin.get("home"), str) else (qin.get("home", [""])[0] if qin.get("home") else "")
    qp_away = qin.get("away") if isinstance(qin.get("away"), str) else (qin.get("away", [""])[0] if qin.get("away") else "")

    default_home = teams.index(qp_home) if qp_home in teams else 0
    default_away = teams.index(qp_away) if qp_away in teams else (1 if len(teams) > 1 else 0)

    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        squadra_casa = st.selectbox("Seleziona Squadra Casa", options=teams, index=default_home, key=_k("squadra_casa"))
    with col_sel2:
        squadra_ospite = st.selectbox("Seleziona Squadra Ospite", options=teams, index=default_away, key=_k("squadra_ospite"))

    # ======== Quote 1X2 per Label detection ========
    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Quota Vincente Casa", min_value=1.01, step=0.01, value=float(qin.get("qh", 2.00)), key=_k("quota_home"))
        st.caption(f"Prob. Casa ({squadra_casa}): **{round(100/odd_home, 2)}%**")
    with c2:
        odd_draw = st.number_input("Quota Pareggio", min_value=1.01, step=0.01, value=float(qin.get("qd", 3.20)), key=_k("quota_draw"))
        st.caption(f"Prob. Pareggio: **{round(100/odd_draw, 2)}%**")
    with c3:
        odd_away = st.number_input("Quota Vincente Ospite", min_value=1.01, step=0.01, value=float(qin.get("qa", 3.80)), key=_k("quota_away"))
        st.caption(f"Prob. Ospite ({squadra_ospite}): **{round(100/odd_away, 2)}%**")

    if not (squadra_casa and squadra_ospite and squadra_casa != squadra_ospite):
        st.info("Seleziona due squadre diverse per procedere.")
        return

    # Label dal range quote inserite
    label = _label_from_odds(float(odd_home), float(odd_away))
    label_type = _label_type(label)

    # Header KPI compatti
    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown(f"**Campionato:** `{league}`")
    header_cols[1].markdown(f"**Label:** `{label}`")
    header_cols[2].markdown(f"**Home:** `{squadra_casa}`")
    header_cols[3].markdown(f"**Away:** `{squadra_ospite}`")

    if label == "Others" or label not in set(df["Label"]):
        st.info("⚠️ Nessuna partita trovata per questo label nel campionato: uso l'intero campionato.")
        label = None

    # Tabs principali (solo PRE-MATCH)
    tab_1x2, tab_roi, tab_ev = st.tabs(["1X2", "ROI mercati", "EV storico squadre"])

    # ==========================
    # TAB 1: 1X2 + macro KPI
    # ==========================
    with tab_1x2:
        df_league_scope = df[df["Label"] == label] if label else df
        profits_back, rois_back, profits_lay, rois_lay, matches_league = _calc_back_lay_1x2(df_league_scope)
        league_stats = _league_data_by_label(df, label) if label else _league_data_by_label(df, _label_from_odds(2.0, 2.0))

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
        if label and label_type in ("Home", "Both"):
            df_home = df[(df["Label"] == label) & (df["Home"] == squadra_casa)]
            if df_home.empty:
                df_home = df[df["Home"] == squadra_casa]
                st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_casa}.")
        else:
            df_home = df[df["Home"] == squadra_casa]

        profits_back, rois_back, profits_lay, rois_lay, matches_home = _calc_back_lay_1x2(df_home)
        row_home = {"LABEL": squadra_casa, "MATCHES": matches_home}
        if matches_home > 0:
            wins = int((df_home["Home Goal FT"] > df_home["Away Goal FT"]).sum())
            draws = int((df_home["Home Goal FT"] == df_home["Away Goal FT"]).sum())
            losses = int((df_home["Home Goal FT"] < df_home["Away Goal FT"]).sum())
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
        if label and label_type in ("Away", "Both"):
            df_away = df[(df["Label"] == label) & (df["Away"] == squadra_ospite)]
            if df_away.empty:
                df_away = df[df["Away"] == squadra_ospite]
                st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_ospite}.")
        else:
            df_away = df[df["Away"] == squadra_ospite]

        profits_back, rois_back, profits_lay, rois_lay, matches_away = _calc_back_lay_1x2(df_away)
        row_away = {"LABEL": squadra_ospite, "MATCHES": matches_away}
        if matches_away > 0:
            wins = int((df_away["Away Goal FT"] > df_away["Home Goal FT"]).sum())
            draws = int((df_away["Away Goal FT"] == df_away["Home Goal FT"]).sum())
            losses = int((df_away["Away Goal FT"] < df_away["Home Goal FT"]).sum())
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
            df_long,
            use_container_width=True,
            height=420,
            column_config={
                "Matches": st.column_config.NumberColumn(format="%.0f"),
                "Win %": st.column_config.NumberColumn(format="%.2f"),
                "Back ROI %": st.column_config.TextColumn(help="ROI medio per scommessa (Back)"),
                "Lay ROI %": st.column_config.TextColumn(help="ROI medio per scommessa (Lay)"),
            },
        )
        _download_df_button(df_long, "1x2_overview.csv", "⬇️ Scarica 1X2 CSV")

        st.divider()
        st.subheader("📌 Macro KPI Squadre")
        stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
        stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")
        if not stats_home or not stats_away:
            st.info("⚠️ Una delle due squadre non ha match disponibili per il confronto.")
        else:
            df_comp = pd.DataFrame({squadra_casa: stats_home, squadra_ospite: stats_away})
            st.dataframe(df_comp, use_container_width=True, height=320)
            _download_df_button(df_comp.reset_index(), "macro_kpi.csv", "⬇️ Scarica Macro KPI CSV")

    # ==========================
    # TAB 2: ROI mercati (campionato/label)
    # ==========================
    with tab_roi:
        st.caption("Calcolo ROI Back & Lay su Over 1.5, 2.5, 3.5 e BTTS (campionato/label)")
        commission = 0.045

        df_ev_scope = df[df["Label"] == label].copy() if label else df.copy()
        df_ev_scope = df_ev_scope.dropna(subset=["Home Goal FT", "Away Goal FT"])

        OVER15_COLS = ["cotao1", "Odd Over 1.5", "odd over 1,5", "Over 1.5"]
        OVER25_COLS = ["cotao", "Odd Over 2.5", "odd over 2,5", "Over 2.5"]
        OVER35_COLS = ["cotao3", "Odd Over 3.5", "odd over 3,5", "Over 3.5"]
        BTTS_YES_COLS = ["gg", "GG", "odd goal", "BTTS Yes", "Odd BTTS Yes"]

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            q_ov15 = st.number_input("📥 Quota Over 1.5 (fallback)", min_value=1.01, step=0.01, value=2.00, key=_k("q_ov15"))
        with c2:
            q_ov25 = st.number_input("📥 Quota Over 2.5 (fallback)", min_value=1.01, step=0.01, value=2.00, key=_k("q_ov25"))
        with c3:
            q_ov35 = st.number_input("📥 Quota Over 3.5 (fallback)", min_value=1.01, step=0.01, value=2.00, key=_k("q_ov35"))
        with c4:
            q_btts = st.number_input("📥 Quota BTTS (fallback)", min_value=1.01, step=0.01, value=2.00, key=_k("q_btts"))

        table_data = [
            _calc_market_roi(df_ev_scope, "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15),
            _calc_market_roi(df_ev_scope, "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25),
            _calc_market_roi(df_ev_scope, "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35),
            _calc_market_roi(df_ev_scope, "BTTS", BTTS_YES_COLS, None, commission, q_btts),
        ]
        df_ev = pd.DataFrame(table_data)
        st.dataframe(
            df_ev,
            use_container_width=True,
            height=360,
            column_config={
                "Quota Media": st.column_config.NumberColumn(format="%.2f"),
                "Esiti %": st.column_config.TextColumn(),
                "ROI Back %": st.column_config.TextColumn(),
                "ROI Lay %": st.column_config.TextColumn(),
                "Match Analizzati": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        _download_df_button(df_ev, "roi_markets.csv", "⬇️ Scarica ROI mercati CSV")

    # ==========================
    # TAB 3: EV Storico – Squadre selezionate + KPI Best EV
    # ==========================
    with tab_ev:
        st.caption("Probabilità storiche per Home@Casa, Away@Trasferta, Blended e H2H")
        use_label = st.checkbox("Usa il filtro Label (se disponibile)", value=bool(label), key=_k("use_label_ev_squadre"))
        last_n = st.slider("Limita agli ultimi N match (0 = tutti)", 0, 50, int(_get_qparams().get("n", 0) or 0), key=_k("last_n_ev"))

        df_home_ctx = df[(df["Home"] == squadra_casa)].copy()
        df_away_ctx = df[(df["Away"] == squadra_ospite)].copy()
        if use_label and label:
            df_home_ctx = df_home_ctx[df_home_ctx["Label"] == label]
            df_away_ctx = df_away_ctx[df_away_ctx["Label"] == label]

        def _limit_last_n(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
            if n and n > 0 and "Data" in df_in.columns:
                s = pd.to_datetime(df_in["Data"], errors="coerce")
                tmp = df_in.copy()
                tmp["_data_"] = s
                tmp = tmp.sort_values("_data_", ascending=False).drop(columns=["_data_"])
                return tmp.head(n)
            return df_in

        df_home_ctx = _limit_last_n(df_home_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)
        df_away_ctx = _limit_last_n(df_away_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)
        df_h2h = _h2h(df, squadra_casa, squadra_ospite, label if use_label else None)
        df_h2h = _limit_last_n(df_h2h.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)

        # Quote per EV
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            quota_ov15 = st.number_input("Quota Over 1.5", min_value=1.01, step=0.01, value=float(_get_qparams().get("ov15", 2.00) or 2.00), key=_k("ev_ov15"))
        with c2:
            quota_ov25 = st.number_input("Quota Over 2.5", min_value=1.01, step=0.01, value=float(_get_qparams().get("ov25", 2.00) or 2.00), key=_k("ev_ov25"))
        with c3:
            quota_ov35 = st.number_input("Quota Over 3.5", min_value=1.01, step=0.01, value=float(_get_qparams().get("ov35", 2.00) or 2.00), key=_k("ev_ov35"))
        with c4:
            quota_btts = st.number_input("Quota BTTS", min_value=1.01, step=0.01, value=float(_get_qparams().get("btts", 2.00) or 2.00), key=_k("ev_btts"))

        # Tabella EV + Best EV
        df_ev_squadre, best = _build_ev_table(
            df_home_ctx, df_away_ctx, df_h2h,
            squadra_casa, squadra_ospite,
            quota_ov15, quota_ov25, quota_ov35, quota_btts
        )

        # KPI Card Best EV
        st.markdown("### 🏅 Best EV (storico)")
        if best and best["ev"] > 0:
            bg = "#052e16"  # verde molto scuro
            st.markdown(
                f"""
                <div style="border:1px solid #16a34a;border-radius:10px;padding:14px;background:{bg};color:#e5fff0;">
                    <div style="font-size:14px;opacity:.9;">Miglior opportunità (storico)</div>
                    <div style="display:flex;gap:20px;align-items:baseline;">
                        <div style="font-size:28px;font-weight:700;">EV {best['ev']:+.2f}</div>
                        <div style="font-size:16px;">Mercato: <b>{best['mercato']}</b></div>
                        <div style="font-size:16px;">Scope: <b>{best['scope']}</b></div>
                        <div style="font-size:16px;">Prob: <b>{best['prob']:.1f}%</b></div>
                        <div style="font-size:16px;">Quota: <b>{best['quota']:.2f}</b></div>
                        <div style="font-size:16px;">Campione: <b>{best['campione']}</b> ({best['qualita']})</div>
                    </div>
                    <div style="font-size:12px;opacity:.8;margin-top:6px;">Nota: EV calcolato su storico; valida sempre la dimensione campione.</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Nessun EV positivo tra Blended e H2H con le quote inserite.")

        # Tabella
        st.subheader("📋 EV storico per mercato e scope")
        st.dataframe(
            df_ev_squadre,
            use_container_width=True,
            height=380,
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
        _download_df_button(df_ev_squadre, "ev_storico_squadre.csv", "⬇️ Scarica EV Storico CSV")

        # Grafico comparativo probabilità
        st.subheader("📊 Probabilità storiche – confronto scope")
        prob_rows = []
        for _, r in df_ev_squadre.iterrows():
            prob_rows += [
                {"Mercato": r["Mercato"], "Scope": f"{squadra_casa} @Casa", "Prob %": r[f"{squadra_casa} @Casa %"]},
                {"Mercato": r["Mercato"], "Scope": f"{squadra_ospite} @Trasferta", "Prob %": r[f"{squadra_ospite} @Trasferta %"]},
                {"Mercato": r["Mercato"], "Scope": "Blended", "Prob %": r["Blended %"]},
                {"Mercato": r["Mercato"], "Scope": "Head-to-Head", "Prob %": r["Head-to-Head %"]},
            ]
        df_prob = pd.DataFrame(prob_rows)
        if not df_prob.empty:
            chart = alt.Chart(df_prob).mark_bar().encode(
                x=alt.X("Mercato:N"),
                y=alt.Y("Prob %:Q"),
                color=alt.Color("Scope:N"),
                column=alt.Column("Scope:N", header=alt.Header(title="")),
                tooltip=["Mercato", "Scope", alt.Tooltip("Prob %:Q", format=".1f")],
            ).properties(height=300)
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Nessun dato per costruire il grafico delle probabilità.")

    # Aggiorna query params (deep-link)
    _set_qparams(
        league=league,
        home=squadra_casa,
        away=squadra_ospite,
        q=team_filter or "",
        qh=odd_home, qd=odd_draw, qa=odd_away,
        ov15=locals().get("quota_ov15", 2.00),
        ov25=locals().get("quota_ov25", 2.00),
        ov35=locals().get("quota_ov35", 2.00),
        btts=locals().get("quota_btts", 2.00),
        n=int(locals().get("last_n", 0)),
    )
