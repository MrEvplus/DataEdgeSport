from __future__ import annotations

import re
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from squadre import compute_team_macro_stats, render_team_stats_tab
from utils import label_match
from correct_score import run_correct_score_panel

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

def _first_present(cols: list[str], columns: pd.Index) -> str | None:
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

# Query params
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
# Stagioni (ordina dalla più recente)
# ==========================
def _season_sort_key(s: str) -> int:
    if not isinstance(s, str):
        s = str(s)
    yrs = [int(x) for x in re.findall(r"\d{4}", s)]
    return max(yrs) if yrs else -1

def _seasons_desc(unique_seasons: list) -> list[str]:
    arr = [str(x) for x in unique_seasons if pd.notna(x)]
    return sorted(arr, key=_season_sort_key, reverse=True)

def _pick_current_season(seasons_desc: list[str]) -> list[str]:
    """
    Sceglie la stagione 'in corso' in base alla data odierna.
    Logica EU: stagione = 1 luglio (sy) -> 30 giugno (ey).
    Riconosce formati: 'YYYY/YYYY+1', 'YYYY-YYYY+1', 'YYYY/YY', 'YYYY-YY', 'YYYY'.
    """
    if not seasons_desc:
        return []

    today = date.today()
    if today.month >= 7:   # lug-dic
        sy, ey = today.year, today.year + 1
    else:                  # gen-giu
        sy, ey = today.year - 1, today.year

    ey2 = str(ey)[-2:]
    candidates = [
        f"{sy}/{ey}", f"{sy}-{ey}",
        f"{sy}/{ey2}", f"{sy}-{ey2}",
        f"{sy}–{ey}",  f"{sy}–{ey2}",  # en dash
        str(sy), str(ey)                   # anno solare
    ]

    # match esatto
    for cand in candidates:
        for s in seasons_desc:
            if s.strip() == cand:
                return [s]

    # fallback 1: contiene sy o ey
    for s in seasons_desc:
        txt = s.strip()
        if str(sy) in txt or str(ey) in txt:
            return [s]

    # fallback 2: la prima più recente
    return seasons_desc[:1]


# ==========================
# Quote condivise (sincronizzate tra tab)
# ==========================
_SHARED_PREFIX = "prematch:shared:"

def _shared_key(name: str) -> str:
    return f"{_SHARED_PREFIX}{name}"

def _init_shared_quotes():
    defaults = {"ov15": 2.00, "ov25": 2.00, "ov35": 2.00, "btts": 2.00}
    for k, v in defaults.items():
        st.session_state.setdefault(_shared_key(k), v)

def _shared_number_input(label: str, shared_name: str, local_key: str,
                         min_value: float = 1.01, step: float = 0.01):
    """Widget locale che scrive sulle quote condivise."""
    _init_shared_quotes()
    # mantieni il locale allineato allo shared
    if local_key not in st.session_state:
        st.session_state[local_key] = float(st.session_state[_shared_key(shared_name)])

    def _on_change():
        st.session_state[_shared_key(shared_name)] = float(st.session_state[local_key])

    return st.number_input(
        label,
        min_value=min_value,
        step=step,
        key=local_key,
        on_change=_on_change,
    )

def _get_shared_quotes() -> dict:
    _init_shared_quotes()
    return {
        "ov15": float(st.session_state[_shared_key("ov15")]),
        "ov25": float(st.session_state[_shared_key("ov25")]),
        "ov35": float(st.session_state[_shared_key("ov35")]),
        "btts": float(st.session_state[_shared_key("btts")]),
    }


# ==========================
# League data by Label (robusta)
# ==========================
def _league_data_by_label(df: pd.DataFrame, label: str) -> dict | None:
    df = df.copy()
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

    row = group[group["Label"] == label]
    return row.iloc[0].to_dict() if not row.empty else None


# ==========================
# Back/Lay 1x2 (robusto)
# ==========================
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


# ==========================
# ROI Over/Under/BTTS
# ==========================
def _calc_market_roi(df: pd.DataFrame, market: str, price_cols: list[str],
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
    if n >= 50:
        return "ALTO"
    if n >= 20:
        return "MEDIO"
    return "BASSO"


# ==========================
# EV storico – tabella + Best EV
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
    candidates_for_best = []

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

        candidates_for_best.extend([
            {"scope": "Blended", "mercato": name, "quota": q, "prob": p_blnd, "ev": ev_blnd, "campione": n_h + n_a, "qualita": qual_blnd},
            {"scope": "Head-to-Head", "mercato": name, "quota": q, "prob": p_h2h, "ev": ev_h2h, "campione": n_h2h, "qualita": qual_h2h},
        ])

    df_ev = pd.DataFrame(rows)

    best = None
    for c in sorted(candidates_for_best, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c
            break
    return df_ev, best


# ==========================
# ENTRY POINT (PRE-MATCH)
# ==========================
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title("📊 Pre-Match – Analisi per Trader Professionisti")

    qin = _get_qparams()
    _init_shared_quotes()  # prepara le quote condivise

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

    # Snapshot del campionato PRIMA del filtro stagioni (serve alla tab Statistiche squadre)
    df_league_all = df.copy()

    df["Home"] = _ensure_str(df["Home"]).str.strip()
    df["Away"] = _ensure_str(df["Away"]).str.strip()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # ======== Filtro Stagioni (globale) ========
    if "Stagione" in df.columns:
        seasons_desc = _seasons_desc(df["Stagione"].dropna().unique().tolist())
        latest = seasons_desc[0] if seasons_desc else None

        with st.expander("⚙️ Filtro Stagioni", expanded=True):
            st.markdown(
                "Seleziona la profondità storica da analizzare. "
                "**Intervallo rapido** parte dalla **stagione più recente** verso il passato."
            )
            colA, colB = st.columns([2, 1])
            with colA:
                seasons_selected = st.multiselect(
                    "Seleziona stagioni (Manuale)",
                    options=seasons_desc,
                    default=[latest] if latest else [],
                    key=_k("stagioni_manual"),
                )
            with colB:
                preset = st.selectbox(
                    "Intervallo rapido",
                    options=["Tutte", "Stagione in corso", "Ultime 10", "Ultime 5", "Ultime 3", "Ultime 2", "Ultime 1"],
                    index=0,
                    key=_k("stagioni_preset"),
                )
                if preset == "Stagione in corso":
                    seasons_selected = _pick_current_season(seasons_desc)
                elif preset != "Tutte" and seasons_desc:
                    try:
                        n = int(preset.split()[-1])
                    except Exception:
                        n = 1
                    seasons_selected = seasons_desc[:n]
                else:  # "Tutte"
                    seasons_selected = []

            if seasons_selected:
                st.caption(f"Stagioni attive: **{', '.join(seasons_selected)}**")
                df = df[df["Stagione"].astype(str).isin(seasons_selected)]
            else:
                st.caption("Filtro stagioni: **Tutte**")

    # ======== Ricerca rapida + assegnazione ruolo ========
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
                    st.info("Nessuna squadra trovata. Mostro tutte.")
                    filtered = all_teams
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
                            st.warning("⚠️ La stessa squadra è già selezionata come Ospite. Cambia Ospite o usa 'Inverti'.")
                        else:
                            st.session_state[_k("squadra_casa")] = selected_from_search
                    else:
                        other = st.session_state.get(_k("squadra_casa"))
                        if other == selected_from_search:
                            st.warning("⚠️ La stessa squadra è già selezionata come Casa. Cambia Casa o usa 'Inverti'.")
                        else:
                            st.session_state[_k("squadra_ospite")] = selected_from_search
                    _set_qparams(q=team_filter or "")
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception:
                    st.warning("Impossibile assegnare la squadra. Riprova.")

    # ======== Selettori Squadre finali + Swap ========
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
                st.warning("⚠️ Casa e Ospite sono uguali: invertire non ha effetto.")
            else:
                _set_qparams(home=current_away, away=current_home)
                st.session_state[_k("swap_trigger")] = True
                try:
                    st.rerun()
                except Exception:
                    st.experimental_rerun()
    with col_sel2:
        squadra_ospite = st.selectbox("Seleziona Squadra Ospite", options=all_teams, index=default_away, key=_k("squadra_ospite"))

    if squadra_casa == squadra_ospite:
        st.error("❌ Casa e Ospite non possono essere la stessa squadra. Modifica la selezione o usa 'Inverti'.")
        return

    # ======== Quote 1X2 e Quote Over/BTTS condivise ========
    st.markdown(
        "Le quote 1X2 servono per **classificare** il match in un *label* (es. H_Fav, A_Fav, Balanced). "
        "Le **quote Over/BTTS** inserite qui sotto sono **condivise** e usate in tutti i tab (ROI, EV, Statistiche)."
    )
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
    with q1:
        _shared_number_input("Quota Over 1.5", "ov15", _k("shared:q_ov15"))
    with q2:
        _shared_number_input("Quota Over 2.5", "ov25", _k("shared:q_ov25"))
    with q3:
        _shared_number_input("Quota Over 3.5", "ov35", _k("shared:q_ov35"))
    with q4:
        _shared_number_input("Quota BTTS",     "btts", _k("shared:q_btts"))

    label = _label_from_odds(float(odd_home), float(odd_away))
    label_t = _label_type(label)

    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown(f"**Campionato:** `{league}`")
    header_cols[1].markdown(f"**Label:** `{label}`")
    header_cols[2].markdown(f"**Home:** `{squadra_casa}`")
    header_cols[3].markdown(f"**Away:** `{squadra_ospite}`")

    if label == "Others" or label not in set(df["Label"]):
        st.info("⚠️ Nessuna partita trovata per questo label nel campionato: uso l'intero campionato.")
        label = None

    # Tabs (PRE-MATCH)
    tab_1x2, tab_roi, tab_ev, tab_stats = st.tabs(
        ["1X2", "ROI mercati", "EV storico squadre", "Statistiche squadre"]
    )

    # ==========================
    # TAB 1: 1X2 + macro KPI
    # ==========================
    with tab_1x2:
        st.markdown(
            "**Cosa vedi qui**: confronto 1X2 tra campionato/label e le due squadre nel loro contesto "
            "(Casa per la squadra Home, Trasferta per la squadra Away). "
            "**Win%** sono le frequenze storiche; **Back/Lay ROI** sono profitti medi per scommessa."
        )

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
        if label and label_t in ("Home", "Both"):
            df_home = df[(df["Label"] == label) & (df["Home"] == squadra_casa)]
            if df_home.empty:
                df_home = df[df["Home"] == squadra_casa]
                st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_casa}.")
        else:
            df_home = df[df["Home"] == squadra_casa]

        profits_back, rois_back, profits_lay, rois_lay, matches_home = _calc_back_lay_1x2(df_home)
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

        profits_back, rois_back, profits_lay, rois_lay, matches_away = _calc_back_lay_1x2(df_away)
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
            df_long,
            use_container_width=True,
            height=420,
            column_config={
                "Matches": st.column_config.NumberColumn(format="%.0f"),
                "Win %": st.column_config.NumberColumn(format="%.2f"),
            },
        )
        _download_df_button(df_long, "1x2_overview.csv", "⬇️ Scarica 1X2 CSV")

        st.divider()
        st.subheader("📌 Macro KPI Squadre")
        st.markdown("Media goal fatti/subiti, esito 1X2 e BTTS% della squadra nel contesto **Home@Casa / Away@Trasferta**.")
        stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
        stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")
        if not stats_home or not stats_away:
            st.info("⚠️ Una delle due squadre non ha match disponibili per il confronto.")
        else:
            df_comp = pd.DataFrame({squadra_casa: stats_home, squadra_ospite: stats_away})
            st.dataframe(df_comp, use_container_width=True, height=320)
            _download_df_button(df_comp.reset_index(), "macro_kpi.csv", "⬇️ Scarica Macro KPI CSV")

    # ==========================
    # TAB 2: ROI mercati (campionato/label + squadre)
    # ==========================
    with tab_roi:
        st.markdown(
            "**Cosa vedi qui**: ROI storico per mercati **Over/BTTS**.\n\n"
            "- **Scope Campionato/Label**: tutte le partite del campionato (o del label selezionato).\n"
            "- **Scope Squadra**: partite della squadra Casa **giocate in Casa** e della squadra Ospite **giocate in Trasferta**.\n"
            "I ROI sono calcolati come profitto medio per scommessa (commissione default 4.5%)."
        )
        commission = 0.045

        df_ev_scope = df[df["Label"] == label].copy() if label else df.copy()
        df_ev_scope = df_ev_scope.dropna(subset=["Home Goal FT", "Away Goal FT"])

        OVER15_COLS = ["cotao1", "Odd Over 1.5", "odd over 1,5", "Over 1.5"]
        OVER25_COLS = ["cotao", "Odd Over 2.5", "odd over 2,5", "Over 2.5"]
        OVER35_COLS = ["cotao3", "Odd Over 3.5", "odd over 3,5", "Over 3.5"]
        BTTS_YES_COLS = ["gg", "GG", "odd goal", "BTTS Yes", "Odd BTTS Yes"]

        # ➜ Usa le quote condivise inserite sopra
        shared = _get_shared_quotes()
        q_ov15, q_ov25, q_ov35, q_btts = shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"]

        def _roi_table_for(df_scope: pd.DataFrame, title: str) -> pd.DataFrame:
            table_data = [
                _calc_market_roi(df_scope, "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15),
                _calc_market_roi(df_scope, "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25),
                _calc_market_roi(df_scope, "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35),
                _calc_market_roi(df_scope, "BTTS",     BTTS_YES_COLS, None, commission, q_btts),
            ]
            df_out = pd.DataFrame(table_data)
            df_out.insert(0, "Scope", title)
            return df_out

        df_roi_league = _roi_table_for(df_ev_scope, "Campionato/Label")
        df_home_ctx = df[(df["Home"] == squadra_casa)].dropna(subset=["Home Goal FT", "Away Goal FT"]).copy()
        df_roi_home = _roi_table_for(df_home_ctx, f"{squadra_casa} @Casa")
        df_away_ctx = df[(df["Away"] == squadra_ospite)].dropna(subset=["Home Goal FT", "Away Goal FT"]).copy()
        df_roi_away = _roi_table_for(df_away_ctx, f"{squadra_ospite} @Trasferta")

        df_roi_all = pd.concat([df_roi_league, df_roi_home, df_roi_away], ignore_index=True)

        st.dataframe(
            df_roi_all,
            use_container_width=True,
            height=420,
            column_config={
                "Quota Media": st.column_config.NumberColumn(format="%.2f"),
                "Esiti %": st.column_config.TextColumn(),
                "ROI Back %": st.column_config.TextColumn(),
                "ROI Lay %": st.column_config.TextColumn(),
                "Match Analizzati": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        _download_df_button(df_roi_all, "roi_markets_all_scopes.csv", "⬇️ Scarica ROI mercati (tutti gli scope)")

    # ==========================
    # TAB 3: EV Storico – Squadre + KPI Best EV
    # ==========================
    with tab_ev:
        st.markdown(
            "**Cosa vedi qui**: stime di Probabilità (%) per Home@Casa, Away@Trasferta, Blended, Head-to-Head. "
            "EV = quota × p − 1. Colonna **Qualità** = solidità del campione."
        )
        use_label = st.checkbox("Usa il filtro Label (se disponibile)", value=True if label else False, key=_k("use_label_ev_squadre"))
        last_n = st.slider("Limita agli ultimi N match (0 = tutti)", 0, 50, 0, key=_k("last_n_ev"))

        def _limit_last_n(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
            if n and n > 0 and "Data" in df_in.columns:
                s = pd.to_datetime(df_in["Data"], errors="coerce")
                tmp = df_in.copy()
                tmp["_data_"] = s
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

        df_h2h = df[
            ((df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)) |
            ((df["Home"] == squadra_ospite) & (df["Away"] == squadra_casa))
        ].copy()
        if use_label and label:
            df_h2h = df_h2h[df_h2h["Label"] == label]
        df_h2h = _limit_last_n(df_h2h.dropna(subset=["Home Goal FT", "Away Goal FT"]), last_n)

        # ➜ Usa quote condivise definite sopra
        shared = _get_shared_quotes()
        quota_ov15, quota_ov25, quota_ov35, quota_btts = shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"]

        df_ev_squadre, best = _build_ev_table(
            df_home_ctx, df_away_ctx, df_h2h,
            squadra_casa, squadra_ospite,
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
                """,
                unsafe_allow_html=True
            )
        else:
            st.info("Nessun EV positivo tra Blended e H2H con le quote inserite.")

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

    # ==========================
    # TAB 4: Statistiche squadre (usa squadre.py)
    # ==========================
    with tab_stats:
        render_team_stats_tab(
            df_league_all,     # campionato PRIMA del filtro stagioni globale
            league,
            squadra_casa,
            squadra_ospite,
        )

    # Aggiorna query params (opzionale)
    shared = _get_shared_quotes()
    _set_qparams(
        league=league,
        home=squadra_casa,
        away=squadra_ospite,
        q=st.session_state.get(_k("search_team"), "") or "",
        qh=odd_home, qd=odd_draw, qa=odd_away,
        ov15=shared["ov15"], ov25=shared["ov25"], ov35=shared["ov35"], btts=shared["btts"],
    )

    with tab_correct_score:
                   run_correct_score_panel(
        		df=df_league_all,                 # stesso df che usi nelle altre sezioni
        		league_code=league,               # es. db_selected upper
        		home_team=squadra_casa,
        		away_team=squadra_ospite,
        		seasons=seasons_selected or None  # se vuoi filtrare come la pagina
  	  )