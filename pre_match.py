# pre_match.py
from __future__ import annotations

import re
from datetime import date
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# Moduli interni
from squadre import compute_team_macro_stats, render_team_stats_tab
from correct_score import run_correct_score_panel
from utils import label_match  # usato in modo difensivo

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
    """Prova ad usare label_match, altrimenti fa un fallback semplice."""
    try:
        # Alcune versioni di label_match lavorano su dict/row
        return label_match({"Odd home": home_odd, "Odd Away": away_odd})
    except Exception:
        if home_odd < away_odd:
            return "H_favorite"
        if away_odd < home_odd:
            return "A_favorite"
        return "Both"

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
    """Cerca 'stagione in corso' in formato 2024/25 o 2024-25, ecc."""
    if not seasons_desc:
        return []
    today = date.today()
    if today.month >= 7:
        sy, ey = today.year, today.year + 1
    else:
        sy, ey = today.year - 1, today.year
    ey2 = str(ey)[-2:]
    cands = [f"{sy}/{ey}", f"{sy}-{ey}", f"{sy}/{ey2}", f"{sy}-{ey2}", str(sy), str(ey)]
    for c in cands:
        for s in seasons_desc:
            if s.strip() == c:
                return [s]
    # fallback: match parziale
    for s in seasons_desc:
        if str(sy) in s or str(ey) in s:
            return [s]
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


# ==========================
# ROI 1X2 e mercati O/U/BTTS (robusti)
# ==========================
def _calc_back_lay_1x2(df: pd.DataFrame, commission: float = 0.0):
    """ROI medio su 1X2 con quote storiche se presenti (fallback su 2.0)."""
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
        prices = {"HOME": float(row.get("Odd home", np.nan)),
                  "DRAW": float(row.get("Odd Draw", np.nan)),
                  "AWAY": float(row.get("Odd Away", np.nan))}
        for k, v in prices.items():
            if not (v and v > 1.0 and np.isfinite(v)):
                prices[k] = 2.0  # fallback prudente

        for outcome in ("HOME", "DRAW", "AWAY"):
            p = prices[outcome]
            # Back
            if result == outcome:
                profits_back[outcome] += (p - 1) * (1 - commission)
            else:
                profits_back[outcome] -= 1
            # Lay
            stake = 1.0 / (p - 1.0)
            if result != outcome:
                profits_lay[outcome] += stake
            else:
                profits_lay[outcome] -= 1.0

    denom = len(valid_rows)
    rois_back = {k: round((v / denom) * 100, 2) for k, v in profits_back.items()}
    rois_lay  = {k: round((v / denom) * 100, 2) for k, v in profits_lay.items()}
    return profits_back, rois_back, profits_lay, rois_lay, denom


def _calc_market_roi(df: pd.DataFrame, market: str, price_cols: list[str],
                     line: float | None, commission: float,
                     manual_price: float | None = None):
    """ROI medio su Over/BTTS (usa quote storiche se presenti, altrimenti manuali)."""
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

        total += 1; qsum += o; qcount += 1
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

    return {"Mercato": market, "Quota Media": avg_quote, "Esiti %": f"{pct}%",
            "ROI Back %": f"{roi_back}%", "ROI Lay %": f"{roi_lay}%", "Match Analizzati": total}


# ==========================
# Probabilità storiche per EV
# ==========================
def _market_prob(df: pd.DataFrame, market: str, line: float | None) -> float:
    if df.empty: return 0.0
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

def _build_ev_table(df_home_ctx: pd.DataFrame, df_away_ctx: pd.DataFrame, df_h2h: pd.DataFrame,
                    squadra_casa: str, squadra_ospite: str,
                    quota_ov15: float, quota_ov25: float, quota_ov35: float, quota_btts: float):
    markets = [("Over 1.5", 1.5, quota_ov15), ("Over 2.5", 2.5, quota_ov25),
               ("Over 3.5", 3.5, quota_ov35), ("BTTS", None, quota_btts)]
    rows, best_list = [], []
    for name, line, q in markets:
        p_home = _market_prob(df_home_ctx, name, line)
        p_away = _market_prob(df_away_ctx, name, line)
        p_blnd = round((p_home + p_away) / 2, 2) if (p_home > 0 or p_away > 0) else 0.0
        p_h2h  = _market_prob(df_h2h, name, line)
        ev_home = round(q * (p_home / 100) - 1, 2)
        ev_away = round(q * (p_away / 100) - 1, 2)
        ev_blnd = round(q * (p_blnd / 100) - 1, 2)
        ev_h2h  = round(q * (p_h2h / 100) - 1, 2)
        n_h, n_a, n_h2h = len(df_home_ctx), len(df_away_ctx), len(df_h2h)
        qual_blnd = _quality_label(n_h + n_a); qual_h2h = _quality_label(n_h2h)
        rows.append({"Mercato": name, "Quota": q,
                     f"{squadra_casa} @Casa %": p_home, f"EV {squadra_casa}": ev_home,
                     f"{squadra_ospite} @Trasferta %": p_away, f"EV {squadra_ospite}": ev_away,
                     "Blended %": p_blnd, "EV Blended": ev_blnd, "Qualità Blended": qual_blnd,
                     "Head-to-Head %": p_h2h, "EV H2H": ev_h2h, "Qualità H2H": qual_h2h,
                     "Match H": n_h, "Match A": n_a, "Match H2H": n_h2h})
        best_list += [
            {"scope": "Blended", "mercato": name, "ev": ev_blnd, "qualita": qual_blnd},
            {"scope": "Head-to-Head", "mercato": name, "ev": ev_h2h, "qualita": qual_h2h},
        ]
    df_ev = pd.DataFrame(rows)
    best = None
    for c in sorted(best_list, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c; break
    return df_ev, best


# ==========================
# ENTRY POINT (PRE-MATCH)
# ==========================
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title("📊 Pre-Match – Analisi per Trader Professionisti")

    qin = _get_qparams()
    _init_shared_quotes()  # quote condivise

    # Normalizzazione e filtro campionato
    if "country" not in df.columns:
        st.error("Dataset senza colonna 'country'."); st.stop()
    df = df.copy()
    df["country"] = _ensure_str(df["country"]).str.strip().str.upper()
    league = (db_selected or "").strip().upper()
    df = df[df["country"] == league]
    if df.empty:
        st.warning(f"Nessun dato per '{league}'."); st.stop()

    # snapshot per Tab Statistiche (prima del filtro stagioni globale)
    df_league_all = df.copy()

    df["Home"] = _ensure_str(df["Home"]).str.strip()
    df["Away"] = _ensure_str(df["Away"]).str.strip()
    if "Label" not in df.columns:
        # calcoliamo etichetta in modo tolerante
        try:
            df["Label"] = df.apply(label_match, axis=1)
        except Exception:
            df["Label"] = "Both"

    # ======== Filtro Stagioni (globale) ========
    seasons_selected: list[str] = []
    if "Stagione" in df.columns:
        seasons_desc = _seasons_desc(df["Stagione"].dropna().unique().tolist())
        latest = seasons_desc[0] if seasons_desc else None
        with st.expander("⚙️ Filtro Stagioni (globale per 1X2/ROI/EV/Correct Score)", expanded=True):
            st.markdown("**Intervallo rapido** parte dalla stagione più recente.")
            colA, colB = st.columns([2, 1])
            with colA:
                seasons_selected = st.multiselect("Selezione Manuale", options=seasons_desc,
                                                  default=[latest] if latest else [], key=_k("stagioni_manual"))
            with colB:
                preset = st.selectbox("Intervallo rapido",
                                      options=["Tutte", "Stagione in corso", "Ultime 10", "Ultime 5", "Ultime 3", "Ultime 2", "Ultime 1"],
                                      index=0, key=_k("stagioni_preset"))
                if preset == "Stagione in corso":
                    seasons_selected = _pick_current_season(seasons_desc)
                elif preset != "Tutte" and seasons_desc:
                    try: n = int(preset.split()[-1])
                    except Exception: n = 1
                    seasons_selected = seasons_desc[:n]
                else:
                    seasons_selected = seasons_selected  # già scelto manualmente
            if seasons_selected:
                st.caption(f"Stagioni attive: **{', '.join(seasons_selected)}**")
                df = df[df["Stagione"].astype(str).isin(seasons_selected)]
            else:
                st.caption("Stagioni attive: **Tutte**")

    # ======== Ricerca/assegnazione squadre ========
    all_teams = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))
    with st.expander("🔎 Ricerca rapida squadre (assegna come Casa/Ospite)", expanded=True):
        col_search, col_role, col_btn = st.columns([2, 1, 1])
        with col_search:
            team_filter = st.text_input("Cerca squadra", value=qin.get("q", ""), key=_k("search_team"))
            flt = str(team_filter or "").strip().lower()
            filtered = [t for t in all_teams if flt in t.lower()] or all_teams
            selected_from_search = st.selectbox("Risultati ricerca", options=filtered, key=_k("search_pick"))
        with col_role:
            role = st.radio("Assegna a", options=["Casa", "Ospite"], horizontal=True, key=_k("search_role"))
        with col_btn:
            if st.button("Assegna", key=_k("assign_btn"), use_container_width=True):
                if role == "Casa":
                    if st.session_state.get(_k("squadra_ospite")) == selected_from_search:
                        st.warning("⚠️ Già selezionata come Ospite.")
                    else:
                        st.session_state[_k("squadra_casa")] = selected_from_search
                else:
                    if st.session_state.get(_k("squadra_casa")) == selected_from_search:
                        st.warning("⚠️ Già selezionata come Casa.")
                    else:
                        st.session_state[_k("squadra_ospite")] = selected_from_search
                _set_qparams(q=team_filter or "")
                try: st.rerun()
                except Exception: st.experimental_rerun()

    # ======== Selettori finiti + swap ========
    default_home = 0; default_away = 1 if len(all_teams) > 1 else 0
    col_sel1, col_swap, col_sel2 = st.columns([2, 0.5, 2])
    with col_sel1:
        squadra_casa = st.selectbox("Seleziona Squadra Casa", options=all_teams, index=default_home, key=_k("squadra_casa"))
    with col_swap:
        st.write("")
        if st.button("🔁 Inverti", use_container_width=True, key=_k("swap")):
            home = st.session_state.get(_k("squadra_casa"), all_teams[default_home])
            away = st.session_state.get(_k("squadra_ospite"), all_teams[default_away])
            if home == away:
                st.warning("⚠️ Casa e Ospite sono uguali.")
            else:
                st.session_state[_k("squadra_casa")] = away
                st.session_state[_k("squadra_ospite")] = home
                try: st.rerun()
                except Exception: st.experimental_rerun()
    with col_sel2:
        squadra_ospite = st.selectbox("Seleziona Squadra Ospite", options=all_teams, index=default_away, key=_k("squadra_ospite"))

    if squadra_casa == squadra_ospite:
        st.error("❌ Casa e Ospite uguali."); return

    # ======== Quote 1X2 + Over/BTTS condivise ========
    st.markdown("Le **quote Over/BTTS** qui sotto sono **condivise** tra i tab.")
    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Quota Vincente Casa", min_value=1.01, step=0.01, value=2.00, key=_k("quota_home"))
    with c2:
        odd_draw = st.number_input("Quota Pareggio", min_value=1.01, step=0.01, value=3.20, key=_k("quota_draw"))
    with c3:
        odd_away = st.number_input("Quota Vincente Ospite", min_value=1.01, step=0.01, value=3.80, key=_k("quota_away"))

    st.markdown("### Quote Over/BTTS (condivise)")
    q1, q2, q3, q4 = st.columns(4)
    with q1: _shared_number_input("Quota Over 1.5", "ov15", _k("shared:q_ov15"))
    with q2: _shared_number_input("Quota Over 2.5", "ov25", _k("shared:q_ov25"))
    with q3: _shared_number_input("Quota Over 3.5", "ov35", _k("shared:q_ov35"))
    with q4: _shared_number_input("Quota BTTS",     "btts", _k("shared:q_btts"))
    shared = _get_shared_quotes()

    label = _label_from_odds(float(odd_home), float(odd_away))
    header_cols = st.columns([1, 1, 1, 1])
    header_cols[0].markdown(f"**Campionato:** `{league}`")
    header_cols[1].markdown(f"**Label:** `{label}`")
    header_cols[2].markdown(f"**Home:** `{squadra_casa}`")
    header_cols[3].markdown(f"**Away:** `{squadra_ospite}`")

    # ======== Dataset di contesto (servono a ROI/EV/CorrectScore) ========
    df_home_ctx = df[df["Home"].astype("string") == squadra_casa].copy()
    df_away_ctx = df[df["Away"].astype("string") == squadra_ospite].copy()
    df_h2h = df[((df["Home"].astype("string") == squadra_casa) & (df["Away"].astype("string") == squadra_ospite)) |
                ((df["Home"].astype("string") == squadra_ospite) & (df["Away"].astype("string") == squadra_casa))].copy()

    # ======== Tabs ========
    tab_1x2, tab_roi, tab_ev, tab_stats, tab_cs = st.tabs(["1X2", "ROI mercati", "EV storico squadre", "Statistiche squadre", "Correct Score"])

    # --- 1X2
    with tab_1x2:
        st.subheader("1X2 – ROI storico (se presenti quote registrate)")
        st.caption("Se il dataset non contiene colonne di quote storiche, il calcolo usa un fallback neutrale (2.00).")
        profits_back, rois_back, profits_lay, rois_lay, n = _calc_back_lay_1x2(df)
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**Back – ROI %**")
            st.table(pd.DataFrame(rois_back, index=["ROI %"]).T)
        with colB:
            st.markdown("**Lay – ROI %**")
            st.table(pd.DataFrame(rois_lay, index=["ROI %"]).T)
        st.caption(f"Match analizzati: **{n}**")

    # --- ROI mercati
    with tab_roi:
        st.subheader("ROI mercati (storico – campionato/filtri selezionati)")
        st.caption("Le quote manuali in alto vengono usate se mancano colonne storiche.")
        rows = []
        rows.append(_calc_market_roi(df, "Over 1.5", ["Odd Over 1.5", "O1.5"], 1.5, commission=0.0, manual_price=shared["ov15"]))
        rows.append(_calc_market_roi(df, "Over 2.5", ["Odd Over 2.5", "O2.5"], 2.5, commission=0.0, manual_price=shared["ov25"]))
        rows.append(_calc_market_roi(df, "Over 3.5", ["Odd Over 3.5", "O3.5"], 3.5, commission=0.0, manual_price=shared["ov35"]))
        rows.append(_calc_market_roi(df, "BTTS",       ["Odd BTTS",       "BTTS"], None, commission=0.0, manual_price=shared["btts"]))
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- EV storico squadre
    with tab_ev:
        st.subheader("EV Storico – Home vs Away")
        st.caption("Scope: **Blended** = media Home@Casa e Away@Trasferta. **Head-to-Head** = scontri diretti.")
        ev_df, best = _build_ev_table(df_home_ctx, df_away_ctx, df_h2h,
                                      squadra_casa, squadra_ospite,
                                      shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"])
        st.dataframe(ev_df, use_container_width=True, hide_index=True)
        if best:
            st.success(f"**Best EV**: {best['mercato']} ({best['scope']}) → EV **{best['ev']:.2f}** · Qualità **{best['qualita']}**")
        else:
            st.info("Nessun EV positivo con le quote attuali.")

    # --- Statistiche squadre (usa df_league_all e filtro stagioni locale interno)
    with tab_stats:
        render_team_stats_tab(df_league_all, league, squadra_casa, squadra_ospite)

    # --- Correct Score (nuovo pannello)
    with tab_cs:
        st.subheader("Correct Score – Poisson + Dixon-Coles")
        st.caption("Stima λ con shrink su media di lega + forma recente. Heatmap, top score e EV su punteggi corretti.")
        run_correct_score_panel(
            df=df_league_all,           # uso il dataset di lega non ristretto dalle stagioni globali? Puoi passare df per coerente
            league_code=league,
            home_team=squadra_casa,
            away_team=squadra_ospite,
            seasons=seasons_selected or None,  # applica lo stesso filtro stagionale globale
            default_rho=-0.05,
            default_kappa=3.0,
            default_recent_weight=0.25,
            default_recent_n=6,
            default_max_goals=6,
        )

    # Query params (comodi per share link)
    _set_qparams(league=league, home=squadra_casa, away=squadra_ospite,
                 qh=odd_home, qd=odd_draw, qa=odd_away,
                 ov15=shared["ov15"], ov25=shared["ov25"], ov35=shared["ov35"], btts=shared["btts"])
