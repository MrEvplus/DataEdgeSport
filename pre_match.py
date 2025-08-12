# pre_match.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from squadre import compute_team_macro_stats
from utils import label_match

# Proviamo a importare extract_minutes da utils; fallback se non c'è
try:
    from utils import extract_minutes as _extract_minutes_util
except Exception:
    _extract_minutes_util = None

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
    """Converte una Serie a float accettando virgole e stringhe."""
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

def _limit_last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Limita agli ultimi N match se presente la colonna Data (altrimenti ritorna df)."""
    if n and n > 0 and "Data" in df.columns:
        s = pd.to_datetime(df["Data"], errors="coerce")
        tmp = df.copy()
        tmp["_data_"] = s
        tmp = tmp.sort_values("_data_", ascending=False).drop(columns=["_data_"])
        return tmp.head(n)
    return df

# ==========================
# Goal minutes parsing & stato al minuto
# ==========================
HOME_MIN_COLS = [
    "minuti goal segnato home", "Minuti Goal Home", "mgolh",
    "home goal minuti", "minuti goal home"
]
AWAY_MIN_COLS = [
    "minuti goal segnato away", "Minuti Goal Away", "mgola",
    "away goal minuti", "minuti goal away"
]

# Colonne enumerate (fallback)
HOME_ENUM_COLS = [
    "home 1 goal segnato(min)", "home 2 goal segnato(min)", "home 3 goal segnato(min)",
    "home 4 goal segnato(min)", "home 5 goal segnato(min)", "home 6 goal segnato(min)",
    "home 7 goal segnato(min)", "home 8 goal segnato(min)", "home 9 goal segnato(min)",
    "gh1", "gh2", "gh3", "gh4", "gh5", "gh6", "gh7", "gh8", "gh9"
]
AWAY_ENUM_COLS = [
    "1 goal away (min)", "2 goal away (min)", "3 goal away (min)",
    "4 goal away (min)", "5 goal away (min)", "6 goal away (min)",
    "7 goal away (min)", "8 goal away (min)", "9 goal away (min)",
    "ga1", "ga2", "ga3", "ga4", "ga5", "ga6", "ga7", "ga8", "ga9"
]

def _extract_minutes(val) -> list[int]:
    """Parsa una lista di minuti da stringhe tipo '12;45;78'. Usa utils.extract_minutes se disponibile."""
    if _extract_minutes_util is not None:
        try:
            return _extract_minutes_util(val)  # gestisce sia Serie che stringhe
        except Exception:
            pass
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return []
    minutes = []
    for part in str(val).replace(",", ";").replace("|", ";").split(";"):
        p = part.strip()
        if p.isdigit():
            minutes.append(int(p))
    return minutes

def _row_goal_minutes(row, side: str) -> list[int]:
    """Ritorna la lista dei minuti goal per 'home' o 'away' cercando in varie colonne."""
    cols = HOME_MIN_COLS if side == "home" else AWAY_MIN_COLS
    enum_cols = HOME_ENUM_COLS if side == "home" else AWAY_ENUM_COLS

    # 1) colonne aggregate tipo 'minuti goal segnato home'
    for c in cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            lst = _extract_minutes(row[c])
            if lst:
                return sorted(lst)

    # 2) colonne enumerate (gh1, ga1, etc.)
    lst = []
    for c in enum_cols:
        if c in row and pd.notna(row[c]) and str(row[c]).strip() != "":
            try:
                v = int(float(str(row[c]).strip()))
                if v > 0:
                    lst.append(v)
            except Exception:
                continue
    if lst:
        return sorted(lst)

    # 3) fallback: se non abbiamo minuti, ritorna lista vuota
    return []

def _goals_until_minute(row, minute: int) -> tuple[int, int]:
    """Numero di gol segnati ENTRO (<=) quel minuto da Home/Away."""
    h_list = _row_goal_minutes(row, "home")
    a_list = _row_goal_minutes(row, "away")
    # Se non abbiamo minuti e siamo oltre 90', possiamo approssimare coi FT (serve per completeness)
    if not h_list and not a_list and minute >= 90:
        try:
            return int(row.get("Home Goal FT", 0)), int(row.get("Away Goal FT", 0))
        except Exception:
            return 0, 0
    h = sum(1 for m in h_list if m <= minute)
    a = sum(1 for m in a_list if m <= minute)
    return h, a

def _filter_by_state(df: pd.DataFrame, minute: int, h_cur: int, a_cur: int) -> pd.DataFrame:
    """Filtra le partite che al minuto indicato avevano esattamente quel punteggio."""
    if df.empty:
        return df
    rows = []
    for _, row in df.iterrows():
        hh, aa = _goals_until_minute(row, minute)
        if hh == h_cur and aa == a_cur:
            rows.append(row)
    if not rows:
        return df.iloc[0:0]
    return pd.DataFrame(rows).reset_index(drop=True)

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
                prices[k] = 2.0  # default

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

    col = _first_present(price_cols, df.columns)  # None se non presente
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
# Probabilità storiche (campionato/label) per EV
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

# ==========================
# Probabilità condizionate al minuto/punteggio (NUOVO)
# ==========================
def _market_prob_conditional(df: pd.DataFrame, market: str, line: float | None,
                             minute: int, h_cur: int, a_cur: int) -> float:
    """
    Probabilità (0-100) che il mercato si verifichi a FT, condizionata
    allo stato (minuto, punteggio live) usando lo storico.
    """
    if df.empty:
        return 0.0
    # Filtra match con stato identico al minuto
    df_state = _filter_by_state(df, minute, h_cur, a_cur)
    if df_state.empty:
        return 0.0
    # Valuta outcome a FT
    if market == "BTTS":
        ok = ((df_state["Home Goal FT"] > 0) & (df_state["Away Goal FT"] > 0)).mean()
    else:
        goals = pd.to_numeric(df_state["Home Goal FT"], errors="coerce").fillna(0) + \
                pd.to_numeric(df_state["Away Goal FT"], errors="coerce").fillna(0)
        ok = (goals > float(line)).mean() if line is not None else 0.0
    return round(float(ok) * 100, 2)

# ==========================
# ENTRY POINT
# ==========================
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title("⚔️ Confronto Pre Match")

    # Normalizzazioni coerenti col campionato
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

    # Label: calcolala una volta sola se manca
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # Stringhe squadre safe
    df["Home"] = _ensure_str(df["Home"]).str.strip()
    df["Away"] = _ensure_str(df["Away"]).str.strip()

    teams_available = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))

    # Stato iniziale
    if "squadra_casa" not in st.session_state:
        st.session_state["squadra_casa"] = teams_available[0] if teams_available else ""
    if "squadra_ospite" not in st.session_state:
        st.session_state["squadra_ospite"] = teams_available[1] if len(teams_available) > 1 else st.session_state["squadra_casa"]

    # Selettori squadre
    c1, c2 = st.columns(2)
    with c1:
        squadra_casa = st.selectbox(
            "Seleziona Squadra Casa",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_casa"]) if st.session_state["squadra_casa"] in teams_available else 0,
            key=_k(f"squadra_casa_{league}")
        )
    with c2:
        squadra_ospite = st.selectbox(
            "Seleziona Squadra Ospite",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_ospite"]) if st.session_state["squadra_ospite"] in teams_available else 0,
            key=_k(f"squadra_ospite_{league}")
        )

    # Quote 1X2 inserite manualmente (per label detection)
    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Quota Vincente Casa", min_value=1.01, step=0.01, value=st.session_state.get("quota_home", 2.00), key=_k("quota_home"))
        st.markdown(f"**Probabilità Casa ({squadra_casa}):** {round(100/odd_home, 2)}%")
    with c2:
        odd_draw = st.number_input("Quota Pareggio", min_value=1.01, step=0.01, value=st.session_state.get("quota_draw", 3.20), key=_k("quota_draw"))
        st.markdown(f"**Probabilità Pareggio:** {round(100/odd_draw, 2)}%")
    with c3:
        odd_away = st.number_input("Quota Vincente Ospite", min_value=1.01, step=0.01, value=st.session_state.get("quota_away", 3.80), key=_k("quota_away"))
        st.markdown(f"**Probabilità Ospite ({squadra_ospite}):** {round(100/odd_away, 2)}%")

    if not (squadra_casa and squadra_ospite and squadra_casa != squadra_ospite):
        st.info("Seleziona due squadre diverse per procedere.")
        return

    # Label dal range quote inserite
    label = _label_from_odds(float(odd_home), float(odd_away))
    label_type = _label_type(label)
    st.markdown(f"### 🎯 Range di quota identificato (Label): `{label}`")

    if label == "Others" or label not in set(df["Label"]):
        st.info("⚠️ Nessuna partita trovata per questo label nel campionato: uso l'intero campionato.")
        label = None

    # ==========================
    # League (per label se disponibile, altrimenti all league)
    # ==========================
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

    # ==========================
    # Squadra Casa / Ospite (1X2)
    # ==========================
    rows = [row_league]

    # Casa
    row_home = {"LABEL": squadra_casa}
    if label and label_type in ("Home", "Both"):
        df_home = df[(df["Label"] == label) & (df["Home"] == squadra_casa)]
        if df_home.empty:
            df_home = df[df["Home"] == squadra_casa]
            st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_casa}.")
    else:
        df_home = df[df["Home"] == squadra_casa]

    profits_back, rois_back, profits_lay, rois_lay, matches_home = _calc_back_lay_1x2(df_home)
    if matches_home > 0:
        wins_home = int((df_home["Home Goal FT"] > df_home["Away Goal FT"]).sum())
        draws_home = int((df_home["Home Goal FT"] == df_home["Away Goal FT"]).sum())
        losses_home = int((df_home["Home Goal FT"] < df_home["Away Goal FT"]).sum())
        row_home["MATCHES"] = matches_home
        row_home["BACK WIN% HOME"] = round((wins_home / matches_home) * 100, 2)
        row_home["BACK WIN% DRAW"] = round((draws_home / matches_home) * 100, 2)
        row_home["BACK WIN% AWAY"] = round((losses_home / matches_home) * 100, 2)
    else:
        row_home["MATCHES"] = 0
        row_home["BACK WIN% HOME"] = row_home["BACK WIN% DRAW"] = row_home["BACK WIN% AWAY"] = 0.0
    for outcome in ("HOME", "DRAW", "AWAY"):
        row_home[f"BACK PTS {outcome}"] = _format_value(profits_back[outcome])
        row_home[f"BACK ROI% {outcome}"] = _format_value(rois_back[outcome], is_roi=True)
        row_home[f"Lay pts {outcome}"] = _format_value(profits_lay[outcome])
        row_home[f"lay ROI% {outcome}"] = _format_value(rois_lay[outcome], is_roi=True)
    rows.append(row_home)

    # Ospite
    row_away = {"LABEL": squadra_ospite}
    if label and label_type in ("Away", "Both"):
        df_away = df[(df["Label"] == label) & (df["Away"] == squadra_ospite)]
        if df_away.empty:
            df_away = df[df["Away"] == squadra_ospite]
            st.info(f"⚠️ Nessuna partita per questo label. Uso tutte le partite di {squadra_ospite}.")
    else:
        df_away = df[df["Away"] == squadra_ospite]

    profits_back, rois_back, profits_lay, rois_lay, matches_away = _calc_back_lay_1x2(df_away)
    if matches_away > 0:
        wins_away = int((df_away["Away Goal FT"] > df_away["Home Goal FT"]).sum())
        draws_away = int((df_away["Away Goal FT"] == df_away["Home Goal FT"]).sum())
        losses_away = int((df_away["Away Goal FT"] < df_away["Home Goal FT"]).sum())
        row_away["MATCHES"] = matches_away
        row_away["BACK WIN% HOME"] = round((losses_away / matches_away) * 100, 2)
        row_away["BACK WIN% DRAW"] = round((draws_away / matches_away) * 100, 2)
        row_away["BACK WIN% AWAY"] = round((wins_away / matches_away) * 100, 2)
    else:
        row_away["MATCHES"] = 0
        row_away["BACK WIN% HOME"] = row_away["BACK WIN% DRAW"] = row_away["BACK WIN% AWAY"] = 0.0
    for outcome in ("HOME", "DRAW", "AWAY"):
        row_away[f"BACK PTS {outcome}"] = _format_value(profits_back[outcome])
        row_away[f"BACK ROI% {outcome}"] = _format_value(rois_back[outcome], is_roi=True)
        row_away[f"Lay pts {outcome}"] = _format_value(profits_lay[outcome])
        row_away[f"lay ROI% {outcome}"] = _format_value(rois_lay[outcome], is_roi=True)
    rows.append(row_away)

    # Tabella long 1X2
    rows_long = []
    for row in rows:
        for outcome in ("HOME", "DRAW", "AWAY"):
            rows_long.append({
                "LABEL": row["LABEL"],
                "SEGNO": outcome,
                "Matches": row.get("MATCHES", 0),
                "Win %": row.get(f"BACK WIN% {outcome}", 0),
                "Back Pts": row.get(f"BACK PTS {outcome}", _format_value(0)),
                "Back ROI %": row.get(f"BACK ROI% {outcome}", _format_value(0, is_roi=True)),
                "Lay Pts": row.get(f"Lay pts {outcome}", _format_value(0)),
                "Lay ROI %": row.get(f"lay ROI% {outcome}", _format_value(0, is_roi=True)),
            })
    df_long = pd.DataFrame(rows_long)
    df_long.loc[df_long.duplicated(subset=["LABEL"]), "LABEL"] = ""

    st.markdown(f"#### Range di quota identificato (Label): `{label or 'ALL'}`")
    st.dataframe(df_long, use_container_width=True)

    # ==========================
    # Confronto macro-kpi squadre
    # ==========================
    st.markdown("---")
    st.markdown("## 📊 Confronto Statistiche Pre-Match")
    stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
    stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")

    if not stats_home or not stats_away:
        st.info("⚠️ Una delle due squadre non ha match disponibili per il confronto.")
        return

    df_comp = pd.DataFrame({squadra_casa: stats_home, squadra_ospite: stats_away})
    st.dataframe(df_comp, use_container_width=True)

    st.success("✅ Confronto Pre Match generato con successo!")
    st.header("📈 ROI Back & Lay + EV Live (Over e BTTS)")

    # ==========================
    # ROI mercati (campionato/label)
    # ==========================
    st.markdown("### 🎯 Calcolo ROI Back & Lay su Over 1.5, 2.5, 3.5 e BTTS (campionato/label)")
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

    table_data = []
    table_data.append(_calc_market_roi(df_ev_scope, "Over 1.5", OVER15_COLS, 1.5, commission, q_ov15))
    table_data.append(_calc_market_roi(df_ev_scope, "Over 2.5", OVER25_COLS, 2.5, commission, q_ov25))
    table_data.append(_calc_market_roi(df_ev_scope, "Over 3.5", OVER35_COLS, 3.5, commission, q_ov35))
    table_data.append(_calc_market_roi(df_ev_scope, "BTTS", BTTS_YES_COLS, None, commission, q_btts))
    df_ev = pd.DataFrame(table_data)
    st.dataframe(df_ev, use_container_width=True)

    # ==========================
    # 🧠 EV Storico – Squadre selezionate
    # ==========================
    st.markdown("---")
    st.markdown("## 🧠 EV Storico – Squadre selezionate")

    use_label = st.checkbox("Usa il filtro Label per il calcolo (se disponibile)", value=bool(label), key=_k("use_label_ev_squadre"))
    last_n = st.slider("Limita agli ultimi N match (0 = tutti)", 0, 30, 0, key=_k("last_n_ev"))

    # Sottoinsiemi
    df_home_ctx = df[(df["Home"] == squadra_casa)].copy()
    df_away_ctx = df[(df["Away"] == squadra_ospite)].copy()

    if use_label and label:
        df_home_ctx = df_home_ctx[df_home_ctx["Label"] == label]
        df_away_ctx = df_away_ctx[df_away_ctx["Label"] == label]

    df_home_ctx = df_home_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"])
    df_away_ctx = df_away_ctx.dropna(subset=["Home Goal FT", "Away Goal FT"])

    df_home_ctx = _limit_last_n(df_home_ctx, last_n)
    df_away_ctx = _limit_last_n(df_away_ctx, last_n)

    df_blend = pd.concat([df_home_ctx, df_away_ctx], ignore_index=True)
    df_h2h = df[
        ((df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)) |
        ((df["Home"] == squadra_ospite) & (df["Away"] == squadra_casa))
    ].copy()
    if use_label and label:
        df_h2h = df_h2h[df_h2h["Label"] == label]
    df_h2h = df_h2h.dropna(subset=["Home Goal FT", "Away Goal FT"])
    df_h2h = _limit_last_n(df_h2h, last_n)

    # Quote live per EV (usate sia per storico generale sia per condizionale)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        quota_ov15 = st.number_input("Quota Live Over 1.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov15"))
    with c2:
        quota_ov25 = st.number_input("Quota Live Over 2.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov25"))
    with c3:
        quota_ov35 = st.number_input("Quota Live Over 3.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov35"))
    with c4:
        quota_btts = st.number_input("Quota Live BTTS", min_value=1.01, step=0.01, value=2.00, key=_k("ev_btts"))

    markets = [
        ("Over 1.5", 1.5, quota_ov15),
        ("Over 2.5", 2.5, quota_ov25),
        ("Over 3.5", 3.5, quota_ov35),
        ("BTTS", None, quota_btts),
    ]

    # === EV storico “semplice” (non condizionato al minuto)
    rows_ev = []
    prob_chart_rows = []
    for name, line, q in markets:
        p_home = _market_prob(df_home_ctx, name, line)
        p_away = _market_prob(df_away_ctx, name, line)
        p_blnd = round((p_home + p_away) / 2, 2) if (p_home > 0 or p_away > 0) else 0.0
        p_h2h  = _market_prob(df_h2h, name, line)

        ev_home = round(q * (p_home / 100) - 1, 2)
        ev_away = round(q * (p_away / 100) - 1, 2)
        ev_blnd = round(q * (p_blnd / 100) - 1, 2)
        ev_h2h  = round(q * (p_h2h / 100) - 1, 2)

        rows_ev.append({
            "Mercato": name,
            "Quota": q,
            f"{squadra_casa} @Casa %": p_home,
            f"EV {squadra_casa}": ev_home,
            f"{squadra_ospite} @Trasferta %": p_away,
            f"EV {squadra_ospite}": ev_away,
            "Blended %": p_blnd,
            "EV Blended": ev_blnd,
            "Head-to-Head %": p_h2h,
            "EV H2H": ev_h2h,
            "Match H": len(df_home_ctx),
            "Match A": len(df_away_ctx),
            "Match H2H": len(df_h2h),
        })

        for label_scope, prob in [
            (f"{squadra_casa} @Casa", p_home),
            (f"{squadra_ospite} @Trasferta", p_away),
            ("Blended", p_blnd),
            ("Head-to-Head", p_h2h),
        ]:
            prob_chart_rows.append({"Mercato": name, "Scope": label_scope, "Prob %": prob})

    st.subheader("📋 EV storico per mercati (squadre selezionate)")
    st.dataframe(pd.DataFrame(rows_ev), use_container_width=True)

    st.subheader("📊 Probabilità storiche per mercato e scope")
    df_prob = pd.DataFrame(prob_chart_rows)
    if not df_prob.empty:
        chart = alt.Chart(df_prob).mark_bar().encode(
            x=alt.X("Mercato:N"),
            y=alt.Y("Prob %:Q"),
            color=alt.Color("Scope:N"),
            column=alt.Column("Scope:N", header=alt.Header(title="")),
            tooltip=["Mercato", "Scope", alt.Tooltip("Prob %:Q", format=".1f")]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Nessun dato sufficiente per il grafico delle probabilità.")

    # ==========================
    # ⚡️ EV Live condizionato al minuto & punteggio (NUOVO)
    # ==========================
    st.markdown("---")
    st.markdown("## ⚡️ EV Live condizionato al minuto & punteggio")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        minute = st.slider("Minuto attuale di gioco", 0, 120, 60, key=_k("minute_live"))
    with c2:
        h_cur = st.number_input("Gol Home (live)", min_value=0, step=1, value=0, key=_k("h_cur"))
    with c3:
        a_cur = st.number_input("Gol Away (live)", min_value=0, step=1, value=0, key=_k("a_cur"))

    # Filtra i dataset sulle partite che al minuto avevano quel punteggio
    df_home_m = _filter_by_state(df_home_ctx, minute, h_cur, a_cur)
    df_away_m = _filter_by_state(df_away_ctx, minute, h_cur, a_cur)
    df_blend_m = _filter_by_state(df_blend, minute, h_cur, a_cur)
    df_h2h_m = _filter_by_state(df_h2h, minute, h_cur, a_cur)

    rows_live = []
    prob_live_rows = []
    for name, line, q in markets:
        p_home_m = _market_prob_conditional(df_home_ctx, name, line, minute, h_cur, a_cur)
        p_away_m = _market_prob_conditional(df_away_ctx, name, line, minute, h_cur, a_cur)
        p_blnd_m = _market_prob_conditional(df_blend,    name, line, minute, h_cur, a_cur)
        p_h2h_m  = _market_prob_conditional(df_h2h,      name, line, minute, h_cur, a_cur)

        ev_home_m = round(q * (p_home_m / 100) - 1, 2)
        ev_away_m = round(q * (p_away_m / 100) - 1, 2)
        ev_blnd_m = round(q * (p_blnd_m / 100) - 1, 2)
        ev_h2h_m  = round(q * (p_h2h_m / 100) - 1, 2)

        rows_live.append({
            "Mercato": name,
            "Quota Live": q,
            f"{squadra_casa} @Casa % (m{minute} {h_cur}-{a_cur})": p_home_m,
            f"EV {squadra_casa} Live": ev_home_m,
            f"{squadra_ospite} @Trasferta % (m{minute} {h_cur}-{a_cur})": p_away_m,
            f"EV {squadra_ospite} Live": ev_away_m,
            "Blended % Live": p_blnd_m,
            "EV Blended Live": ev_blnd_m,
            "Head-to-Head % Live": p_h2h_m,
            "EV H2H Live": ev_h2h_m,
            "Match H (state)": len(df_home_m),
            "Match A (state)": len(df_away_m),
            "Match H2H (state)": len(df_h2h_m),
        })

        for label_scope, prob in [
            (f"{squadra_casa} @Casa (m{minute})", p_home_m),
            (f"{squadra_ospite} @Trasferta (m{minute})", p_away_m),
            ("Blended (m{})".format(minute), p_blnd_m),
            ("Head-to-Head (m{})".format(minute), p_h2h_m),
        ]:
            prob_live_rows.append({"Mercato": name, "Scope": label_scope, "Prob %": prob})

    st.subheader("📋 EV Live (condizionato a minuto & punteggio)")
    st.dataframe(pd.DataFrame(rows_live), use_container_width=True)

    st.subheader("📊 Probabilità condizionate (minuto/punteggio)")
    df_prob_live = pd.DataFrame(prob_live_rows)
    if not df_prob_live.empty:
        chart_live = alt.Chart(df_prob_live).mark_bar().encode(
            x=alt.X("Mercato:N"),
            y=alt.Y("Prob %:Q"),
            color=alt.Color("Scope:N"),
            column=alt.Column("Scope:N", header=alt.Header(title="")),
            tooltip=["Mercato", "Scope", alt.Tooltip("Prob %:Q", format=".1f")]
        ).properties(height=300)
        st.altair_chart(chart_live, use_container_width=True)
    else:
        st.info("Nessun match storico con lo stesso stato (minuto/punteggio). Prova a cambiare parametri o togliere il filtro Label.")

    # ==========================
    # EV Manuale (campionato/label) – riepilogo
    # ==========================
    st.markdown("---")
    st.markdown("## 📌 EV Manuale (campionato/label) – riferimento rapido")
    ev_rows = []
    for name, q, line in [
        ("Over 1.5", quota_ov15, 1.5),
        ("Over 2.5", quota_ov25, 2.5),
        ("Over 3.5", quota_ov35, 3.5),
        ("BTTS",  quota_btts, None),
    ]:
        if name == "BTTS":
            prob_hist = float(((df_ev_scope["Home Goal FT"] > 0) & (df_ev_scope["Away Goal FT"] > 0)).mean() * 100) if not df_ev_scope.empty else 0.0
        else:
            prob_hist = float(((pd.to_numeric(df_ev_scope["Home Goal FT"], errors="coerce").fillna(0) +
                                pd.to_numeric(df_ev_scope["Away Goal FT"], errors="coerce").fillna(0)) > (line or 0)).mean() * 100) if not df_ev_scope.empty else 0.0
        ev = round(q * (prob_hist / 100) - 1, 2)
        nota = "🟢 EV+" if ev > 0 else ("🔴 EV-" if ev < 0 else "⚪️ Neutro")
        ev_rows.append({
            "Mercato": name,
            "Quota": q,
            "Probabilità Storica Campionato/Label": f"{prob_hist:.1f}%",
            "EV": ev,
            "Note": nota
        })
    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)
