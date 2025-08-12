# pre_match.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

from utils import label_match
from squadre import compute_team_macro_stats

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
        import pandas as pd  # local ref
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.astype("string")
    except Exception:
        pass
    return s.astype("string")

def _coerce_float(s: pd.Series) -> pd.Series:
    """Converte una Serie a float accettando virgole e stringhe."""
    if s is None:
        return pd.Series(dtype="float")
    return pd.to_numeric(
        _ensure_str(s).str.replace(",", ".", regex=False),
        errors="coerce"
    )

def _first_present(cols: list[str], columns: pd.Index) -> str | None:
    """Ritorna il primo nome colonna presente tra quelli proposti."""
    for c in cols:
        if c in columns:
            return c
    return None

def _get_val(row: pd.Series, alternatives: list[str], default=np.nan):
    for c in alternatives:
        if c in row and pd.notna(row[c]):
            return row[c]
    return default

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

# ==========================
# Dati “league” per label
# ==========================
def _league_data_by_label(df: pd.DataFrame, label: str) -> dict | None:
    if "Label" not in df.columns:
        df = df.copy()
        df["Label"] = df.apply(label_match, axis=1)

    # risultato 1X2
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
    """
    Calcola profitti e ROI% per BACK e LAY (liability fissa=1) su HOME/DRAW/AWAY.
    commission: usata solo per BACK (se si vuole simularla), 0 di default.
    """
    if df.empty:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, zero, zero, 0

    # Quote come float
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
        # sanitize
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
            # LAY liability=1
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
    - price_cols: sinonimi accettati per la colonna quota (es. ["cotao", "Odd Over 2.5", "odd over 2,5"])
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
        # riempi i buchi con manual_price se presente
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

    # DEBUG opzionale
    with st.expander("🔧 DEBUG", expanded=False):
        st.write("Campionato:", league)
        st.write("Label attivo:", label)
        st.write("Righe campionato:", len(df))

    # ==========================
    # League (per label se disponibile, altrimenti all league)
    # ==========================
    df_league_scope = df[df["Label"] == label] if label else df
    profits_back, rois_back, profits_lay, rois_lay, matches_league = _calc_back_lay_1x2(df_league_scope)

    league_stats = _league_data_by_label(df, label) if label else _league_data_by_label(df, _label_from_odds(2.0, 2.0))  # fallback neutro
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
    # Squadra Casa
    # ==========================
    rows = [row_league]
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

    # ==========================
    # Squadra Ospite
    # ==========================
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
        # prospettiva “segno”: per la squadra away, la % HOME WIN è la % di sconfitte
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

    # Tabella long
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
    # ROI mercati (solo label o tutto)
    # ==========================
    st.markdown("### 🎯 Calcolo ROI Back & Lay su Over 1.5, 2.5, 3.5 e BTTS")
    commission = 0.045

    # Scope per EV/ROI mercati: usa label se disponibile, altrimenti tutto il campionato
    df_ev_scope = df[df["Label"] == label].copy() if label else df.copy()
    # filtra solo match con punteggi noti
    df_ev_scope = df_ev_scope.dropna(subset=["Home Goal FT", "Away Goal FT"])

    # Sinonimi colonne quote
    OVER15_COLS = ["cotao1", "Odd Over 1.5", "odd over 1,5", "Over 1.5"]
    OVER25_COLS = ["cotao", "Odd Over 2.5", "odd over 2,5", "Over 2.5"]
    OVER35_COLS = ["cotao3", "Odd Over 3.5", "odd over 3,5", "Over 3.5"]
    BTTS_YES_COLS = ["gg", "GG", "odd goal", "BTTS Yes", "Odd BTTS Yes"]

    # Quote manuali di fallback
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
    # EV manuale (placeholder probabilità)
    # ==========================
    st.markdown("## 🧠 Expected Value (EV) Manuale")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        quota_ov15 = st.number_input("Quota Live Over 1.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov15"))
    with c2:
        quota_ov25 = st.number_input("Quota Live Over 2.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov25"))
    with c3:
        quota_ov35 = st.number_input("Quota Live Over 3.5", min_value=1.01, step=0.01, value=2.00, key=_k("ev_ov35"))
    with c4:
        quota_btts = st.number_input("Quota Live BTTS", min_value=1.01, step=0.01, value=2.00, key=_k("ev_btts"))

    # TODO: calcolare prob storiche dal df_ev_scope (es. % Over 2.5)
    ev_rows = []
    for name, q, line in [
        ("Over 1.5", quota_ov15, 1.5),
        ("Over 2.5", quota_ov25, 2.5),
        ("Over 3.5", quota_ov35, 3.5),
        ("BTTS",  quota_btts, None),
    ]:
        # Probabilità storica grezza come esempio (puoi raffinare con filtri su squadra, label, etc.)
        if name == "BTTS":
            prob_hist = float(((df_ev_scope["Home Goal FT"] > 0) & (df_ev_scope["Away Goal FT"] > 0)).mean() * 100) if not df_ev_scope.empty else 0.0
        else:
            prob_hist = float(((df_ev_scope["Home Goal FT"] + df_ev_scope["Away Goal FT"]) > line).mean() * 100) if not df_ev_scope.empty else 0.0

        ev = round(q * (prob_hist / 100) - 1, 2)
        nota = "🟢 EV+" if ev > 0 else ("🔴 EV-" if ev < 0 else "⚪️ Neutro")
        ev_rows.append({
            "Mercato": name,
            "Quota": q,
            "Probabilità Storica": f"{prob_hist:.1f}%",
            "EV": ev,
            "Note": nota
        })

    st.dataframe(pd.DataFrame(ev_rows), use_container_width=True)
