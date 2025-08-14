# pre_match.py — Pre-Match pulito: usa SOLO i filtri globali dell’Hub
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Moduli locali
from utils import label_match
from squadre import compute_team_macro_stats, render_team_stats_tab

# Correct score: usa il modulo esistente (compatibile)
try:
    from correct_score_ev_sezione import run_correct_score_ev as _run_correct_score
except Exception:
    _run_correct_score = None

# Live: prova più nomi funzione per massima compatibilità
_run_live = None
try:
    from analisi_live_minuto import run_live_minuto_analysis as _run_live  # nome 1
except Exception:
    try:
        from analisi_live_minuto import run_live_minute_analysis as _run_live  # nome 2
    except Exception:
        _run_live = None


# ====== chiavi dei filtri GLOBALI impostati in app.py ======
GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"   # lista di stagioni (stringhe)


# --------------------------
# Utility di contesto
# --------------------------
def _apply_global_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Applica i filtri globali (Hub). In app.py il df è già filtrato,
    ma qui li ri-applichiamo in modo idempotente per sicurezza."""
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    seasons = st.session_state.get(GLOBAL_SEASONS_KEY)
    out = df.copy()
    if champ and "country" in out.columns:
        out = out[out["country"].astype(str) == str(champ)]
    if seasons and "Stagione" in out.columns:
        out = out[out["Stagione"].astype(str).isin([str(s) for s in seasons])]
    return out


def _context_badges(db_label: str):
    champ = st.session_state.get(GLOBAL_CHAMP_KEY)
    seasons = st.session_state.get(GLOBAL_SEASONS_KEY)
    txt_seas = ", ".join([str(s) for s in seasons]) if seasons else "tutte"
    st.markdown(
        f"""
        <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin:.25rem 0 1rem 0">
          <span style="border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6">
            🏆 <b>{champ or db_label}</b>
          </span>
          <span style="border:1px solid #e5e7eb;padding:.25rem .6rem;border-radius:999px;background:#f3f4f6">
            🗓️ <b>Stagioni:</b> {txt_seas}
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _download(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, filename, "text/csv")


# --------------------------
# Quote condivise in sessione
# --------------------------
SHARED_KEY = "PM_shared_odds"

def _init_shared_odds():
    st.session_state.setdefault(SHARED_KEY, {
        # 1X2
        "1_back": None, "X_back": None, "2_back": None,
        "1_lay":  None, "X_lay":  None, "2_lay":  None,
        # Over & BTTS
        "O05": None, "O15": None, "O25": None, "O35": None, "BTTS": None,
    })


# --------------------------
# 1X2 Back/Lay ROI (semplice storico)
# --------------------------
def _calc_1x2_rois(df: pd.DataFrame, commission: float = 0.045):
    """Ritorna ROI Back/Lay % per HOME/DRAW/AWAY sul campione."""
    if df.empty:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, 0

    # Assicura colonne gol (numeriche)
    hg = pd.to_numeric(df.get("Home Goal FT"), errors="coerce")
    ag = pd.to_numeric(df.get("Away Goal FT"), errors="coerce")
    mask = ~(hg.isna() | ag.isna())
    df = df.loc[mask].copy()
    if df.empty:
        zero = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
        return zero, zero, 0

    results = np.where(hg.values > ag.values, "HOME", np.where(hg.values < ag.values, "AWAY", "DRAW"))
    n = len(results)

    # Quote storiche, se presenti
    oh = pd.to_numeric(df.get("Odd home"), errors="coerce").fillna(2.0).values
    od = pd.to_numeric(df.get("Odd Draw"), errors="coerce").fillna(3.2).values
    oa = pd.to_numeric(df.get("Odd Away"), errors="coerce").fillna(3.2).values
    prices = {"HOME": oh, "DRAW": od, "AWAY": oa}

    back_profit = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}
    lay_profit  = {"HOME": 0.0, "DRAW": 0.0, "AWAY": 0.0}

    for i in range(n):
        res = results[i]
        for outc, p_arr in prices.items():
            p = max(float(p_arr[i]), 1.01)
            # back
            if res == outc:
                back_profit[outc] += (p - 1) * (1 - commission)
            else:
                back_profit[outc] -= 1
            # lay
            stake = 1.0 / (p - 1.0)
            if res != outc:
                lay_profit[outc] += stake
            else:
                lay_profit[outc] -= 1.0

    back_roi = {k: round((v / n) * 100, 2) for k, v in back_profit.items()}
    lay_roi  = {k: round((v / n) * 100, 2) for k, v in lay_profit.items()}
    return back_roi, lay_roi, n


# --------------------------
# ROI Over/BTTS (storico) con supporto quota manuale
# --------------------------
def _roi_over_btts(df: pd.DataFrame, market: str, line: float | None, manual_price: float | None, commission: float = 0.045):
    df = df.dropna(subset=["Home Goal FT", "Away Goal FT"])
    if df.empty:
        return {"Mercato": market, "Quota": manual_price, "Esiti %": "0.0%", "ROI Back %": "0.0%", "ROI Lay %": "0.0%", "N": 0}

    # Colonna quota, se c'è
    col_map = {
        "Over 0.5": ["cotao0", "Odd Over 0.5", "Over 0.5"],
        "Over 1.5": ["cotao1", "Odd Over 1.5", "Over 1.5"],
        "Over 2.5": ["cotao",  "Odd Over 2.5", "Over 2.5"],
        "Over 3.5": ["cotao3", "Odd Over 3.5", "Over 3.5"],
        "BTTS":     ["gg", "GG", "BTTS Yes", "Odd BTTS Yes"],
    }
    col = next((c for c in col_map[market] if c in df.columns), None)
    if col:
        odds = pd.to_numeric(df[col], errors="coerce")
    else:
        odds = pd.Series([np.nan] * len(df), index=df.index)

    if manual_price:
        odds = odds.fillna(manual_price).where(odds >= 1.01, manual_price)

    hit = 0; back_p = 0.0; lay_p = 0.0; tot = 0; qsum = 0.0; qn = 0

    for i, row in df.iterrows():
        o = float(odds.loc[i]) if i in odds.index else float("nan")
        if not (o and o >= 1.01 and np.isfinite(o)):
            continue
        hg = int(pd.to_numeric(row["Home Goal FT"], errors="coerce"))
        ag = int(pd.to_numeric(row["Away Goal FT"], errors="coerce"))
        g = hg + ag
        ok = (g > float(line)) if market != "BTTS" else ((hg > 0) and (ag > 0))

        tot += 1; qsum += o; qn += 1
        if ok:
            hit += 1; back_p += (o - 1) * (1 - commission); lay_p -= 1
        else:
            lay_p += 1 / (o - 1); back_p -= 1

    pct = round((hit / tot) * 100, 2) if tot else 0.0
    rb = round((back_p / tot) * 100, 2) if tot else 0.0
    rl = round((lay_p  / tot) * 100, 2) if tot else 0.0

    return {
        "Mercato": market,
        "Quota": round(qsum / qn, 2) if qn else manual_price,
        "Esiti %": f"{pct}%",
        "ROI Back %": f"{rb}%",
        "ROI Lay %": f"{rl}%",
        "N": tot
    }


# --------------------------
# ENTRY POINT
# --------------------------
def run_pre_match(df: pd.DataFrame, db_label: str = "Dataset"):
    st.title("🧮 Pre-Match – Analisi per Trader Professionisti")

    # Contesto globale (solo lettura) e dati
    df = _apply_global_filters(df)
    _context_badges(db_label)

    if df.empty:
        st.warning("Nessun dato disponibile per il contesto selezionato.")
        return

    # Etichette & dtypes
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # ---- Ricerca & assegnazione squadre (semplice) ----
    st.subheader("🔎 Ricerca rapida squadre (assegna come Casa/Ospite)")
    teams_all = sorted(set(df["Home"].dropna().astype(str)) | set(df["Away"].dropna().astype(str)))

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        query = st.text_input("Cerca squadra", key="pm_search")
        results = [t for t in teams_all if query.strip().lower() in t.lower()] if query else teams_all
        team_pick = st.selectbox("Risultati", results, index=0 if results else None, key="pm_pick")
    with col2:
        role = st.radio("Assegna a", ["Casa", "Ospite"], horizontal=True)
    with col3:
        if st.button("Assegna", use_container_width=True):
            if role == "Casa":
                st.session_state["PM_home_team"] = team_pick
            else:
                st.session_state["PM_away_team"] = team_pick
            st.success("Contesto squadre aggiornato.")

    # Selettori finali (con default da sessione)
    c1, cswap, c2 = st.columns([2, 0.6, 2])
    with c1:
        home_team = st.selectbox("Squadra Casa", options=teams_all,
                                 index=teams_all.index(st.session_state.get("PM_home_team", teams_all[0]))
                                 if teams_all else 0, key="PM_home_team_select")
    with cswap:
        st.write("")
        if st.button("🔁 Inverti", use_container_width=True):
            h = st.session_state.get("PM_home_team_select")
            a = st.session_state.get("PM_away_team_select")
            st.session_state["PM_home_team_select"] = a
            st.session_state["PM_away_team_select"] = h
            st.experimental_rerun()
    with c2:
        away_team = st.selectbox("Squadra Ospite", options=teams_all,
                                 index=teams_all.index(st.session_state.get("PM_away_team", teams_all[1 if len(teams_all) > 1 else 0]))
                                 if teams_all else 0, key="PM_away_team_select")

    if home_team == away_team:
        st.error("Casa e Ospite non possono essere uguali.")
        return

    # ---- Quote condivise ----
    _init_shared_odds()
    shared = st.session_state[SHARED_KEY]

    st.subheader("💹 Quote condivise")
    st.caption("Queste quote (1X2, Over x.5, BTTS) sono memorizzate in sessione e riutilizzate nelle altre sezioni.")

    b1, b2, b3 = st.columns(3)
    shared["1_back"] = b1.number_input("1 (Back)", min_value=1.01, max_value=1000.0, step=0.01,
                                       value=shared["1_back"], format="%.2f")
    shared["X_back"] = b2.number_input("X (Back)", min_value=1.01, max_value=1000.0, step=0.01,
                                       value=shared["X_back"], format="%.2f")
    shared["2_back"] = b3.number_input("2 (Back)", min_value=1.01, max_value=1000.0, step=0.01,
                                       value=shared["2_back"], format="%.2f")

    l1, l2, l3 = st.columns(3)
    shared["1_lay"] = l1.number_input("1 (Lay)", min_value=1.01, max_value=1000.0, step=0.01,
                                      value=shared["1_lay"], format="%.2f")
    shared["X_lay"] = l2.number_input("X (Lay)", min_value=1.01, max_value=1000.0, step=0.01,
                                      value=shared["X_lay"], format="%.2f")
    shared["2_lay"] = l3.number_input("2 (Lay)", min_value=1.01, max_value=1000.0, step=0.01,
                                      value=shared["2_lay"], format="%.2f")

    o1, o2, o3, o4, o5 = st.columns(5)
    shared["O05"]  = o1.number_input("Over 0.5", min_value=1.01, max_value=1000.0, step=0.01,
                                     value=shared["O05"], format="%.2f")
    shared["O15"]  = o2.number_input("Over 1.5", min_value=1.01, max_value=1000.0, step=0.01,
                                     value=shared["O15"], format="%.2f")
    shared["O25"]  = o3.number_input("Over 2.5", min_value=1.01, max_value=1000.0, step=0.01,
                                     value=shared["O25"], format="%.2f")
    shared["O35"]  = o4.number_input("Over 3.5", min_value=1.01, max_value=1000.0, step=0.01,
                                     value=shared["O35"], format="%.2f")
    shared["BTTS"] = o5.number_input("BTTS (GG)", min_value=1.01, max_value=1000.0, step=0.01,
                                     value=shared["BTTS"], format="%.2f")

    st.session_state[SHARED_KEY] = shared  # salva

    st.markdown("---")

    # ====== TABS ======
    t1, t2, t3, t4, t5 = st.tabs(["1X2", "ROI Over/BTTS", "EV Storico", "Statistiche Squadre", "Correct Score"])

    # --- TAB 1: 1X2 ---
    with t1:
        st.markdown("**Win% storiche e ROI Back/Lay** per campionato filtrato. "
                    "Le Win% sono calcolate dai risultati FT.")
        league_df = df.copy()
        # Win% complessive
        hg = pd.to_numeric(league_df.get("Home Goal FT"), errors="coerce")
        ag = pd.to_numeric(league_df.get("Away Goal FT"), errors="coerce")
        league_df = league_df.loc[~(hg.isna() | ag.isna())]
        if league_df.empty:
            st.info("Campione vuoto per 1X2.")
        else:
            wins_home = (league_df["Home Goal FT"] > league_df["Away Goal FT"]).mean() * 100
            wins_draw = (league_df["Home Goal FT"] == league_df["Away Goal FT"]).mean() * 100
            wins_away = (league_df["Home Goal FT"] < league_df["Away Goal FT"]).mean() * 100

            back_roi, lay_roi, n = _calc_1x2_rois(league_df)
            df_1x2 = pd.DataFrame([
                ["HOME", round(wins_home, 2), back_roi["HOME"], lay_roi["HOME"]],
                ["DRAW", round(wins_draw, 2), back_roi["DRAW"], lay_roi["DRAW"]],
                ["AWAY", round(wins_away, 2), back_roi["AWAY"], lay_roi["AWAY"]],
            ], columns=["Segno", "Win %", "Back ROI %", "Lay ROI %"])
            st.dataframe(df_1x2, use_container_width=True, hide_index=True)
            _download(df_1x2, "1x2_roi.csv", "⬇️ Scarica 1X2 CSV")

    # --- TAB 2: ROI Over/BTTS ---
    with t2:
        st.markdown("**ROI storico** per Over/BTTS sul campionato filtrato. "
                    "Se le quote non sono nel dataset, uso le **quote inserite sopra**.")
        q = st.session_state[SHARED_KEY]
        table = [
            _roi_over_btts(df, "Over 0.5", 0.5, q["O05"]),
            _roi_over_btts(df, "Over 1.5", 1.5, q["O15"]),
            _roi_over_btts(df, "Over 2.5", 2.5, q["O25"]),
            _roi_over_btts(df, "Over 3.5", 3.5, q["O35"]),
            _roi_over_btts(df, "BTTS", None,  q["BTTS"]),
        ]
        df_roi = pd.DataFrame(table)
        st.dataframe(df_roi, use_container_width=True, hide_index=True)
        _download(df_roi, "roi_over_btts.csv", "⬇️ Scarica ROI CSV")

    # --- TAB 3: EV Storico (Home@Casa, Away@Trasferta, Blend, H2H) ---
    with t3:
        st.markdown("**EV = quota × p − 1**. Le probabilità p sono stimate su:"
                    " Home@Casa, Away@Trasferta, Blend (media), Head-to-Head.")
        q = st.session_state[SHARED_KEY]

        def _p_over(df_scope: pd.DataFrame, line: float | None, is_btts: bool = False) -> float:
            if df_scope.empty:
                return 0.0
            hg = pd.to_numeric(df_scope["Home Goal FT"], errors="coerce").fillna(0)
            ag = pd.to_numeric(df_scope["Away Goal FT"], errors="coerce").fillna(0)
            if is_btts:
                return round(((hg > 0) & (ag > 0)).mean() * 100, 2)
            return round(((hg + ag) > float(line)).mean() * 100, 2)

        home_ctx = df[df["Home"].astype(str) == str(home_team)]
        away_ctx = df[df["Away"].astype(str) == str(away_team)]
        h2h = df[
            ((df["Home"].astype(str) == str(home_team)) & (df["Away"].astype(str) == str(away_team))) |
            ((df["Home"].astype(str) == str(away_team)) & (df["Away"].astype(str) == str(home_team)))
        ]

        rows = []
        for name, line, price in [("Over 0.5", 0.5, q["O05"]), ("Over 1.5", 1.5, q["O15"]),
                                  ("Over 2.5", 2.5, q["O25"]), ("Over 3.5", 3.5, q["O35"]),
                                  ("BTTS", None, q["BTTS"])]:
            is_btts = (name == "BTTS")
            ph = _p_over(home_ctx, line, is_btts)
            pa = _p_over(away_ctx, line, is_btts)
            pb = round((ph + pa) / 2, 2) if (ph or pa) else 0.0
            ph2h = _p_over(h2h, line, is_btts)

            rows.append({
                "Mercato": name, "Quota": price,
                f"{home_team} @Casa %": ph, f"{away_team} @Trasferta %": pa,
                "Blended %": pb, "H2H %": ph2h,
                "EV Blended": round(price * (pb/100.0) - 1, 2),
                "EV H2H":     round(price * (ph2h/100.0) - 1, 2),
                "N Home": len(home_ctx), "N Away": len(away_ctx), "N H2H": len(h2h)
            })

        df_ev = pd.DataFrame(rows)
        st.dataframe(df_ev, use_container_width=True, hide_index=True)
        _download(df_ev, "ev_storico.csv", "⬇️ Scarica EV CSV")

    # --- TAB 4: Statistiche Squadre (modulo esistente) ---
    with t4:
        render_team_stats_tab(
            df,                      # campionato filtrato
            st.session_state.get(GLOBAL_CHAMP_KEY) or db_label,
            home_team, away_team
        )

    # --- TAB 5: Correct Score (modulo esistente) ---
    with t5:
        if _run_correct_score is None:
            st.info("Modulo 'Correct Score' non disponibile nel progetto.")
        else:
            st.subheader("Correct Score — EV su punteggi corretti")
            _run_correct_score(df, st.session_state.get(GLOBAL_CHAMP_KEY) or db_label)

            st.caption("Il pannello usa il campione filtrato corrente. "
                       "Se vuoi, possiamo aggiungere qui un prefill Home/Away e un collegamento alle quote.")
