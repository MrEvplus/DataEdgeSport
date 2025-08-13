# analisi_live_minuto.py — v2.0 ProTrading Live
# Mantiene struttura esistente, aggiunge KPI, shrinkage, EV, CS dinamico e tabs pulite.

import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from utils import label_match, extract_minutes  # esistenti

# -----------------------------
# --------- STYLES ------------
# -----------------------------
def color_stat_rows(row):
    styles = []
    for col, val in row.items():
        if col == "Matches" and row.name == "Matches":
            styles.append("font-weight: bold; color: black; background-color: transparent")
        elif isinstance(val, float) and ("%" in col or row.name.endswith("%") or col == "%"):
            styles.append(color_pct(val))
        else:
            styles.append("")
    return styles

def color_pct(val):
    try:
        v = float(val)
    except:
        return ""
    if v < 50:
        return "background-color: #ffd6d6; color: #000;"   # rosso chiaro
    elif v < 70:
        return "background-color: #fff5b5; color: #000;"   # giallo
    else:
        return "background-color: #c9f7c5; color: #000;"   # verde

def sample_badge(n: int) -> str:
    if n < 30:  return "🔴 Campione piccolo"
    if n < 100: return "🟡 Campione medio"
    return "🟢 Campione robusto"

# -----------------------------
# --------- UTILS -------------
# -----------------------------
def safe_parse_score(txt: str):
    """Accetta '1-1', '1 : 1', '1–1' ecc."""
    if not isinstance(txt, str):
        return None
    cleaned = txt.replace(" ", "").replace(":", "-").replace("–", "-").replace("—", "-")
    parts = cleaned.split("-")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except:
        return None

def wilson_ci(successes: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2/(2*n)) / denom
    half = z * math.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    low, high = max(0.0, center - half), min(1.0, center + half)
    return (p, low, high)

def shrink_pct(success: int, total: int, prior: float = 0.5, strength: float = 20.0) -> float:
    """Jeffreys/Beta prior: α=prior*strength, β=(1-prior)*strength."""
    if total <= 0:
        return prior
    a = strength * prior + success
    b = strength * (1 - prior) + (total - success)
    return a / (a + b)

def ev_back(p: float, price: float, comm: float = 0.045) -> float:
    """EV per 1 unità di stake."""
    return p * (price - 1) * (1 - comm) - (1 - p)

def ev_lay(p: float, price: float) -> float:
    """EV per liability=1 (utile come indice)."""
    if price <= 1.0:
        return -p
    return (1 - p) * (1 / (price - 1)) - p

def parse_data_to_datetime(s):
    """Prova a convertire 'Data' a datetime tz-naive; fallback stringa invariata."""
    try:
        return pd.to_datetime(s, errors="coerce")
    except:
        return pd.NaT

def exp_weights_by_recency(dt_series: pd.Series, half_life_days: float = 180.0) -> np.ndarray:
    """Pesa più gli eventi recenti (mezzo‑vita di default 180gg)."""
    if dt_series.isna().all():
        # se non ci sono date, ritorna pesi uniformi
        return np.ones(len(dt_series))
    now = pd.Timestamp.utcnow().tz_localize(None)
    age_days = (now - dt_series.dt.tz_localize(None)).dt.days.fillna(0).clip(lower=0)
    lam = math.log(2) / max(1e-9, half_life_days)
    w = np.exp(-lam * age_days)
    # evita tutti zero
    if w.sum() == 0:
        w = np.ones_like(w)
    return w

# -----------------------------
# ------- CORE STATS ----------
# -----------------------------
@st.cache_data(show_spinner=False)
def compute_post_minute_stats(df: pd.DataFrame, current_min: int, label: str):
    """(Evoluzione) Statistiche per bande di tempo dopo il minuto corrente."""
    tf_bands = [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 90)]
    tf_labels = [f"{a}-{b}" for a, b in tf_bands]
    data = {lbl: {"GF": 0, "GS": 0, "Match_1+": 0, "Match_2+": 0, "TotalMatch": 0} for lbl in tf_labels}

    for _, row in df.iterrows():
        mh = extract_minutes(pd.Series([row.get("minuti goal segnato home", "")]))
        ma = extract_minutes(pd.Series([row.get("minuti goal segnato away", "")]))
        all_post = [(m, "H") for m in mh if m > current_min] + [(m, "A") for m in ma if m > current_min]

        goals_by_tf = {lbl: {"GF": 0, "GS": 0} for lbl in tf_labels}
        for m, side in all_post:
            for lbl, (a, b) in zip(tf_labels, tf_bands):
                if a < m <= b:
                    if label.startswith("H_"):
                        if side == "H": goals_by_tf[lbl]["GF"] += 1
                        else:           goals_by_tf[lbl]["GS"] += 1
                    elif label.startswith("A_"):
                        if side == "A": goals_by_tf[lbl]["GF"] += 1
                        else:           goals_by_tf[lbl]["GS"] += 1
                    else:
                        if side == "H": goals_by_tf[lbl]["GF"] += 1
                        else:           goals_by_tf[lbl]["GS"] += 1
                    break

        for lbl in tf_labels:
            gf = goals_by_tf[lbl]["GF"]
            gs = goals_by_tf[lbl]["GS"]
            total = gf + gs
            if total > 0:  data[lbl]["Match_1+"] += 1
            if total >= 2: data[lbl]["Match_2+"] += 1
            data[lbl]["GF"] += gf
            data[lbl]["GS"] += gs
            data[lbl]["TotalMatch"] += 1

    df_stats = pd.DataFrame([
        {
            "Intervallo": lbl,
            "GF": v["GF"],
            "GS": v["GS"],
            "% 1+ Goal": round((v["Match_1+"] / v["TotalMatch"]) * 100, 2) if v["TotalMatch"] > 0 else 0.0,
            "% 2+ Goal": round((v["Match_2+"] / v["TotalMatch"]) * 100, 2) if v["TotalMatch"] > 0 else 0.0,
        }
        for lbl, v in data.items()
    ])
    return df_stats

def prob_goal_next_window(df: pd.DataFrame, current_min: int, window: int = 10) -> tuple[float,int,int]:
    """Prob. che accada almeno 1 gol nei prossimi 'window' minuti (league matched)."""
    succ = 0
    n = 0
    for _, r in df.iterrows():
        mh = extract_minutes(pd.Series([r.get("minuti goal segnato home", "")]))
        ma = extract_minutes(pd.Series([r.get("minuti goal segnato away", "")]))
        # gol nel (current_min, current_min+window]
        any_goal = any(current_min < m <= current_min + window for m in mh + ma)
        succ += 1 if any_goal else 0
        n += 1
    p, lo, hi = wilson_ci(succ, n)
    return p, succ, n

def estimate_remaining_lambdas(df: pd.DataFrame, current_min: int, label_side_home: bool):
    """Stima λ residui (Poisson) per Home/Away nel tempo rimanente, usando GF/GS post‑minuto."""
    tf = compute_post_minute_stats(df, current_min, "H_" if label_side_home else "A_")
    # somma gol fatti/subiti in tutte le bande future e scala per match considerati
    # Nota: compute_post_minute_stats già conta 'TotalMatch' implicitamente come righe iterate.
    # Qui stimiamo lambda per match = (totale GF)/(numero di partite)
    # Per robustezza, se campione=0 ritorna 0.2 (prior tenue)
    total_gf = tf["GF"].sum()
    total_gs = tf["GS"].sum()
    n = len(df) if len(df) > 0 else 1
    lam_for = total_gf / max(1, n)
    lam_against = total_gs / max(1, n)
    # shrink leggero verso 0.6 (circa media gol complessiva residua)
    lam_for = 0.5 * lam_for + 0.5 * 0.6
    lam_against = 0.5 * lam_against + 0.5 * 0.6
    return max(0.01, lam_for), max(0.01, lam_against)

def poisson_pmf(k: int, lam: float) -> float:
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0

def final_cs_distribution(lam_home_add: float, lam_away_add: float, cur_h: int, cur_a: int, max_goals_delta: int = 6):
    """Distribuzione di Correct Score finale con Poisson indipendenti sui gol rimanenti."""
    probs = {}
    for x in range(0, max_goals_delta + 1):
        for y in range(0, max_goals_delta + 1):
            p = poisson_pmf(x, lam_home_add) * poisson_pmf(y, lam_away_add)
            cs = f"{cur_h + x}-{cur_a + y}"
            probs[cs] = probs.get(cs, 0.0) + p
    # normalizza (tagliando la coda)
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

# -----------------------------
# --------- APP ---------------
# -----------------------------
def run_live_minute_analysis(df: pd.DataFrame):
    st.set_page_config(page_title="Analisi Live Minuto", layout="wide")
    st.title("⏱️ Analisi Live — Cosa succede da questo minuto?")

    # ---------------- Controls top ----------------
    col0, col1 = st.columns([1, 2])
    with col0:
        # Campionato corrente: lo prendo da sessione se presente, altrimenti primo del df
        champ_default = st.session_state.get("campionato_corrente", str(df["country"].iloc[0]))
        champ = st.selectbox("🏆 Campionato", sorted(df["country"].dropna().astype(str).unique()), index=sorted(df["country"].dropna().astype(str).unique()).index(champ_default) if champ_default in df["country"].astype(str).unique() else 0, key="champ_live")

    with col1:
        c1, c2 = st.columns(2)
        with c1:
            home_team = st.selectbox("🏠 Squadra in casa", sorted(df["Home"].dropna().unique()), key="home_live")
        with c2:
            away_team = st.selectbox("🚪 Squadra in trasferta", sorted(df["Away"].dropna().unique()), key="away_live")

    c_odds = st.columns(3)
    with c_odds[0]:
        odd_home = st.number_input("📈 Quota Home", 1.01, 50.0, 2.00, step=0.01, key="odd_h")
    with c_odds[1]:
        odd_draw = st.number_input("⚖️ Quota Pareggio", 1.01, 50.0, 3.20, step=0.01, key="odd_d")
    with c_odds[2]:
        odd_away = st.number_input("📉 Quota Away", 1.01, 50.0, 3.80, step=0.01, key="odd_a")

    c_live = st.columns([2,1,1,1])
    with c_live[0]:
        current_min = st.slider("⏲️ Minuto attuale", 1, 120, 45, key="minlive")
    with c_live[1]:
        live_score_txt = st.text_input("📟 Risultato live", "1-1", key="scorelive")
    with c_live[2]:
        use_recent_weight = st.toggle("🎚️ Pesa forma recente", value=True, help="Half-life 180gg")
    with c_live[3]:
        momentum_boost = st.slider("📈 Boost momentum (%)", 0, 20, 0, help="Aggiunge fino a +20% (cap) su P(gol breve)")

    parsed = safe_parse_score(live_score_txt)
    if not parsed:
        st.error("⚠️ Formato risultato non valido. Usa ad esempio: 1-1, 0-0, 2-1")
        return
    live_h, live_a = parsed
    cur_score_str = f"{live_h}-{live_a}"

    # ---------------- Label & dataset ----------------
    st.divider()
    label_live = label_match({"Odd home": odd_home, "Odd Away": odd_away})
    st.markdown(f"🔖 **Label**: `{label_live}`")

    df = df.copy()
    df["Label"] = df.apply(label_match, axis=1)
    df["Data_dt"] = df["Data"].apply(parse_data_to_datetime)

    # filtro campionato + label
    df_league = df[(df["country"] == champ) & (df["Label"] == label_live)]

    # stato-partita (favorito avanti/pari/indietro) calcolato con label + punteggio live
    favorito_home = label_live.startswith("H_")
    stato = "pari"
    if favorito_home and (live_h > live_a):      stato = "fav_avanti"
    elif favorito_home and (live_h < live_a):    stato = "fav_sotto"
    elif (not favorito_home) and (live_a > live_h): stato = "fav_avanti"
    elif (not favorito_home) and (live_a < live_h): stato = "fav_sotto"
    st.caption(f"📌 Stato-partita: **{stato}** (favorito: {'Home' if favorito_home else 'Away'})")

    # match che replicano score al minuto corrente
    matched = []
    for _, r in df_league.iterrows():
        mh = extract_minutes(pd.Series([r.get("minuti goal segnato home", "")]))
        ma = extract_minutes(pd.Series([r.get("minuti goal segnato away", "")]))
        gh = sum(m <= current_min for m in mh)
        ga = sum(m <= current_min for m in ma)
        if gh == live_h and ga == live_a:
            matched.append(r)
    df_matched = pd.DataFrame(matched)

    # Team di riferimento dal punto di vista del favorito/etichetta
    team_target = home_team if favorito_home else away_team

    # subset team
    matched_team = []
    for _, r in df_league.iterrows():
        if r["Home"] != team_target and r["Away"] != team_target:
            continue
        mh = extract_minutes(pd.Series([r.get("minuti goal segnato home", "")]))
        ma = extract_minutes(pd.Series([r.get("minuti goal segnato away", "")]))
        gh = sum(m <= current_min for m in mh)
        ga = sum(m <= current_min for m in ma)
        if gh == live_h and ga == live_a:
            matched_team.append(r)
    df_team = pd.DataFrame(matched_team)

    st.success(f"✅ {len(df_matched)} partite trovate a {cur_score_str} al minuto {current_min}′ | Team focus: {team_target}")

    # Espansore elenco partite
    with st.expander("📑 Partite del campionato considerate per l'analisi"):
        if not df_matched.empty:
            cols_show = ["Stagione","Data","Home","Away","Home Goal FT","Away Goal FT","minuti goal segnato home","minuti goal segnato away"]
            cols_show = [c for c in cols_show if c in df_matched.columns]
            st.dataframe(
                df_matched[cols_show]
                .sort_values(["Stagione","Data"], ascending=[False, False])
                .reset_index(drop=True),
                use_container_width=True
            )
        else:
            st.info("Nessuna partita storica trovata con lo stesso stato.")

    # ---------------- KPI TOP ----------------
    # P(gol prossimi 10')
    p10_raw, succ10, n10 = prob_goal_next_window(df_matched, current_min, window=10)
    # Bias campionato corrente come prior per shrinkage: stimo % goal in 10' su TUTTE le partite della lega (fallback)
    league_p10_prior, prior_succ, prior_n = prob_goal_next_window(df_league, current_min, window=10) if len(df_league) else (0.35, 0, 0)
    p10_shrunk = shrink_pct(succ10, n10, prior=float(league_p10_prior), strength=20.0)
    # Momentum boost (cap a 20%)
    p10_boosted = min(1.0, p10_shrunk * (1.0 + momentum_boost/100.0))

    # EV Over 2.5 esempio (usiamo probabilità post-minuto di raggiungere >2.5 gol)
    tot_gf = (df_matched["Home Goal FT"] + df_matched["Away Goal FT"]) if not df_matched.empty else pd.Series([], dtype=float)
    # successi: match che hanno chiuso con >2.5
    succ_over25 = int((tot_gf > 2.5).sum()) if len(tot_gf) else 0
    n_over25 = len(tot_gf)
    league_over25_prior = (df_league["Home Goal FT"] + df_league["Away Goal FT"] > 2.5).mean() if len(df_league) else 0.5
    p_over25 = shrink_pct(succ_over25, n_over25, prior=float(league_over25_prior), strength=20.0)
    # EV back (richiede quota live Over 2.5 inserita a mano)
    with st.expander("⚙️ Imposta (facoltativo) quote mercato per EV"):
        c0, c1, c2, c3, c4 = st.columns(5)
        with c0:
            q_over05 = st.number_input("Quota Over 0.5", 1.01, 50.0, 1.30, step=0.01)
        with c1:
            q_over15 = st.number_input("Quota Over 1.5", 1.01, 50.0, 1.65, step=0.01)
        with c2:
            q_over25 = st.number_input("Quota Over 2.5", 1.01, 50.0, 2.40, step=0.01)
        with c3:
            q_btts   = st.number_input("Quota BTTS", 1.01, 50.0, 2.10, step=0.01)
        with c4:
            q_draw   = st.number_input("Quota Pareggio (Lay)", 1.01, 50.0, float(odd_draw), step=0.01)

    colK1, colK2, colK3, colK4 = st.columns(4)
    colK1.metric("Sample Campionato", len(df_matched), help=sample_badge(len(df_matched)))
    colK2.metric("Sample Squadra", len(df_team), help=sample_badge(len(df_team)))
    colK3.metric("P(gol prossimi 10')", f"{p10_boosted*100:.1f}%", delta=f"{(p10_boosted - league_p10_prior)*100:.1f}pp")
    colK4.metric("P(>2.5 FT)", f"{p_over25*100:.1f}%", help="Shrinkage vs bias campionato")

    st.caption(f"{sample_badge(len(df_matched))} | {sample_badge(len(df_team))}")

    # ---------------- Layout Destro/Sinistro con Tabs ----------------
    left, right = st.columns(2)

    # ---------- Left: CAMPIONATO ----------
    with left:
        st.subheader("📊 Campionato (stesso label & stato)")
        tabsL = st.tabs(["Esiti", "Over / EV", "Post‑minuto", "CS / Hedge"])

        # --- Esiti
        with tabsL[0]:
            if len(df_matched):
                home_w = (df_matched["Home Goal FT"] > df_matched["Away Goal FT"]).mean()
                draw   = (df_matched["Home Goal FT"] == df_matched["Away Goal FT"]).mean()
                away_w = (df_matched["Home Goal FT"] < df_matched["Away Goal FT"]).mean()
                # shrink verso bias di lega
                league_home = (df_league["Home Goal FT"] > df_league["Away Goal FT"]).mean() if len(df_league) else 0.45
                league_draw = (df_league["Home Goal FT"] == df_league["Away Goal FT"]).mean() if len(df_league) else 0.27
                league_away = (df_league["Home Goal FT"] < df_league["Away Goal FT"]).mean() if len(df_league) else 0.28

                home_w_s = shrink_pct(int(home_w*len(df_matched)), len(df_matched), prior=league_home, strength=20.0)
                draw_s   = shrink_pct(int(draw*len(df_matched)),   len(df_matched), prior=league_draw, strength=20.0)
                away_w_s = shrink_pct(int(away_w*len(df_matched)), len(df_matched), prior=league_away, strength=20.0)

                df_league_stats = pd.DataFrame(
                    {"Campionato": [len(df_matched), home_w_s*100, draw_s*100, away_w_s*100]},
                    index=["Matches", "Home %", "Draw %", "Away %"]
                )
                st.dataframe(df_league_stats.style.format("{:.2f}").apply(color_stat_rows, axis=1), use_container_width=True)
            else:
                st.info("Nessun match nel campione per calcolare gli esiti.")

        # --- Over / EV
        with tabsL[1]:
            if len(df_matched):
                # calcolo pesato opzionale
                if use_recent_weight:
                    w = exp_weights_by_recency(df_matched["Data_dt"])
                else:
                    w = np.ones(len(df_matched))
                tot_gf = (df_matched["Home Goal FT"] + df_matched["Away Goal FT"]).values
                cur_sum = live_h + live_a
                # prob > X.5 è prob che (FT goals) - cur_sum > X.5 -> FT > cur_sum + X.5
                # stimiamo p_k per threshold [0.5,1.5,2.5,3.5,4.5]
                rows = []
                thresholds = [0.5,1.5,2.5,3.5,4.5]
                quotes = [q_over05, q_over15, q_over25, None, None]  # puoi aggiungere input per 3.5/4.5 se vuoi
                league_bias = [(df_league["Home Goal FT"] + df_league["Away Goal FT"] > (cur_sum + t)).mean() if len(df_league) else 0.5 for t in thresholds]
                for i, t in enumerate(thresholds):
                    succ = ((tot_gf > (cur_sum + t))*w).sum()
                    n = w.sum()
                    p_raw = succ / n if n>0 else 0.0
                    p = shrink_pct(int(round(p_raw*n)), int(round(n)), prior=float(league_bias[i]), strength=20.0)
                    q = quotes[i]
                    ev_b = ev_l = None
                    if q:
                        ev_b = ev_back(p, q, comm=0.045)
                        ev_l = ev_lay(p, q)
                    rows.append({"Mercato": f"Over {t:.1f}".replace(".0",""), "P stimata": p*100, "Quota": q, "EV Back": ev_b, "EV Lay": ev_l})
                df_over = pd.DataFrame(rows)
                st.dataframe(df_over.style.format({"P stimata":"{:.2f}%", "Quota":"{:.2f}", "EV Back":"{:.3f}", "EV Lay":"{:.3f}"}), use_container_width=True)
            else:
                st.info("Nessun match per calcolare Over/EV.")

        # --- Post‑minuto
        with tabsL[2]:
            if len(df_matched):
                df_tf_league = compute_post_minute_stats(df_matched, current_min, label_live)
                st.dataframe(df_tf_league.style.apply(color_stat_rows, axis=1), use_container_width=True)
                st.caption(f"Qualità minuti goal: {'Alta' if ('minuti goal segnato home' in df_matched.columns and df_matched['minuti goal segnato home'].notna().mean()>0.8) else 'Variabile'}")
            else:
                st.info("Nessun match per calcolare le bande post‑minuto.")

        # --- CS / Hedge
        with tabsL[3]:
            if len(df_matched):
                lam_for, lam_against = estimate_remaining_lambdas(df_matched, current_min, favorito_home)
                # distribuzione CS finale
                top_cs = final_cs_distribution(
                    lam_home_add = lam_for if favorito_home else lam_against,
                    lam_away_add = lam_against if favorito_home else lam_for,
                    cur_h = live_h, cur_a = live_a, max_goals_delta=6
                )[:6]
                st.write("**Top Correct Score (probabilità)**")
                st.table(pd.DataFrame([{"CS": k, "Prob %": v*100} for k, v in top_cs]).style.format({"Prob %":"{:.2f}"}))

                st.markdown("**Valuta coperture (quote inserite manualmente)**")
                ccs1, ccs2, ccs3 = st.columns(3)
                with ccs1:
                    cs1 = st.text_input("CS #1", value=top_cs[0][0] if top_cs else "1-1")
                    q_cs1 = st.number_input("Quota CS #1", 1.01, 200.0, 6.0, step=0.01)
                with ccs2:
                    cs2 = st.text_input("CS #2", value=top_cs[1][0] if len(top_cs)>1 else "2-1")
                    q_cs2 = st.number_input("Quota CS #2", 1.01, 200.0, 9.0, step=0.01)
                with ccs3:
                    cs3 = st.text_input("CS #3", value=top_cs[2][0] if len(top_cs)>2 else "1-2")
                    q_cs3 = st.number_input("Quota CS #3", 1.01, 200.0, 10.0, step=0.01)

                def prob_from_list(target, pairs):
                    for k, v in pairs:
                        if k == target: return v
                    return 0.0

                rows_cs = []
                for cs, q in [(cs1, q_cs1),(cs2, q_cs2),(cs3, q_cs3)]:
                    p = prob_from_list(cs, top_cs)
                    rows_cs.append({"CS": cs, "Prob %": p*100, "Quota": q, "EV Back": ev_back(p, q, 0.045)})
                st.dataframe(pd.DataFrame(rows_cs).style.format({"Prob %":"{:.2f}","Quota":"{:.2f}","EV Back":"{:.3f}"}), use_container_width=True)
            else:
                st.info("Nessun match per stimare Correct Score.")

    # ---------- Right: SQUADRA ----------
    with right:
        st.subheader(f"📊 Squadra — {team_target}")
        tabsR = st.tabs(["Esiti", "Over / EV", "Post‑minuto", "CS / Hedge"])

        # --- Esiti
        with tabsR[0]:
            if len(df_team) and ("Home Goal FT" in df_team.columns) and ("Away Goal FT" in df_team.columns):
                if favorito_home:
                    win  = (df_team["Home Goal FT"] > df_team["Away Goal FT"]).mean()
                    draw = (df_team["Home Goal FT"] == df_team["Away Goal FT"]).mean()
                    lose = (df_team["Home Goal FT"] < df_team["Away Goal FT"]).mean()
                else:
                    win  = (df_team["Away Goal FT"] > df_team["Home Goal FT"]).mean()
                    draw = (df_team["Away Goal FT"] == df_team["Home Goal FT"]).mean()
                    lose = (df_team["Away Goal FT"] < df_team["Home Goal FT"]).mean()
                # shrink vs bias squadra dentro la lega (fallback lega)
                league_win = (df_league["Home Goal FT"] > df_league["Away Goal FT"]).mean() if len(df_league) else 0.45
                league_draw= (df_league["Home Goal FT"] == df_league["Away Goal FT"]).mean() if len(df_league) else 0.27
                league_lose= (df_league["Home Goal FT"] < df_league["Away Goal FT"]).mean() if len(df_league) else 0.28

                nT = len(df_team)
                win_s  = shrink_pct(int(win*nT),  nT, prior=league_win,  strength=15.0)
                draw_s = shrink_pct(int(draw*nT), nT, prior=league_draw, strength=15.0)
                lose_s = shrink_pct(int(lose*nT), nT, prior=league_lose, strength=15.0)

                df_team_stats = pd.DataFrame(
                    {team_target: [nT, win_s*100, draw_s*100, lose_s*100]},
                    index=["Matches","Win %","Draw %","Lose %"]
                )
                st.dataframe(df_team_stats.style.format("{:.2f}").apply(color_stat_rows, axis=1), use_container_width=True)
            else:
                st.warning(f"Dati insufficienti per la squadra {team_target}.")

        # --- Over / EV
        with tabsR[1]:
            if len(df_team) and {"Home Goal FT","Away Goal FT"}.issubset(df_team.columns):
                if use_recent_weight:
                    wT = exp_weights_by_recency(df_team["Data_dt"])
                else:
                    wT = np.ones(len(df_team))
                tot_gf_T = (df_team["Home Goal FT"] + df_team["Away Goal FT"]).values
                cur_sum = live_h + live_a

                rowsT = []
                thresholds = [0.5,1.5,2.5,3.5,4.5]
                quotesT = [q_over05, q_over15, q_over25, None, None]
                league_bias = [(df_league["Home Goal FT"] + df_league["Away Goal FT"] > (cur_sum + t)).mean() if len(df_league) else 0.5 for t in thresholds]
                for i, t in enumerate(thresholds):
                    succ = ((tot_gf_T > (cur_sum + t))*wT).sum()
                    n = wT.sum()
                    p_raw = succ / n if n>0 else 0.0
                    p = shrink_pct(int(round(p_raw*n)), int(round(n)), prior=float(league_bias[i]), strength=15.0)
                    q = quotesT[i]
                    ev_b = ev_l = None
                    if q:
                        ev_b = ev_back(p, q, comm=0.045)
                        ev_l = ev_lay(p, q)
                    rowsT.append({"Mercato": f"Over {t:.1f}".replace(".0",""), "P stimata": p*100, "Quota": q, "EV Back": ev_b, "EV Lay": ev_l})
                st.dataframe(pd.DataFrame(rowsT).style.format({"P stimata":"{:.2f}%", "Quota":"{:.2f}", "EV Back":"{:.3f}", "EV Lay":"{:.3f}"}), use_container_width=True)
            else:
                st.info("Nessun dato sufficiente per Over/EV squadra.")

        # --- Post‑minuto
        with tabsR[2]:
            if len(df_team) and {"Home Goal FT","Away Goal FT"}.issubset(df_team.columns):
                df_tf_team = compute_post_minute_stats(df_team, current_min, label_live)
                st.dataframe(df_tf_team.style.apply(color_stat_rows, axis=1), use_container_width=True)
            else:
                st.info("Nessun dato per bande post‑minuto della squadra.")

        # --- CS / Hedge
        with tabsR[3]:
            if len(df_team):
                lam_for_T, lam_against_T = estimate_remaining_lambdas(df_team, current_min, favorito_home)
                top_cs_T = final_cs_distribution(
                    lam_home_add = lam_for_T if favorito_home else lam_against_T,
                    lam_away_add = lam_against_T if favorito_home else lam_for_T,
                    cur_h = live_h, cur_a = live_a, max_goals_delta=6
                )[:6]
                st.write("**Top Correct Score (squadra focus)**")
                st.table(pd.DataFrame([{"CS": k, "Prob %": v*100} for k, v in top_cs_T]).style.format({"Prob %":"{:.2f}"}))
            else:
                st.info("Nessun match squadra per stimare CS.")

    st.divider()
    # ---------------- Piano operativo suggerito (stile IVO/Rebelo) ----------------
    # semplice regola: se EV back Over 2.5 > 0 e P>40% e campione ok -> suggerisci
    ev_over25_back = None
    if q_over25:
        ev_over25_back = ev_back(p_over25, q_over25, 0.045)
    suggestion = []
    if (ev_over25_back is not None) and (ev_over25_back > 0) and (p_over25 > 0.40) and (len(df_matched) >= 30):
        # Kelly 1/2
        edge = (q_over25 * p_over25 - (1 - p_over25)) / (q_over25 - 1)
        kelly = max(0.0, min(1.0, edge))
        kelly_half = 0.5 * kelly
        suggestion.append(f"🟢 **Ingresso Over 2.5** a quota ~{q_over25:.2f} | EV={ev_over25_back:.3f} | Stake=½ Kelly ≈ {kelly_half*100:.1f}% bankroll")
        suggestion.append("Uscita: cashout in profitto al primo gol; se no‑goal entro 10′ rivaluta con P(gol 10′) aggiornata.")
    if favorito_home and stato == "fav_sotto" and (current_min <= 75):
        # idea Lay X quando favorito in svantaggio non troppo tardi
        # P(no goal 10') ~ (1 - p10_boosted); se goal, pareggio sale ➜ profit sul Lay? dipende minuto, ma lo proponiamo come edge.
        suggestion.append("🟡 **Idea Lay Pareggio**: favorito sotto; target cashout su goal nei prossimi 10′. Se niente gol, chiusura ridotta per limitare drift.")
    if suggestion:
        st.markdown("### 🧭 Piano Operativo Suggerito")
        st.write("\n\n".join(suggestion))
    else:
        st.markdown("### 🧭 Piano Operativo Suggerito")
        st.info("Nessuna opportunità EV+ forte. **NO BET** finché il contesto non migliora.")

# NOTE:
# - File di partenza e struttura originale mantenuti, con estensioni secondo specifica. :contentReference[oaicite:1]{index=1}
# - Collega poi i KPI/outputs su pre_match come da step successivo.
