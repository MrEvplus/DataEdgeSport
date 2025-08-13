# analisi_live_minuto.py — v3.0 Live 1X2 + Over/BTTS + EV Advisor (AI scoring)
# Mantiene compatibilità con la tua app e degrada se moduli esterni non sono disponibili.

import math
import numpy as np
import pandas as pd
import streamlit as st

from utils import label_match, extract_minutes

# =============== STYLES ===============
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
        return "background-color: #ffd6d6; color: #222;"
    elif v < 70:
        return "background-color: #fff6bf; color: #222;"
    else:
        return "background-color: #c8f7c5; color: #222;"

def sample_badge(n: int) -> str:
    if n < 30:  return "🔴 Campione piccolo"
    if n < 100: return "🟡 Campione medio"
    return "🟢 Campione robusto"

# =============== UTILS ===============
def safe_parse_score(txt: str):
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

def _goals_up_to(series, minute):
    mins = extract_minutes(pd.Series([series if isinstance(series, str) else str(series or "")]))
    return sum(m <= minute for m in mins)

def _matches_matching_state(df, minute, live_h, live_a):
    rows = []
    for _, r in df.iterrows():
        gh = _goals_up_to(r.get("minuti goal segnato home", ""), minute)
        ga = _goals_up_to(r.get("minuti goal segnato away", ""), minute)
        if gh == live_h and ga == live_a:
            rows.append(r)
    return pd.DataFrame(rows)

def _result_probs(df):
    if df is None or df.empty:
        return (1/3, 1/3, 1/3)
    h = (df["Home Goal FT"] > df["Away Goal FT"]).mean()
    d = (df["Home Goal FT"] == df["Away Goal FT"]).mean()
    a = (df["Home Goal FT"] < df["Away Goal FT"]).mean()
    h = 0 if np.isnan(h) else float(h)
    d = 0 if np.isnan(d) else float(d)
    a = 0 if np.isnan(a) else float(a)
    s = h + d + a
    if s <= 0:
        return (1/3, 1/3, 1/3)
    return (h/s, d/s, a/s)

def _btts_prob(df):
    if df is None or df.empty:
        return 0.5
    val = ((df["Home Goal FT"] > 0) & (df["Away Goal FT"] > 0)).mean()
    return 0 if np.isnan(val) else float(val)

def _over_prob(df, current_h, current_a, threshold):
    if df is None or df.empty:
        return 0.5
    extra = (df["Home Goal FT"] + df["Away Goal FT"]) - (current_h + current_a)
    val = (extra > threshold).mean()
    return 0 if np.isnan(val) else float(val)

def _blend(p_main, n_main, p_side, n_side, clamp=200):
    n_main = max(int(n_main), 0)
    n_side = max(int(n_side), 0)
    if n_main + n_side == 0: return p_main
    n_main = min(n_main, clamp)
    n_side = min(n_side, clamp // 2)
    return (n_main * p_main + n_side * p_side) / (n_main + n_side)

# =============== EV MATH ===============
def ev_back(prob, odds, commission=0.0):
    odds = max(1.01, float(odds or 1.01))
    win_ret = (odds - 1.0) * (1.0 - commission)
    return prob * win_ret - (1 - prob) * 1.0

def ev_lay(prob, odds, commission=0.0):
    odds = max(1.01, float(odds or 1.01))
    L = max(odds - 1.0, 1e-9)
    s = 1.0 / L
    return (1 - prob) * s * (1.0 - commission) - prob * 1.0

def kelly_fraction(prob, odds):
    # f* = (o*p - (1-p)) / (o-1)  (stake as fraction of bankroll)
    o = max(1.01, float(odds))
    p = float(prob)
    return max(0.0, (o*p - (1-p)) / (o-1))

def badge_ev(ev):
    if ev >= 0.05:  return f"🟢 **{ev*100:.1f}%**"
    if ev >= 0.02:  return f"🟡 {ev*100:.1f}%"
    return f"🔴 {ev*100:.1f}%"

# =============== PRIORS ===============
def league_priors(df_league, current_h, current_a, over_lines):
    # 1X2
    pH_L, pD_L, pA_L = _result_probs(df_league)
    priors = {
        "1": pH_L, "X": pD_L, "2": pA_L,
        "BTTS": _btts_prob(df_league)
    }
    for line in over_lines:
        priors[f"Over {line}"] = _over_prob(df_league, current_h, current_a, line)
    return priors

# =============== OPTIONAL EXTERNAL SIGNALS ===============
def get_external_signals(df_league, home_team, away_team):
    """
    Integra segnali da moduli opzionali.
    - squadre.compute_team_macro_stats(df, team, "Home"/"Away")
    - pattern_analysis.live_signals(df, home, away)          # se disponibile
    - macros.league_bias(df_league)                          # se disponibile
    """
    out = {"notes": []}

    # squadre.py – macro KPI
    try:
        from squadre import compute_team_macro_stats
        m_home = compute_team_macro_stats(df_league, home_team, "Home")
        m_away = compute_team_macro_stats(df_league, away_team, "Away")
        out["macro_home"] = m_home
        out["macro_away"] = m_away
        if m_home and m_away:
            # esempio: differenziale BTTS/Over come contesto
            try:
                bt_home = float(m_home.get("BTTS %", 0)/100 if isinstance(m_home.get("BTTS %"), (int,float)) else str(m_home.get("BTTS %")).replace("%",""))
            except:
                bt_home = 0
            try:
                bt_away = float(m_away.get("BTTS %", 0)/100 if isinstance(m_away.get("BTTS %"), (int,float)) else str(m_away.get("BTTS %")).replace("%",""))
            except:
                bt_away = 0
            out["notes"].append(f"KPI: BTTS {home_team}≈{m_home.get('BTTS %','-')} | {away_team}≈{m_away.get('BTTS %','-')}")
            out["btts_bias_hint"] = (bt_home + bt_away)/2
    except Exception:
        pass

    # pattern_analysis.py – pattern/tempi-gol live (se presente)
    try:
        import pattern_analysis as pa
        if hasattr(pa, "live_signals"):
            sig = pa.live_signals(df_league, home_team, away_team)
            out["pattern_signals"] = sig
            out["notes"].append("Pattern live attivi" if sig else "Pattern: nessun segnale forte")
    except Exception:
        pass

    # macros.py – bias campionato (se presente)
    try:
        import macros as m
        if hasattr(m, "league_bias"):
            bias = m.league_bias(df_league)  # es: {"over":..., "btts":...}
            out["macros_bias"] = bias
            out["notes"].append("Bias lega integrato (macros)")
    except Exception:
        pass

    return out

# =============== POST-MINUTE TABLE ===============
def compute_post_minute_stats(df, current_min):
    tf_bands = [(0,15),(16,30),(31,45),(46,60),(61,75),(76,90)]
    tf_labels = [f"{a}-{b}" for a,b in tf_bands]
    rec = {lbl: {"GF":0,"GS":0,"1+":0,"2+":0,"N":0} for lbl in tf_labels}

    for _, r in df.iterrows():
        mh = extract_minutes(pd.Series([r.get("minuti goal segnato home","")]))
        ma = extract_minutes(pd.Series([r.get("minuti goal segnato away","")]))
        future = [(m,"H") for m in mh if m>current_min] + [(m,"A") for m in ma if m>current_min]

        bucket = {lbl: {"GF":0,"GS":0} for lbl in tf_labels}
        for m, side in future:
            for lbl,(a,b) in zip(tf_labels, tf_bands):
                if a < m <= b:
                    bucket[lbl]["GF"] += 1 if side=="H" else 0
                    bucket[lbl]["GS"] += 1 if side=="A" else 0
                    break

        for lbl in tf_labels:
            gf, gs = bucket[lbl]["GF"], bucket[lbl]["GS"]
            t = gf+gs
            if t>0: rec[lbl]["1+"] += 1
            if t>=2: rec[lbl]["2+"] += 1
            rec[lbl]["GF"] += gf; rec[lbl]["GS"] += gs; rec[lbl]["N"] += 1

    return pd.DataFrame([{
        "Intervallo": lbl,
        "GF":v["GF"], "GS":v["GS"],
        "% 1+ Goal": round((v["1+"]/v["N"])*100,2) if v["N"]>0 else 0.0,
        "% 2+ Goal": round((v["2+"]/v["N"])*100,2) if v["N"]>0 else 0.0,
    } for lbl,v in rec.items()])

# =============== MAIN ===============
def run_live_minute_analysis(df: pd.DataFrame):
    st.set_page_config(page_title="Analisi Live Minuto", layout="wide")
    st.title("⏱️ Analisi Live — EV Advisor (AI)")

    # --- Selettori base
    champ_options = sorted(df["country"].dropna().astype(str).unique())
    champ_default = st.session_state.get("campionato_corrente", champ_options[0])
    col0,col1 = st.columns([1,2])
    with col0:
        champ = st.selectbox("🏆 Campionato", champ_options, index=champ_options.index(champ_default) if champ_default in champ_options else 0, key="champ_live")
    with col1:
        c1,c2 = st.columns(2)
        with c1:
            home_team = st.selectbox("🏠 Casa", sorted(df["Home"].dropna().unique()), key="home_live")
        with c2:
            away_team = st.selectbox("🚪 Trasferta", sorted(df["Away"].dropna().unique()), key="away_live")

    # --- Quote 1X2 Back/Lay
    q1,q2,q3 = st.columns(3)
    with q1:
        odd_home = st.number_input("📈 Quota Home (BACK)", 1.01, 50.0, float(st.session_state.get("odd_h",2.00)), step=0.01, key="odd_h")
    with q2:
        odd_draw = st.number_input("⚖️ Quota Pareggio (BACK)", 1.01, 50.0, float(st.session_state.get("odd_d",3.20)), step=0.01, key="odd_d")
    with q3:
        odd_away = st.number_input("📉 Quota Away (BACK)", 1.01, 50.0, float(st.session_state.get("odd_a",3.80)), step=0.01, key="odd_a")

    l1,l2,l3 = st.columns(3)
    with l1:
        lay_home = st.number_input("Quota Home (LAY)", 1.01, 50.0, value=float(round(odd_home+0.06,2)), step=0.01)
    with l2:
        lay_draw = st.number_input("Quota Pareggio (LAY)", 1.01, 50.0, value=float(round(odd_draw+0.06,2)), step=0.01)
    with l3:
        lay_away = st.number_input("Quota Away (LAY)", 1.01, 50.0, value=float(round(odd_away+0.06,2)), step=0.01)

    # Persist per Pre-Match
    st.session_state["quota_home"] = float(odd_home)
    st.session_state["quota_draw"] = float(odd_draw)
    st.session_state["quota_away"] = float(odd_away)

    # --- Stato live
    c_live = st.columns([2,1,1,1])
    with c_live[0]:
        current_min = st.slider("⏲️ Minuto attuale", 1, 120, int(st.session_state.get("minlive",45)), key="minlive")
    with c_live[1]:
        live_score_txt = st.text_input("📟 Risultato live", str(st.session_state.get("scorelive","0-0")), key="scorelive")
    with c_live[2]:
        commission = st.number_input("💸 Commissione exch.", 0.0, 0.10, 0.045, step=0.005)
    with c_live[3]:
        show_ext = st.toggle("🔎 Usa segnali esterni (se disponibili)", value=True)

    parsed = safe_parse_score(live_score_txt)
    if not parsed:
        st.error("Formato risultato non valido. Esempio: 1-1")
        return
    live_h, live_a = parsed
    st.caption(f"🔖 Label: `{label_match({'Odd home':odd_home,'Odd Away':odd_away})}` | Score live: **{live_h}-{live_a}** al **{current_min}′**")

    # --- Filtro dataset base
    df = df.copy()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)
    label_live = label_match({"Odd home": odd_home, "Odd Away": odd_away})
    df_league = df[(df["country"]==champ) & (df["Label"]==label_live)].copy()

    # Campione matchato & subset per squadra
    df_matched = _matches_matching_state(df_league, current_min, live_h, live_a)
    df_home_side = _matches_matching_state(df_league[df_league["Home"]==home_team], current_min, live_h, live_a)
    df_away_side = _matches_matching_state(df_league[df_league["Away"]==away_team], current_min, live_h, live_a)

    st.success(f"✅ {len(df_matched)} partite trovate | {sample_badge(len(df_matched))}  •  Team focus: {home_team} / {away_team}")

    # --- Probabilità 1X2 (blend)
    pH_L,pD_L,pA_L = _result_probs(df_matched)
    pH_H, pD_H, _   = _result_probs(df_home_side)
    _,    pD_A, pA_A= _result_probs(df_away_side)

    p_home = _blend(pH_L, len(df_matched), pH_H, len(df_home_side))
    p_away = _blend(pA_L, len(df_matched), pA_A, len(df_away_side))
    p_draw_side = _blend(pD_H, len(df_home_side), pD_A, len(df_away_side))
    p_draw = _blend(pD_L, len(df_matched), p_draw_side, len(df_home_side)+len(df_away_side))
    s = p_home + p_draw + p_away
    if s>0: p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s

    # --- Probabilità Over/BTTS (blend) — include Over 0.5
    over_lines = [0.5, 1.5, 2.5, 3.5]
    probs_over = {}
    for line in over_lines:
        pL = _over_prob(df_matched, live_h, live_a, line)
        pH = _over_prob(df_home_side, live_h, live_a, line)
        pA = _over_prob(df_away_side, live_h, live_a, line)
        probs_over[line] = _blend(pL, len(df_matched), (pH+pA)/2 if (len(df_home_side)+len(df_away_side))>0 else pL, len(df_home_side)+len(df_away_side))
    p_btts_L = _btts_prob(df_matched)
    p_btts_side = _blend(_btts_prob(df_home_side), len(df_home_side), _btts_prob(df_away_side), len(df_away_side))
    p_btts = _blend(p_btts_L, len(df_matched), p_btts_side, len(df_home_side)+len(df_away_side))

    # --- Priors (campionato/label, ignorando lo stato minuto/score)
    priors = league_priors(df_league, live_h, live_a, over_lines)

    # --- Quote mercati goal / BTTS (Over 0.5 incluso)
    with st.expander("⚙️ Imposta quote mercato per EV (live)", expanded=False):
        oc1, oc2, oc3, oc4, oc5 = st.columns(5)
        with oc1: q_over05 = st.number_input("Over 0.5", 1.01, 50.0, 1.30, step=0.01)
        with oc2: q_over15 = st.number_input("Over 1.5", 1.01, 50.0, 1.65, step=0.01)
        with oc3: q_over25 = st.number_input("Over 2.5", 1.01, 50.0, 2.40, step=0.01)
        with oc4: q_over35 = st.number_input("Over 3.5", 1.01, 50.0, 3.75, step=0.01)
        with oc5: q_btts   = st.number_input("BTTS (GG)", 1.01, 50.0, 2.10, step=0.01)

    # Persist comodo per Pre-Match ROI/EV manuali
    st.session_state["quota_over"]  = float(q_over25)
    st.session_state["quota_under"] = 1.80  # placeholder se servisse altrove

    # --- EV table
    rows = []
    def add_row(market, kind, price, p, prior_p):
        ev_b = ev_back(p, price, commission) if kind=="Back" else ev_lay(p, price, commission)
        fair = 1/max(p,1e-9)
        edge = (fair - price)/fair  # positivo se prezzo < fair (value per Back)
        kelly = kelly_fraction(p, price) if kind=="Back" else None
        quality = len(df_matched)
        delta = abs(p - prior_p)
        # AI score: combina EV, qualità campione, delta vs prior
        ev_pos = max(0.0, ev_b)
        q_w = min(1.0, math.log1p(quality)/math.log1p(150))  # 0..1
        d_w = 1.0 + min(0.4, delta)                          # boost max +40pp
        ai_score = min(100.0, 100.0 * (ev_pos*4.0) * q_w * d_w)  # EV 0.05 → 20%; scalato
        rows.append({
            "Mercato": market, "Tipo": kind, "Quota": float(price),
            "Prob": float(p), "Prob %": round(p*100,1),
            "Fair": round(fair,2), "Edge": edge, "EV": ev_b, "EV %": round(ev_b*100,1),
            "½-Kelly %": round(kelly*50*100,1) if kelly is not None else None,
            "Campione": quality, "Δ vs prior": round((p-prior_p)*100,1),
            "AI score": round(ai_score,1)
        })

    # 1X2
    add_row("1 (Home)", "Back", odd_home, p_home, priors["1"])
    add_row("X (Draw)", "Back", odd_draw, p_draw, priors["X"])
    add_row("2 (Away)", "Back", odd_away, p_away, priors["2"])
    add_row("1 (Home)", "Lay",  lay_home, p_home, priors["1"])
    add_row("X (Draw)", "Lay",  lay_draw, p_draw, priors["X"])
    add_row("2 (Away)", "Lay",  lay_away, p_away, priors["2"])

    # Over
    q_map = {0.5:q_over05, 1.5:q_over15, 2.5:q_over25, 3.5:q_over35}
    for line, q in q_map.items():
        add_row(f"Over {line}", "Back", q, probs_over[line], priors[f"Over {line}"])
        add_row(f"Over {line}", "Lay",  q, probs_over[line], priors[f"Over {line}"])

    # BTTS
    add_row("BTTS (GG)", "Back", q_btts, p_btts, priors["BTTS"])
    add_row("BTTS (GG)", "Lay",  q_btts, p_btts, priors["BTTS"])

    df_ev = pd.DataFrame(rows).sort_values(["EV","AI score"], ascending=[False,False]).reset_index(drop=True)

    st.markdown("## 🧠 EV Advisor — opportunità ordinate per valore atteso")
    st.dataframe(
        df_ev[["Mercato","Tipo","Quota","Prob %","Fair","Edge","EV %","½-Kelly %","Campione","Δ vs prior","AI score"]],
        use_container_width=True, height=360
    )

    # Suggerimenti rapidi
    top = df_ev.head(3).to_dict(orient="records")
    if top:
        tips = []
        for r in top:
            pill = badge_ev(r["EV"]/100 if r["EV"]>1 else r["EV"])
            kelly_txt = f" | ½-Kelly≈{r['½-Kelly %']:.1f}%" if r["½-Kelly %"] is not None else ""
            tips.append(f"**{r['Mercato']} — {r['Tipo']}** → {pill} (p≈{r['Prob %']:.0f}%, q={r['Quota']:.2f}{kelly_txt})")
        st.info("\n".join(tips))

    # Riquadri riassuntivi campione
    colK1,colK2,colK3,colK4 = st.columns(4)
    colK1.metric("Sample Campionato", len(df_matched), help=sample_badge(len(df_matched)))
    colK2.metric("Sample Squadra Home", len(df_home_side), help=sample_badge(len(df_home_side)))
    colK3.metric("Sample Squadra Away", len(df_away_side), help=sample_badge(len(df_away_side)))
    colK4.metric("P(Over 0.5) now→FT", f"{probs_over[0.5]*100:.1f}%")

    # Sezioni statistiche
    left, right = st.columns(2)
    with left:
        st.subheader("📊 Campionato (stesso label & stato)")
        if len(df_matched):
            home_w = (df_matched["Home Goal FT"] > df_matched["Away Goal FT"]).mean()*100
            draw   = (df_matched["Home Goal FT"] == df_matched["Away Goal FT"]).mean()*100
            away_w = (df_matched["Home Goal FT"] < df_matched["Away Goal FT"]).mean()*100
            df_lea = pd.DataFrame({"Campionato":[len(df_matched),home_w,draw,away_w]},
                                  index=["Matches","Home %","Draw %","Away %"])
            st.dataframe(df_lea.style.format("{:.2f}").apply(color_stat_rows, axis=1), use_container_width=True)
            st.caption("⏱️ Goal per intervalli (post-minuto)")
            st.dataframe(compute_post_minute_stats(df_matched, current_min), use_container_width=True)
        else:
            st.info("Nessun match nel campione.")

    with right:
        st.subheader(f"📊 Squadre — {home_team} / {away_team}")
        if len(df_home_side) or len(df_away_side):
            def _mk(team_df, who):
                if team_df.empty: return pd.DataFrame({who:[0,0,0,0]}, index=["Matches","Win %","Draw %","Lose %"])
                win = (team_df["Home Goal FT"] > team_df["Away Goal FT"]).mean()*100 if who=="Home" else (team_df["Away Goal FT"] > team_df["Home Goal FT"]).mean()*100
                draw= (team_df["Home Goal FT"] == team_df["Away Goal FT"]).mean()*100
                lose= 100-win-draw
                return pd.DataFrame({who:[len(team_df),win,draw,lose]}, index=["Matches","Win %","Draw %","Lose %"])
            st.dataframe(pd.concat([_mk(df_home_side,"Home"), _mk(df_away_side,"Away")],axis=1).style.format("{:.2f}").apply(color_stat_rows, axis=1),
                         use_container_width=True)
        else:
            st.info("Nessun match specifico per le squadre con questo stato.")

    # --- Segnali esterni (facoltativi)
    if show_ext:
        ext = get_external_signals(df_league, home_team, away_team)
        if ext.get("macro_home") or ext.get("pattern_signals") or ext.get("macros_bias"):
            st.subheader("🧩 Segnali esterni (pattern, macro KPI, bias lega)")
            if ext.get("macro_home") and ext.get("macro_away"):
                st.write("**Macro KPI (estratto):**")
                st.json({"Home": ext["macro_home"], "Away": ext["macro_away"]})
            if ext.get("pattern_signals"):
                st.write("**Pattern live**:")
                st.json(ext["pattern_signals"])
            if ext.get("macros_bias"):
                st.write("**Bias lega (macros)**:")
                st.json(ext["macros_bias"])
            if ext.get("notes"):
                st.caption(" • ".join(ext["notes"]))
        else:
            st.caption("Nessun segnale esterno disponibile nei moduli caricati.")
