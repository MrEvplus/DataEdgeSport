# analisi_live_minuto.py — v3.5.1 UI Pro (fix styler) — Live 1X2 + Over/BTTS + EV Advisor (AI) + Write-back + Hedging

import math
import numpy as np
import pandas as pd
import streamlit as st

from utils import label_match, extract_minutes

# =========================
# ---- CONFIG / SHARED ----
# =========================
_SHARED_PREFIX = "prematch:shared:"

def _shared_key(name: str) -> str:
    return f"{_SHARED_PREFIX}{name}"

def _set_shared_quote(name: str, value: float):
    """Aggiorna le quote condivise usate in Pre-Match (one-way write-back)."""
    st.session_state[_shared_key(name)] = float(value)

# =========================
# -------- THEME/CSS -------
# =========================
_BASE_CSS = """
<style>
:root {
  --bg: #0b1220;
  --card: #111827;
  --muted: #6b7280;
  --accent: #22c55e;
  --accent-soft: rgba(34,197,94,.14);
  --danger: #ef4444;
  --danger-soft: rgba(239,68,68,.14);
  --warn: #f59e0b;
  --warn-soft: rgba(245,158,11,.14);
  --chip: #1f2937;
  --chip-border: #374151;
}
.block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
div.stTabs [role="tablist"] button {font-weight:600;}
.badge {display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem; border-radius:999px; font-size:.82rem; border:1px solid var(--chip-border); background:var(--chip);}
.badge b {color:#fff;}
.kpi {display:flex; gap:.5rem; align-items:center;}
.kpi .dot {width:.55rem; height:.55rem; border-radius:999px; background:var(--muted);}
.kpi .dot.ok {background:var(--accent);}
.kpi .dot.mid {background:var(--warn);}
.kpi .dot.low {background:var(--danger);}
.evpos {background: var(--accent-soft);}
.evneg {background: var(--danger-soft);}
.small {color:var(--muted); font-size:.85rem;}
.ev-pill {padding:.2rem .45rem; border-radius:.5rem; background:var(--chip); border:1px solid var(--chip-border); font-size:.78rem;}
table td, table th {vertical-align: middle;}
</style>
"""

def _inject_css():
    st.markdown(_BASE_CSS, unsafe_allow_html=True)

# =========================
# ---------- UTILS --------
# =========================
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

def sample_badge(n: int) -> str:
    if n < 30:  return "🔴 Campione piccolo"
    if n < 100: return "🟡 Campione medio"
    return "🟢 Campione robusto"

# =========================
# ---------- EV -----------
# =========================
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
    o = max(1.01, float(odds))
    p = float(prob)
    return max(0.0, (o*p - (1-p)) / (o-1))

def badge_ev(ev):
    if ev >= 0.05:  return f"🟢 **{ev*100:.1f}%**"
    if ev >= 0.02:  return f"🟡 {ev*100:.1f}%"
    return f"🔴 {ev*100:.1f}%"

# =========================
# ---------- PRIORS -------
# =========================
def league_priors(df_league, current_h, current_a, over_lines):
    pH_L, pD_L, pA_L = _result_probs(df_league)
    priors = {"1": pH_L, "X": pD_L, "2": pA_L, "BTTS": _btts_prob(df_league)}
    for line in over_lines:
        priors[f"Over {line}"] = _over_prob(df_league, current_h, current_a, line)
    return priors

# =========================
# -- EXTERNAL SIGNALS -----
# =========================
def get_external_signals(df_league, home_team, away_team):
    out = {"notes": []}
    # Macro KPI (squadre.py)
    try:
        from squadre import compute_team_macro_stats
        m_home = compute_team_macro_stats(df_league, home_team, "Home")
        m_away = compute_team_macro_stats(df_league, away_team, "Away")
        out["macro_home"] = m_home
        out["macro_away"] = m_away
        if m_home and m_away:
            out["notes"].append("Macro KPI caricati")
    except Exception:
        pass
    # Pattern (pattern_analysis.py)
    try:
        import pattern_analysis as pa
        if hasattr(pa, "live_signals"):
            sig = pa.live_signals(df_league, home_team, away_team)
            out["pattern_signals"] = sig
            out["notes"].append("Pattern live attivi" if sig else "Pattern: nessun segnale forte")
    except Exception:
        pass
    # Bias lega (macros.py)
    try:
        import macros as m
        if hasattr(m, "league_bias"):
            bias = m.league_bias(df_league)
            out["macros_bias"] = bias
            out["notes"].append("Bias lega integrato (macros)")
    except Exception:
        pass
    return out

# =========================
# -- POST-MINUTE TABLE ----
# =========================
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
                    if side=="H": bucket[lbl]["GF"] += 1
                    else:         bucket[lbl]["GS"] += 1
                    break
        for lbl in tf_labels:
            gf, gs = bucket[lbl]["GF"], bucket[lbl]["GS"]
            t = gf+gs
            if t>0:   rec[lbl]["1+"] += 1
            if t>=2:  rec[lbl]["2+"] += 1
            rec[lbl]["GF"] += gf; rec[lbl]["GS"] += gs; rec[lbl]["N"] += 1

    return pd.DataFrame([{
        "Intervallo": lbl,
        "GF":v["GF"], "GS":v["GS"],
        "% 1+ Goal": round((v["1+"]/v["N"])*100,2) if v["N"]>0 else 0.0,
        "% 2+ Goal": round((v["2+"]/v["N"])*100,2) if v["N"]>0 else 0.0,
    } for lbl,v in rec.items()])

# =========================
# --------- MAIN ----------
# =========================
def run_live_minute_analysis(df: pd.DataFrame):
    st.set_page_config(page_title="Analisi Live Minuto", layout="wide")
    _inject_css()
    st.title("⏱️ Analisi Live — EV Advisor (AI)")

    # ========== TAB 1: SETUP ==========
    tab_setup, tab_ev, tab_league, tab_teams, tab_signals = st.tabs(
        ["🎛️ Setup", "🧠 EV Advisor", "📊 Campionato", "📈 Squadre", "🧩 Segnali"]
    )

    with tab_setup:
        # --- Selettori base
        champ_options = sorted(df["country"].dropna().astype(str).unique())
        champ_default = st.session_state.get("campionato_corrente", champ_options[0])

        col0,col1 = st.columns([1.2,2])
        with col0:
            champ = st.selectbox("🏆 Campionato", champ_options,
                                 index=champ_options.index(champ_default) if champ_default in champ_options else 0,
                                 key="champ_live")
        with col1:
            c1,c2 = st.columns(2)
            with c1:
                home_team = st.selectbox("🏠 Casa", sorted(df["Home"].dropna().unique()), key="home_live")
            with c2:
                away_team = st.selectbox("🚪 Trasferta", sorted(df["Away"].dropna().unique()), key="away_live")

        # --- Quote 1X2 Back/Lay
        st.subheader("Quote 1X2 Live")
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

        # --- Stato live / commissione
        c_live = st.columns([2,1,1,1])
        with c_live[0]:
            current_min = st.slider("⏲️ Minuto attuale", 1, 120, int(st.session_state.get("minlive",45)), key="minlive")
        with c_live[1]:
            live_score_txt = st.text_input("📟 Risultato live", str(st.session_state.get("scorelive","0-0")), key="scorelive")
        with c_live[2]:
            commission = st.number_input("💸 Commissione exch.", 0.0, 0.10, 0.045, step=0.005, help="Commissione applicata a vincite")
        with c_live[3]:
            show_ext = st.toggle("🔎 Usa segnali esterni", value=True)

        parsed = safe_parse_score(live_score_txt)
        if not parsed:
            st.error("Formato risultato non valido (esempio: 1-1).")
            return
        live_h, live_a = parsed

        # --- Label + chip
        label_live = label_match({"Odd home": odd_home, "Odd Away": odd_away})
        chip_html = f"""
        <span class='badge'>🔖 <b>{label_live}</b></span>
        <span class='badge'>⏱️ <b>{current_min}'</b></span>
        <span class='badge'>📟 <b>{live_h}-{live_a}</b></span>
        <span class='badge small'>campionato <b>{champ}</b></span>
        """
        st.markdown(chip_html, unsafe_allow_html=True)

        # --- Quote mercati goal / BTTS (Over 0.5 incluso)
        with st.expander("⚙️ Imposta quote mercato per EV (live)", expanded=False):
            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            with oc1: q_over05 = st.number_input("Over 0.5", 1.01, 50.0, 1.30, step=0.01)
            with oc2: q_over15 = st.number_input("Over 1.5", 1.01, 50.0, 1.65, step=0.01)
            with oc3: q_over25 = st.number_input("Over 2.5", 1.01, 50.0, 2.40, step=0.01)
            with oc4: q_over35 = st.number_input("Over 3.5", 1.01, 50.0, 3.75, step=0.01)
            with oc5: q_btts   = st.number_input("BTTS (GG)", 1.01, 50.0, 2.10, step=0.01)

        # Write-back verso Pre-Match (one-way)
        _set_shared_quote("ov05", q_over05)
        _set_shared_quote("ov15", q_over15)
        _set_shared_quote("ov25", q_over25)
        _set_shared_quote("ov35", q_over35)
        _set_shared_quote("btts", q_btts)

        # Persist comodo
        st.session_state["quota_over"]  = float(q_over25)
        st.session_state["quota_under"] = 1.80  # placeholder

        # --- Prepara dataset filtrati (riuso nelle altre tab)
        st.session_state["_live_ctx"] = {
            "champ": champ, "home": home_team, "away": away_team,
            "odd_home": odd_home, "odd_draw": odd_draw, "odd_away": odd_away,
            "lay_home": lay_home, "lay_draw": lay_draw, "lay_away": lay_away,
            "minute": current_min, "score": (live_h, live_a),
            "commission": commission, "label": label_live,
            "q_over": {0.5:q_over05,1.5:q_over15,2.5:q_over25,3.5:q_over35}, "q_btts": q_btts,
        }

    # ========== PRE-CALCOLI COMUNI ==========
    if "_live_ctx" not in st.session_state:
        st.stop()

    ctx = st.session_state["_live_ctx"]
    champ, home_team, away_team = ctx["champ"], ctx["home"], ctx["away"]
    odd_home, odd_draw, odd_away = ctx["odd_home"], ctx["odd_draw"], ctx["odd_away"]
    lay_home, lay_draw, lay_away = ctx["lay_home"], ctx["lay_draw"], ctx["lay_away"]
    current_min, (live_h, live_a) = ctx["minute"], ctx["score"]
    commission, label_live = ctx["commission"], ctx["label"]
    q_map = ctx["q_over"]; q_over05,q_over15,q_over25,q_over35 = q_map[0.5],q_map[1.5],q_map[2.5],q_map[3.5]
    q_btts = ctx["q_btts"]

    df = df.copy()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)
    df_league = df[(df["country"]==champ) & (df["Label"]==label_live)].copy()
    df_matched   = _matches_matching_state(df_league, current_min, live_h, live_a)
    df_home_side = _matches_matching_state(df_league[df_league["Home"]==home_team], current_min, live_h, live_a)
    df_away_side = _matches_matching_state(df_league[df_league["Away"]==away_team], current_min, live_h, live_a)

    # Probabilità 1X2
    pH_L,pD_L,pA_L = _result_probs(df_matched)
    pH_H, pD_H, _   = _result_probs(df_home_side)
    _,    pD_A, pA_A= _result_probs(df_away_side)
    p_home = _blend(pH_L, len(df_matched), pH_H, len(df_home_side))
    p_away = _blend(pA_L, len(df_matched), pA_A, len(df_away_side))
    p_draw_side = _blend(pD_H, len(df_home_side), pD_A, len(df_away_side))
    p_draw = _blend(pD_L, len(df_matched), p_draw_side, len(df_home_side)+len(df_away_side))
    s = p_home + p_draw + p_away
    if s>0: p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s

    # Over/BTTS
    over_lines = [0.5, 1.5, 2.5, 3.5]
    probs_over = {}
    for line in over_lines:
        pL = _over_prob(df_matched,    live_h, live_a, line)
        pH = _over_prob(df_home_side,  live_h, live_a, line)
        pA = _over_prob(df_away_side,  live_h, live_a, line)
        probs_over[line] = _blend(pL, len(df_matched), (pH+pA)/2 if (len(df_home_side)+len(df_away_side))>0 else pL,
                                  len(df_home_side)+len(df_away_side))
    p_btts_L = _btts_prob(df_matched)
    p_btts_side = _blend(_btts_prob(df_home_side), len(df_home_side), _btts_prob(df_away_side), len(df_away_side))
    p_btts = _blend(p_btts_L, len(df_matched), p_btts_side, len(df_home_side)+len(df_away_side))

    priors = league_priors(df_league, live_h, live_a, over_lines)

    # Helper per EV rows
    def _ev_rows():
        rows = []
        def add_row(market, kind, price, p, prior_p):
            ev_b = ev_back(p, price, commission) if kind=="Back" else ev_lay(p, price, commission)
            fair = 1/max(p,1e-9)
            edge = (fair - price)/fair
            kelly = kelly_fraction(p, price) if kind=="Back" else None
            quality = len(df_matched)
            delta = abs(p - prior_p)
            ev_pos = max(0.0, ev_b)
            q_w = min(1.0, math.log1p(max(1,quality))/math.log1p(150))
            d_w = 1.0 + min(0.4, delta)
            ai_score = min(100.0, 100.0 * (ev_pos*4.0) * q_w * d_w)
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
        for line, q in {0.5:q_over05, 1.5:q_over15, 2.5:q_over25, 3.5:q_over35}.items():
            add_row(f"Over {line}", "Back", q, probs_over[line], priors[f"Over {line}"])
            add_row(f"Over {line}", "Lay",  q, probs_over[line], priors[f"Over {line}"])
        # BTTS
        add_row("BTTS (GG)", "Back", q_btts, p_btts, priors["BTTS"])
        add_row("BTTS (GG)", "Lay",  q_btts, p_btts, priors["BTTS"])
        return pd.DataFrame(rows)

    df_ev_full = _ev_rows()

    # Styler EV (robusto alle colonne presenti)
    def _style_ev(df_):
        fmt_map = {}
        if "Quota" in df_.columns:    fmt_map["Quota"] = "{:.2f}"
        if "Prob %%" in df_.columns:  fmt_map["Prob %%"] = "{:.1f}%"
        if "Prob %" in df_.columns:   fmt_map["Prob %"] = "{:.1f}%"
        if "Fair" in df_.columns:     fmt_map["Fair"] = "{:.2f}"
        if "Edge" in df_.columns:     fmt_map["Edge"] = "{:.1%}"
        if "EV %" in df_.columns:     fmt_map["EV %"] = "{:.1f}%"
        if "½-Kelly %" in df_.columns:fmt_map["½-Kelly %"] = "{:.1f}%"

        def _bg(s):
            # verde per valori >0, rosso per <0
            out = []
            for v in s:
                try:
                    val = float(v)
                except Exception:
                    out.append("")
                    continue
                if val > 0:
                    out.append("background-color: rgba(34,197,94,0.14)")
                elif val < 0:
                    out.append("background-color: rgba(239,68,68,0.14)")
                else:
                    out.append("")
            return out

        sty = df_.style.format(fmt_map)
        # Applica highlight solo alle colonne esistenti
        if "EV" in df_.columns:
            sty = sty.apply(_bg, subset=["EV"])
        if "EV %" in df_.columns:
            sty = sty.apply(_bg, subset=["EV %"])
        if "Edge" in df_.columns:
            sty = sty.apply(_bg, subset=["Edge"])
        return sty

    # ========== TAB 2: EV ADVISOR ==========
    with tab_ev:
        st.subheader("EV Advisor — opportunità ordinate per valore atteso")
        cflt1, cflt2, cflt3, cflt4 = st.columns([1,1,1,1.2])
        with cflt1:
            only_pos = st.checkbox("Solo EV+", value=True)
        with cflt2:
            thr = st.number_input("Soglia EV% min", -20.0, 20.0, 0.0, step=0.5)
        with cflt3:
            min_samp = st.number_input("Min campione", 0, 500, 30, step=10)
        with cflt4:
            order = st.selectbox("Ordina per", ["EV", "AI score", "Edge", "½-Kelly %"], index=0)

        view = df_ev_full.copy()
        if only_pos:
            view = view[view["EV"] > 0]
        view = view[view["EV %"] >= thr]
        view = view[view["Campione"] >= min_samp]
        view = view.sort_values(order, ascending=False).reset_index(drop=True)

        st.dataframe(
            _style_ev(view[["Mercato","Tipo","Quota","Prob %","Fair","Edge","EV %","½-Kelly %","Campione","Δ vs prior","AI score"]]),
            use_container_width=True, height=420
        )

        # Hedging rapido (migliori segnali Back)
        top_back = view[(view["Tipo"]=="Back") & (view["EV"]>0)].head(3).to_dict(orient="records")
        if top_back:
            with st.expander("🛡️ Hedging rapido (green-up su quota target)"):
                st.caption("Formula: Back a quota **ob** con stake **B** → Lay a quota **ol** con **L = (B·ob)/ol** per profitto uguale.")
                for i, r in enumerate(top_back, start=1):
                    st.markdown(f"**#{i} {r['Mercato']}** — Back {r['Quota']:.2f} | p≈{r['Prob %']:.0f}% | EV {r['EV %']:.1f}%")
                    cA, cB, cC = st.columns([1,1,1])
                    with cA:
                        stake_B = st.number_input(f"Stake Back (#{i})", 1.0, 10000.0, 100.0, step=10.0, key=f"hedge_B_{i}")
                    with cB:
                        target_ol = st.number_input(f"Target Lay (#{i})", 1.01, 50.0, max(1.01, round(float(r['Quota'])*0.85,2)), step=0.01, key=f"hedge_ol_{i}")
                    with cC:
                        L = (stake_B * float(r["Quota"])) / float(target_ol)
                        profit = L - stake_B
                        st.write(f"**Lay stake** ≈ {L:.2f}")
                        st.write(f"**Profit atteso** ≈ {profit:.2f} (verde)")

        # Tips rapidi
        top = view.head(3).to_dict(orient="records")
        if top:
            pills = []
            for r in top:
                pill = badge_ev(r["EV"]/100 if r["EV"]>1 else r["EV"])
                kelly_txt = f" • ½-Kelly≈{r['½-Kelly %']:.1f}%" if r["½-Kelly %"] is not None else ""
                pills.append(f"<span class='ev-pill'><b>{r['Mercato']} {r['Tipo']}</b> {pill} — q={r['Quota']:.2f}{kelly_txt}</span>")
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        # KPI riassuntivi
        colK1,colK2,colK3,colK4 = st.columns(4)
        colK1.metric("Sample Campionato", len(df_matched), help=sample_badge(len(df_matched)))
        colK2.metric("Sample Home", len(df_home_side), help=sample_badge(len(df_home_side)))
        colK3.metric("Sample Away", len(df_away_side), help=sample_badge(len(df_away_side)))
        colK4.metric("P(Over 0.5) now→FT", f"{probs_over[0.5]*100:.1f}%")

    # ========== TAB 3: CAMPIONATO ==========
    with tab_league:
        st.subheader("Campionato — stesso label & stato live")
        if len(df_matched):
            home_w = (df_matched["Home Goal FT"] > df_matched["Away Goal FT"]).mean()*100
            draw   = (df_matched["Home Goal FT"] == df_matched["Away Goal FT"]).mean()*100
            away_w = (df_matched["Home Goal FT"] < df_matched["Away Goal FT"]).mean()*100
            df_lea = pd.DataFrame({"Campionato":[len(df_matched),home_w,draw,away_w]},
                                  index=["Matches","Home %","Draw %","Away %"])
            st.dataframe(df_lea.style.format("{:.2f}"), use_container_width=True)
            st.caption("⏱️ Goal per intervalli (post-minuto)")
            st.dataframe(compute_post_minute_stats(df_matched, current_min), use_container_width=True)
        else:
            st.info("Nessun match nel campione.")

        with st.expander("📑 Partite considerate", expanded=False):
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

    # ========== TAB 4: SQUADRE ==========
    with tab_teams:
        st.subheader(f"Squadre — {home_team} (Home) / {away_team} (Away)")
        if len(df_home_side) or len(df_away_side):
            def _mk(team_df, who):
                if team_df.empty: return pd.DataFrame({who:[0,0,0,0]}, index=["Matches","Win %","Draw %","Lose %"])
                win = (team_df["Home Goal FT"] > team_df["Away Goal FT"]).mean()*100 if who=="Home" else (team_df["Away Goal FT"] > team_df["Home Goal FT"]).mean()*100
                draw= (team_df["Home Goal FT"] == team_df["Away Goal FT"]).mean()*100
                lose= 100-win-draw
                return pd.DataFrame({who:[len(team_df),win,draw,lose]}, index=["Matches","Win %","Draw %","Lose %"])
            st.dataframe(pd.concat([_mk(df_home_side,"Home"), _mk(df_away_side,"Away")],axis=1).style.format("{:.2f}"), use_container_width=True)
            st.caption("⏱️ Goal per intervalli (post-minuto)")
            tf_home = compute_post_minute_stats(df_home_side, current_min) if len(df_home_side)>0 else pd.DataFrame()
            tf_away = compute_post_minute_stats(df_away_side, current_min) if len(df_away_side)>0 else pd.DataFrame()
            if not tf_home.empty: st.dataframe(tf_home, use_container_width=True)
            if not tf_away.empty: st.dataframe(tf_away, use_container_width=True)
        else:
            st.info("Nessun match specifico per le squadre con questo stato.")

    # ========== TAB 5: SEGNALI ==========
    with tab_signals:
        st.subheader("Segnali esterni (pattern, macro KPI, bias lega)")
        ext = get_external_signals(df_league, home_team, away_team)
        if ext.get("macro_home") or ext.get("pattern_signals") or ext.get("macros_bias"):
            if ext.get("macro_home") and ext.get("macro_away"):
                st.write("**Macro KPI (estratto)**")
                st.json({"Home": ext["macro_home"], "Away": ext["macro_away"]})
            if ext.get("pattern_signals"):
                st.write("**Pattern live**")
                st.json(ext["pattern_signals"])
            if ext.get("macros_bias"):
                st.write("**Bias lega (macros)**")
                st.json(ext["macros_bias"])
            if ext.get("notes"):
                st.caption(" • ".join(ext["notes"]))
        else:
            st.caption("Nessun segnale esterno disponibile nei moduli caricati.")
