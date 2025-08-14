# analisi_live_minuto.py — v3.9 ProTrader (UI polish)
# UI professionale a TAB per trader di calcio + EV 1X2 Back/Lay, Over 0.5/1.5/2.5/3.5, BTTS
# EV Advisor (AI score), CS/Hedge, segnali esterni (pattern/squadre/macros), write-back per Pre-Match.
# LOGICA PREESISTENTE INVARIATA, SOLO ESTENSIONI/UX.

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
  --muted: #9ca3af;
  --text: #e5e7eb;
  --accent: #22c55e;
  --accent-soft: rgba(34,197,94,.14);
  --danger: #ef4444;
  --danger-soft: rgba(239,68,68,.14);
  --warn: #f59e0b;
  --warn-soft: rgba(245,158,11,.14);
  --chip: #111827;
  --chip-border: #374151;
}
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
div.stTabs [role="tablist"] button {font-weight:600;}
.badge {display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem; border-radius:999px; font-size:.82rem; border:1px solid var(--chip-border); background:var(--chip); color:var(--text);}
.badge b {color:#fff;}
.small {color:var(--muted); font-size:.85rem;}
.ev-pill {padding:.2rem .45rem; border-radius:.5rem; background:var(--chip); border:1px solid var(--chip-border); font-size:.78rem; color:var(--text);}
table td, table th {vertical-align: middle;}
.dataframe td {font-size: 0.92rem;}
.dataframe th {font-size: 0.86rem; color: var(--muted);}
.kpi .dot {width:.55rem; height:.55rem; border-radius:999px; display:inline-block; margin-right:.35rem; background:var(--muted);}
.kpi .ok {background:var(--accent);}
.kpi .mid {background:var(--warn);}
.kpi .low {background:var(--danger);}
.hint {color: var(--muted); font-size: .9rem;}
.card {background: var(--card); border: 1px solid var(--chip-border); border-radius: .9rem; padding: 1rem;}
.card h4 {margin: 0 0 .5rem 0; font-size: 1.05rem;}
.grid {display:grid; grid-template-columns: repeat(2,minmax(0,1fr)); gap:.5rem .75rem;}
.kv {display:flex; justify-content:space-between; gap:.5rem; font-size:.95rem;}
.kv span:first-child {color: var(--muted);}
.pills {display:flex; flex-wrap:wrap; gap:.5rem;}
.pill {padding:.25rem .55rem; border-radius:999px; border:1px solid var(--chip-border); background:var(--chip); font-size:.75rem;}
.good {background: var(--accent-soft);}
.warn {background: var(--warn-soft);}
.bad  {background: var(--danger-soft);}
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
# ----------- EV ----------
# =========================
def ev_back(prob, odds, commission=0.0):
    odds = max(1.01, float(odds or 1.01))
    win_ret = (odds - 1.0) * (1.0 - commission)
    return prob * win_ret - (1 - prob) * 1.0

def ev_lay(prob, odds, commission=0.0):
    odds = max(1.01, float(odds or 1.01))
    L = max(odds - 1.0, 1e-9)   # liability per 1 stake
    s = 1.0 / L                 # lay stake per liability 1
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
# ---- CS / Hedge utils ---
# =========================
def poisson_pmf(k: int, lam: float) -> float:
    try:
        return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError:
        return 0.0

def final_cs_distribution(lam_home_add: float, lam_away_add: float, cur_h: int, cur_a: int, max_goals_delta: int = 6):
    probs = {}
    for x in range(0, max_goals_delta + 1):
        for y in range(0, max_goals_delta + 1):
            p = poisson_pmf(x, lam_home_add) * poisson_pmf(y, lam_away_add)
            cs = f"{cur_h + x}-{cur_a + y}"
            probs[cs] = probs.get(cs, 0.0) + p
    s = sum(probs.values())
    if s > 0:
        for k in probs:
            probs[k] /= s
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

def estimate_remaining_lambdas(df: pd.DataFrame, current_min: int, focus_home: bool):
    tf = compute_post_minute_stats(df, current_min)
    total_gf = tf["GF"].sum()
    total_gs = tf["GS"].sum()
    n = len(df) if len(df) > 0 else 1
    lam_for = total_gf / max(1, n)
    lam_against = total_gs / max(1, n)
    lam_for = 0.5 * lam_for + 0.5 * 0.6
    lam_against = 0.5 * lam_against + 0.5 * 0.6
    return max(0.01, lam_for), max(0.01, lam_against)

# =========================
# -------- STYLERS --------
# =========================
def _style_table(df_):
    fmt_map = {}
    for col in df_.columns:
        if col in ("Quota","Fair"): fmt_map[col] = "{:.2f}"
        if col.endswith("%"):       fmt_map[col] = "{:.1f}%"
        if col in ("EV","Edge"):    fmt_map[col] = "{:.3f}"
        if col == "½-Kelly %":      fmt_map[col] = "{:.1f}%"
    def _bg_posneg(s):
        out=[]
        for v in s:
            try: fv=float(v)
            except: out.append(""); continue
            if fv>0:  out.append("background-color: rgba(34,197,94,0.14)")
            elif fv<0:out.append("background-color: rgba(239,68,68,0.14)")
            else:     out.append("")
        return out
    sty = df_.style.format(fmt_map)
    for c in ("EV","EV %","Edge"):
        if c in df_.columns:
            sty = sty.apply(_bg_posneg, subset=[c])
    return sty

# ======== helpers Segnali (solo UI) ========
def _pct_str(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        s = str(x)
        return s if s.endswith("%") else f"{s}%"

def _card_macro(title: str, stats: dict):
    if not isinstance(stats, dict):
        st.write("—")
        return
    win = _pct_str(stats.get("Win %", stats.get("Win%", stats.get("Win", ""))))
    draw = _pct_str(stats.get("Draw %", stats.get("Draw%", stats.get("Draw", ""))))
    loss = _pct_str(stats.get("Loss %", stats.get("Loss%", stats.get("Loss", ""))))
    avg_for = stats.get("Avg Goals Scored", stats.get("GF avg", stats.get("GF", "-")))
    avg_ag  = stats.get("Avg Goals Conceded", stats.get("GA avg", stats.get("GA", "-")))
    btts    = _pct_str(stats.get("BTTS %", stats.get("BTTS%", stats.get("BTTS", ""))))
    mp      = stats.get("Matches Played", stats.get("Matches", "-"))
    st.markdown(f"""
    <div class="card">
      <h4>{title}</h4>
      <div class="grid">
        <div class="kv"><span>Matches</span><b>{mp}</b></div>
        <div class="kv"><span>BTTS%</span><b>{btts}</b></div>
        <div class="kv"><span>Win%</span><b>{win}</b></div>
        <div class="kv"><span>Draw%</span><b>{draw}</b></div>
        <div class="kv"><span>Loss%</span><b>{loss}</b></div>
        <div class="kv"><span>Avg GF</span><b>{avg_for}</b></div>
        <div class="kv"><span>Avg GA</span><b>{avg_ag}</b></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def _pills_from_patterns(pattern_obj):
    # pattern_obj può essere lista o dict. Creiamo etichette leggibili.
    pills = []
    try:
        if isinstance(pattern_obj, dict):
            items = sorted(pattern_obj.items(), key=lambda kv: str(kv[0]))
            for k,v in items:
                label = f"{k}: {v}"
                pills.append(label)
        elif isinstance(pattern_obj, list):
            for it in pattern_obj:
                pills.append(str(it))
    except Exception:
        pass
    return pills

# =========================
# ---------- MAIN ---------
# =========================
def run_live_minute_analysis(df: pd.DataFrame):
    st.set_page_config(page_title="Analisi Live Minuto — ProTrader", layout="wide")
    _inject_css()
    st.title("⏱️ Analisi Live — ProTrader Suite")

    # =======================
    # TAB: SETUP & QUOTE
    # =======================
    tab_setup, tab_ev, tab_camp, tab_team, tab_signals = st.tabs(
        ["🎛️ Setup", "🧠 EV Advisor", "🏆 Campionato (stesso stato)", "📈 Squadra focus", "🧩 Segnali"]
    )

    with tab_setup:
        champ_options = sorted(df["country"].dropna().astype(str).unique())
        champ_default = st.session_state.get("campionato_corrente", champ_options[0] if champ_options else "N/A")
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

        st.session_state["quota_home"] = float(odd_home)
        st.session_state["quota_draw"] = float(odd_draw)
        st.session_state["quota_away"] = float(odd_away)

        c_live = st.columns([2,1,1,1])
        with c_live[0]:
            current_min = st.slider("⏲️ Minuto attuale", 1, 120, int(st.session_state.get("minlive",45)), key="minlive")
        with c_live[1]:
            live_score_txt = st.text_input("📟 Risultato live", str(st.session_state.get("scorelive","0-0")), key="scorelive")
        with c_live[2]:
            commission = st.number_input("💸 Commissione exchange", 0.0, 0.10, 0.045, step=0.005)
        with c_live[3]:
            show_ext = st.toggle("🔎 Usa segnali esterni", value=True)

        parsed = safe_parse_score(live_score_txt)
        if not parsed:
            st.error("Formato risultato non valido (esempio: 1-1).")
            return
        live_h, live_a = parsed

        label_live = label_match({"Odd home": odd_home, "Odd Away": odd_away})
        st.markdown(
            f"<span class='badge'>🔖 <b>{label_live}</b></span> "
            f"<span class='badge'>⏱️ <b>{current_min}'</b></span> "
            f"<span class='badge'>📟 <b>{live_h}-{live_a}</b></span> "
            f"<span class='badge small'>campionato <b>{champ}</b></span>",
            unsafe_allow_html=True
        )

        with st.expander("⚙️ Quote mercati Goal/BTTS (per EV)", expanded=False):
            oc1, oc2, oc3, oc4, oc5 = st.columns(5)
            with oc1: q_over05 = st.number_input("Over 0.5", 1.01, 50.0, 1.30, step=0.01)
            with oc2: q_over15 = st.number_input("Over 1.5", 1.01, 50.0, 1.65, step=0.01)
            with oc3: q_over25 = st.number_input("Over 2.5", 1.01, 50.0, 2.40, step=0.01)
            with oc4: q_over35 = st.number_input("Over 3.5", 1.01, 50.0, 3.75, step=0.01)
            with oc5: q_btts   = st.number_input("BTTS (GG)", 1.01, 50.0, 2.10, step=0.01)

        # Write-back verso Pre-Match
        for k,v in [("ov05",q_over05),("ov15",q_over15),("ov25",q_over25),("ov35",q_over35),("btts",q_btts)]:
            _set_shared_quote(k, v)

        # Persist per uso cross-tab
        st.session_state["_live_ctx"] = {
            "champ": champ, "home": home_team, "away": away_team,
            "odd_home": odd_home, "odd_draw": odd_draw, "odd_away": odd_away,
            "lay_home": lay_home, "lay_draw": lay_draw, "lay_away": lay_away,
            "minute": current_min, "score": (live_h, live_a),
            "commission": commission, "label": label_live,
            "q_over": {0.5:q_over05,1.5:q_over15,2.5:q_over25,3.5:q_over35}, "q_btts": q_btts,
            "show_ext": show_ext
        }

    # =======================
    # PRE-CALCOLI COMUNI
    # =======================
    if "_live_ctx" not in st.session_state:
        st.stop()
    ctx = st.session_state["_live_ctx"]
    champ, home_team, away_team = ctx["champ"], ctx["home"], ctx["away"]
    odd_home, odd_draw, odd_away = ctx["odd_home"], ctx["odd_draw"], ctx["odd_away"]
    lay_home, lay_draw, lay_away = ctx["lay_home"], ctx["lay_draw"], ctx["lay_away"]
    current_min, (live_h, live_a) = ctx["minute"], ctx["score"]
    commission, label_live = ctx["commission"], ctx["label"]
    q_map = ctx["q_over"]; q_over05,q_over15,q_over25,q_over35 = q_map[0.5],q_map[1.5],q_map[2.5],q_map[3.5]
    q_btts, show_ext = ctx["q_btts"], ctx["show_ext"]

    df = df.copy()
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)
    df_league = df[(df["country"]==champ) & (df["Label"]==label_live)].copy()

    df_matched   = _matches_matching_state(df_league, current_min, live_h, live_a)
    df_home_side = _matches_matching_state(df_league[df_league["Home"]==home_team], current_min, live_h, live_a)
    df_away_side = _matches_matching_state(df_league[df_league["Away"]==away_team], current_min, live_h, live_a)

    st.caption(f"✅ Campione: {len(df_matched)} | {sample_badge(len(df_matched))} • Team focus: {home_team} / {away_team}")

    # Probabilità 1X2 (blend campionato + subset squadra)
    pH_L,pD_L,pA_L = _result_probs(df_matched)
    pH_H, pD_H, _   = _result_probs(df_home_side)
    _,    pD_A, pA_A= _result_probs(df_away_side)
    p_home = _blend(pH_L, len(df_matched), pH_H, len(df_home_side))
    p_away = _blend(pA_L, len(df_matched), pA_A, len(df_away_side))
    p_draw_side = _blend(pD_H, len(df_home_side), pD_A, len(df_away_side))
    p_draw = _blend(pD_L, len(df_matched), p_draw_side, len(df_home_side)+len(df_away_side))
    s = p_home + p_draw + p_away
    if s>0: p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s

    # Over/BTTS (blend)
    over_lines = [0.5, 1.5, 2.5, 3.5]
    probs_over = {}
    for line in over_lines:
        pL = _over_prob(df_matched,    live_h, live_a, line)
        pH = _over_prob(df_home_side,  live_h, live_a, line)
        pA = _over_prob(df_away_side,  live_h, live_a, line)
        side_n = len(df_home_side)+len(df_away_side)
        probs_over[line] = _blend(pL, len(df_matched), (pH+pA)/2 if side_n>0 else pL, side_n)
    p_btts_L = _btts_prob(df_matched)
    p_btts_side = _blend(_btts_prob(df_home_side), len(df_home_side), _btts_prob(df_away_side), len(df_away_side))
    p_btts = _blend(p_btts_L, len(df_matched), p_btts_side, len(df_home_side)+len(df_away_side))

    # Priors lega/label
    priors = league_priors(df_league, live_h, live_a, over_lines)

    # =======================
    # EV ADVISOR (AI)
    # =======================
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
                "Prob %": round(p*100,1), "Fair": round(fair,2), "Edge": edge,
                "EV": ev_b, "EV %": round(ev_b*100,1),
                "½-Kelly %": round((kelly*50)*100,1) if kelly is not None else None,
                "Campione": quality, "Δ vs prior": round((p-prior_p)*100,1),
                "AI score": round(ai_score,1)
            })
        # 1X2 Back/Lay
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

    with tab_ev:
        st.subheader("EV Advisor — ranking opportunità")
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
        if only_pos: view = view[view["EV"] > 0]
        view = view[view["EV %"] >= thr]
        view = view[view["Campione"] >= min_samp]
        view = view.sort_values(order, ascending=False).reset_index(drop=True)

        st.dataframe(
            _style_table(view[["Mercato","Tipo","Quota","Prob %","Fair","Edge","EV","EV %","½-Kelly %","Campione","Δ vs prior","AI score"]]),
            use_container_width=True, height=420
        )

        # ⬇️ CSV snapshot
        st.download_button(
            "⬇️ Esporta ranking EV (CSV)",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="ev_advisor_snapshot.csv",
            mime="text/csv"
        )

        # Hedging rapido per top Back
        top_back = view[(view["Tipo"]=="Back") & (view["EV"]>0)].head(3).to_dict(orient="records")
        if top_back:
            with st.expander("🛡️ Hedging rapido (green-up su quota target)"):
                st.caption("Back a quota **ob** con stake **B** → Lay a quota **ol** con **L = (B·ob)/ol** (profitto uguale su esiti).")
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
                        st.write(f"**Profit atteso** ≈ {profit:.2f}")

        # Pills top idea
        top = view.head(3).to_dict(orient="records")
        if top:
            pills = []
            for r in top:
                pill = badge_ev(r["EV"])
                kelly_txt = f" • ½-Kelly≈{r['½-Kelly %']:.1f}%" if r["½-Kelly %"] is not None else ""
                pills.append(f"<span class='ev-pill'><b>{r['Mercato']} {r['Tipo']}</b> {pill} — q={r['Quota']:.2f}{kelly_txt}</span>")
            st.markdown(" ".join(pills), unsafe_allow_html=True)

        colK1,colK2,colK3,colK4 = st.columns(4)
        colK1.metric("Sample Campionato", len(df_matched), help=sample_badge(len(df_matched)))
        colK2.metric("Sample Home", len(df_home_side), help=sample_badge(len(df_home_side)))
        colK3.metric("Sample Away", len(df_away_side), help=sample_badge(len(df_away_side)))
        colK4.metric("P(Over 0.5)→FT", f"{probs_over[0.5]*100:.1f}%")

    # =======================
    # CAMPIONATO: TABS PRO
    # =======================
    with tab_camp:
        st.subheader("🏆 Campionato — stesso label & stato live")
        t1, t2, t3, t4 = st.tabs(["Esiti 1X2", "Over / EV", "Post-minuto", "CS / Hedge"])

        # --- Esiti 1X2
        with t1:
            if len(df_matched):
                d = pd.DataFrame([
                    {"Esito":"1 (Home)","Prob %":p_home*100,"Fair":1/max(p_home,1e-9),
                     "Back q":odd_home,"EV Back":ev_back(p_home,odd_home,commission),
                     "Lay q":lay_home,"EV Lay":ev_lay(p_home,lay_home,commission),
                     "½-Kelly %": kelly_fraction(p_home,odd_home)*50*100},
                    {"Esito":"X (Draw)","Prob %":p_draw*100,"Fair":1/max(p_draw,1e-9),
                     "Back q":odd_draw,"EV Back":ev_back(p_draw,odd_draw,commission),
                     "Lay q":lay_draw,"EV Lay":ev_lay(p_draw,lay_draw,commission),
                     "½-Kelly %": kelly_fraction(p_draw,odd_draw)*50*100},
                    {"Esito":"2 (Away)","Prob %":p_away*100,"Fair":1/max(p_away,1e-9),
                     "Back q":odd_away,"EV Back":ev_back(p_away,odd_away,commission),
                     "Lay q":lay_away,"EV Lay":ev_lay(p_away,lay_away,commission),
                     "½-Kelly %": kelly_fraction(p_away,odd_away)*50*100},
                ])
                st.dataframe(_style_table(d), use_container_width=True)
            else:
                st.info("Nessun match nel campione.")

        # --- Over / EV
        with t2:
            if len(df_matched):
                rows=[]
                for line, q in {0.5:q_over05,1.5:q_over15,2.5:q_over25,3.5:q_over35}.items():
                    p = probs_over[line]; fair = 1/max(p,1e-9)
                    rows.append({"Mercato":f"Over {line}","Prob %":p*100,"Fair":fair,"Quota":q,
                                 "EV Back":ev_back(p,q,commission),"EV Lay":ev_lay(p,q,commission),
                                 "½-Kelly %":kelly_fraction(p,q)*50*100})
                rows.append({"Mercato":"BTTS (GG)","Prob %":p_btts*100,"Fair":1/max(p_btts,1e-9),"Quota":q_btts,
                             "EV Back":ev_back(p_btts,q_btts,commission),"EV Lay":ev_lay(p_btts,q_btts,commission),
                             "½-Kelly %":kelly_fraction(p_btts,q_btts)*50*100})
                df_over = pd.DataFrame(rows)
                st.dataframe(_style_table(df_over), use_container_width=True)
            else:
                st.info("Nessun match nel campione.")

        # --- Post-minuto
        with t3:
            if len(df_matched):
                st.dataframe(compute_post_minute_stats(df_matched, current_min), use_container_width=True)
            else:
                st.info("Nessun match per analisi post-minuto.")

        # --- CS / Hedge
        with t4:
            if len(df_matched):
                lam_for, lam_against = estimate_remaining_lambdas(df_matched, current_min, True)
                top_cs = final_cs_distribution(lam_for, lam_against, live_h, live_a, max_goals_delta=6)[:6]
                st.write("**Top Correct Score (probabilità)**")
                st.table(pd.DataFrame([{"CS": k, "Prob %": v*100} for k, v in top_cs]).style.format({"Prob %":"{:.2f}"}))

                st.markdown("**Coperture CS (manuale quote)**")
                c1,c2,c3 = st.columns(3)
                with c1:
                    cs1 = st.text_input("CS #1", value=top_cs[0][0] if top_cs else "1-1")
                    q_cs1 = st.number_input("Quota CS #1", 1.01, 200.0, 6.0, step=0.01)
                with c2:
                    cs2 = st.text_input("CS #2", value=top_cs[1][0] if len(top_cs)>1 else "2-1")
                    q_cs2 = st.number_input("Quota CS #2", 1.01, 200.0, 9.0, step=0.01)
                with c3:
                    cs3 = st.text_input("CS #3", value=top_cs[2][0] if len(top_cs)>2 else "1-2")
                    q_cs3 = st.number_input("Quota CS #3", 1.01, 200.0, 10.0, step=0.01)

                def prob_from_list(target, pairs):
                    for k, v in pairs:
                        if k == target: return v
                    return 0.0

                rows_cs=[]
                for cs, q in [(cs1,q_cs1),(cs2,q_cs2),(cs3,q_cs3)]:
                    p = prob_from_list(cs, top_cs)
                    rows_cs.append({"CS":cs, "Prob %":p*100, "Quota":q, "EV Back":ev_back(p,q,commission)})
                st.dataframe(_style_table(pd.DataFrame(rows_cs)), use_container_width=True)
            else:
                st.info("Nessun match per stimare CS.")

    # =======================
    # SQUADRA: TABS PRO
    # =======================
    with tab_team:
        st.subheader(f"📈 Squadra — {home_team} (Home) / {away_team} (Away)")
        t1, t2, t3, t4 = st.tabs(["Esiti 1X2", "Over / EV", "Post-minuto", "CS / Hedge"])

        df_team_focus = df_home_side if p_home >= p_away else df_away_side
        team_name = home_team if p_home >= p_away else away_team

        with t1:
            if len(df_team_focus):
                if team_name == home_team:
                    win  = (df_team_focus["Home Goal FT"] > df_team_focus["Away Goal FT"]).mean()
                    draw = (df_team_focus["Home Goal FT"] == df_team_focus["Away Goal FT"]).mean()
                    lose = (df_team_focus["Home Goal FT"] < df_team_focus["Away Goal FT"]).mean()
                else:
                    win  = (df_team_focus["Away Goal FT"] > df_team_focus["Home Goal FT"]).mean()
                    draw = (df_team_focus["Away Goal FT"] == df_team_focus["Home Goal FT"]).mean()
                    lose = (df_team_focus["Away Goal FT"] < df_team_focus["Home Goal FT"]).mean()
                s = max(1e-9, win+draw+lose)
                win,draw,lose = win/s, draw/s, lose/s
                d = pd.DataFrame([
                    {"Esito":f"{team_name} Win","Prob %":win*100,"Fair":1/max(win,1e-9),
                     "Back q":(odd_home if team_name==home_team else odd_away),
                     "EV Back":ev_back(win,(odd_home if team_name==home_team else odd_away),commission)},
                    {"Esito":"Draw","Prob %":draw*100,"Fair":1/max(draw,1e-9),
                     "Back q":odd_draw,"EV Back":ev_back(draw,odd_draw,commission)},
                    {"Esito":f"{'Opp.' if team_name==home_team else home_team} Win","Prob %":lose*100,"Fair":1/max(lose,1e-9),
                     "Back q":(odd_away if team_name==home_team else odd_home),
                     "EV Back":ev_back(lose,(odd_away if team_name==home_team else odd_home),commission)},
                ])
                st.dataframe(_style_table(d), use_container_width=True)
                st.caption(f"Campione squadra: {len(df_team_focus)} ({sample_badge(len(df_team_focus))})")
            else:
                st.info("Nessun match squadra con questo stato.")

        with t2:
            if len(df_team_focus):
                rows=[]
                for line in [0.5,1.5,2.5,3.5]:
                    p = _over_prob(df_team_focus, live_h, live_a, line)
                    rows.append({"Mercato":f"Over {line}","Prob %":p*100,"Fair":1/max(p,1e-9),
                                 "Quota":q_map[line],"EV Back":ev_back(p,q_map[line],commission),
                                 "EV Lay":ev_lay(p,q_map[line],commission),
                                 "½-Kelly %":kelly_fraction(p,q_map[line])*50*100})
                pT = _btts_prob(df_team_focus)
                rows.append({"Mercato":"BTTS (GG)","Prob %":pT*100,"Fair":1/max(pT,1e-9),
                             "Quota":q_btts,"EV Back":ev_back(pT,q_btts,commission),
                             "EV Lay":ev_lay(pT,q_btts,commission),
                             "½-Kelly %":kelly_fraction(pT,q_btts)*50*100})
                st.dataframe(_style_table(pd.DataFrame(rows)), use_container_width=True)
            else:
                st.info("Nessun match squadra per Over/EV.")

        with t3:
            if len(df_team_focus):
                st.dataframe(compute_post_minute_stats(df_team_focus, current_min), use_container_width=True)
            else:
                st.info("Nessun match squadra per post-minuto.")

        with t4:
            if len(df_team_focus):
                lam_for_T, lam_against_T = estimate_remaining_lambdas(df_team_focus, current_min, team_name==home_team)
                top_cs_T = final_cs_distribution(lam_for_T, lam_against_T, live_h, live_a, max_goals_delta=6)[:6]
                st.write("**Top Correct Score (squadra focus)**")
                st.table(pd.DataFrame([{"CS": k, "Prob %": v*100} for k, v in top_cs_T]).style.format({"Prob %":"{:.2f}"}))
            else:
                st.info("Nessun match squadra per stimare CS.")

    # =======================
    # SEGNALI ESTERNI (UI)
    # =======================
    with tab_signals:
        st.subheader("🧩 Segnali esterni (pattern, macro KPI, bias lega)")
        if show_ext:
            ext = get_external_signals(df_league, home_team, away_team)
            has_any = ext.get("macro_home") or ext.get("macro_away") or ext.get("pattern_signals") or ext.get("macros_bias")
            if has_any:
                # Macro KPI cards
                if ext.get("macro_home") or ext.get("macro_away"):
                    st.markdown("**Macro KPI (estratto)**")
                    c1, c2 = st.columns(2)
                    with c1:
                        _card_macro(f"Home — {home_team}", ext.get("macro_home", {}))
                    with c2:
                        _card_macro(f"Away — {away_team}", ext.get("macro_away", {}))

                # Pattern → pills
                if ext.get("pattern_signals"):
                    st.markdown("**Pattern live**")
                    pills = _pills_from_patterns(ext["pattern_signals"])
                    if pills:
                        st.markdown(
                            "<div class='pills'>" + " ".join([f"<span class='pill good'>{p}</span>" for p in pills]) + "</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.caption("Nessun pattern forte rilevato.")

                # Bias lega (macros)
                if ext.get("macros_bias"):
                    st.markdown("**Bias lega (macros)**")
                    bias = ext["macros_bias"]
                    try:
                        df_bias = pd.DataFrame([bias])
                        st.dataframe(df_bias, use_container_width=True)
                    except Exception:
                        st.json(bias)

                if ext.get("notes"):
                    st.caption(" • ".join(ext["notes"]))
            else:
                st.caption("Nessun segnale esterno disponibile.")
        else:
            st.caption("Segnali esterni disattivati nel Setup.")
