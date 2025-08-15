# analisi_live_minuto.py — v4.9 (fix indent, Over prob, team focus post-minuto)
import re
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import streamlit as st

from utils import label_match, get_global_source_df
from minutes import unify_goal_minute_columns, parse_goal_times

_SHARED_PREFIX = "prematch:shared:"
GLOBAL_CHAMP_KEY   = "global_country"
GLOBAL_SEASONS_KEY = "global_seasons"

def _shared_key(name: str) -> str:
    return f"{_SHARED_PREFIX}{name}"

def _set_shared_quote(name: str, value: float):
    st.session_state[_shared_key(name)] = float(value)

_BASE_CSS = """
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
.badge {display:inline-flex; align-items:center; gap:.5rem; padding:.25rem .6rem; border-radius:999px; font-size:.82rem; border:1px solid #374151; background:#111827; color:#e5e7eb;}
.badge b {color:#fff;}
.small {color:#9ca3af; font-size:.85rem;}
</style>
"""
def _inject_css(): st.markdown(_BASE_CSS, unsafe_allow_html=True)

def safe_parse_score(txt: str):
    if not isinstance(txt, str): return None
    cleaned = txt.replace(" ", "").replace(":", "-").replace("–", "-").replace("—", "-")
    parts = cleaned.split("-")
    if len(parts) != 2: return None
    try: return int(parts[0]), int(parts[1])
    except: return None

def _goals_up_to(series, minute):
    s = series if isinstance(series, str) else str(series or "")
    mins = parse_goal_times(s)
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
    if df is None or df.empty: return (1/3,1/3,1/3)
    h = (df["Home Goal FT"] > df["Away Goal FT"]).mean()
    d = (df["Home Goal FT"] == df["Away Goal FT"]).mean()
    a = (df["Home Goal FT"] < df["Away Goal FT"]).mean()
    h = 0 if np.isnan(h) else float(h)
    d = 0 if np.isnan(d) else float(d)
    a = 0 if np.isnan(a) else float(a)
    s = h + d + a
    return (1/3,1/3,1/3) if s<=0 else (h/s, d/s, a/s)

def _btts_prob(df):
    if df is None or df.empty: return 0.5
    val = ((df["Home Goal FT"] > 0) & (df["Away Goal FT"] > 0)).mean()
    return 0 if np.isnan(val) else float(val)

def _over_prob(df, current_h, current_a, threshold):
    """
    Probabilità che il TOTALE FINALE superi 'threshold', condizionato allo stato live corrente.
    Confronta i gol EXTRA con quanti gol mancano alla linea.
    """
    if df is None or df.empty: return 0.5
    extra = (df["Home Goal FT"] + df["Away Goal FT"]) - (current_h + current_a)
    need = max(0.0, float(threshold) - float(current_h + current_a))
    val = (extra > need).mean()
    return 0 if np.isnan(val) else float(val)

def _blend(p_main, n_main, p_side, n_side, clamp=200):
    n_main = max(int(n_main), 0)
    n_side = max(int(n_side), 0)
    if n_main + n_side == 0: return p_main
    n_main = min(n_main, clamp); n_side = min(n_side, clamp//2)
    return (n_main*p_main + n_side*p_side) / (n_main+n_side)

def sample_badge(n: int) -> str:
    if n < 30:  return "🔴 Campione piccolo"
    if n < 100: return "🟡 Campione medio"
    return "🟢 Campione robusto"

def _league_cache_key(champ: str, label: str) -> str:
    return f"cache:league_norm:{champ}:{label}"

def get_df_league_cached(df: pd.DataFrame, champ: str, label: str) -> pd.DataFrame:
    k = _league_cache_key(champ, label)
    if k in st.session_state: return st.session_state[k]
    sub = df[(df["country"].astype(str) == str(champ)) & (df["Label"].astype(str) == str(label))].copy()
    try: sub = unify_goal_minute_columns(sub)
    except Exception: pass
    st.session_state[k] = sub
    return sub

def _match_cache_key(champ: str, label: str, minute: int, score: tuple, home: str, away: str) -> str:
    s = f"{score[0]}-{score[1]}"
    return f"cache:matched:{champ}:{label}:{minute}:{s}:{home}:{away}"

def get_matched_cached(df_league: pd.DataFrame, current_min: int, live_h: int, live_a: int,
                       home_team: str, away_team: str, champ: str, label: str):
    k = _match_cache_key(champ, label, current_min, (live_h, live_a), home_team, away_team)
    if k in st.session_state: return st.session_state[k]
    df_matched   = _matches_matching_state(df_league, current_min, live_h, live_a)
    df_home_side = _matches_matching_state(df_league[df_league["Home"]==home_team], current_min, live_h, live_a)
    df_away_side = _matches_matching_state(df_league[df_league["Away"]==away_team], current_min, live_h, live_a)
    st.session_state[k] = (df_matched, df_home_side, df_away_side)
    return st.session_state[k]

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
    o = max(1.01, float(odds)); p = float(prob)
    return max(0.0, (o*p - (1-p)) / (o-1))

def league_priors(df_league, current_h, current_a, over_lines):
    pH_L, pD_L, pA_L = _result_probs(df_league)
    priors = {"1": pH_L, "X": pD_L, "2": pA_L, "BTTS": _btts_prob(df_league)}
    for line in over_lines:
        priors[f"Over {line}"] = _over_prob(df_league, current_h, current_a, line)
    return priors

def get_external_signals(df_league, home_team, away_team):
    out = {"notes": []}
    try:
        from squadre import compute_team_macro_stats
        m_home = compute_team_macro_stats(df_league, home_team, "Home")
        m_away = compute_team_macro_stats(df_league, away_team, "Away")
        out["macro_home"] = m_home; out["macro_away"] = m_away
        if m_home or m_away: out["notes"].append("Macro KPI caricati")
    except Exception:
        pass
    try:
        import pattern_analysis as pa
        if hasattr(pa, "live_signals"):
            sig = pa.live_signals(df_league, home_team, away_team)
            out["pattern_signals"] = sig
            out["notes"].append("Pattern live rilevati" if sig else "Nessun pattern forte")
    except Exception:
        pass
    try:
        import macros as m
        if hasattr(m, "league_bias"):
            bias = m.league_bias(df_league)
            out["macros_bias"] = bias
            out["notes"].append("Bias lega integrato (macros)")
    except Exception:
        pass
    return out

def compute_post_minute_stats(df, current_min):
    tf_bands = [(0,15),(16,30),(31,45),(46,60),(61,75),(76,90)]
    tf_labels = [f"{a}-{b}" for a,b in tf_bands]
    rec = {lbl: {"GF":0,"GS":0,"1+":0,"2+":0,"N":0} for lbl in tf_labels}
    for _, r in df.iterrows():
        mh = parse_goal_times(r.get("minuti goal segnato home",""))
        ma = parse_goal_times(r.get("minuti goal segnato away",""))
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

def poisson_pmf(k: int, lam: float) -> float:
    try: return math.exp(-lam) * (lam ** k) / math.factorial(k)
    except OverflowError: return 0.0

def final_cs_distribution(lam_home_add: float, lam_away_add: float, cur_h: int, cur_a: int, max_goals_delta: int = 6):
    probs = {}
    for x in range(0, max_goals_delta + 1):
        for y in range(0, max_goals_delta + 1):
            p = poisson_pmf(x, lam_home_add) * poisson_pmf(y, lam_away_add)
            cs = f"{cur_h + x}-{cur_a + y}"
            probs[cs] = probs.get(cs, 0.0) + p
    s = sum(probs.values())
    if s > 0:
        for k in probs: probs[k] /= s
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)

def estimate_remaining_lambdas(df: pd.DataFrame, current_min: int, focus_home: bool):
    tf = compute_post_minute_stats(df, current_min)
    total_gf = tf["GF"].sum(); total_gs = tf["GS"].sum()
    n = len(df) if len(df) > 0 else 1
    lam_for = total_gf / max(1, n); lam_against = total_gs / max(1, n)
    lam_for = 0.5 * lam_for + 0.5 * 0.6
    lam_against = 0.5 * lam_against + 0.5 * 0.6
    return max(0.01, lam_for), max(0.01, lam_against)

def _style_table(df_):
    def _fmt_fair(v):
        try:
            fv = float(v)
            return "∞" if fv > 1e6 else f"{fv:.2f}"
        except Exception:
            return v
    fmt_map = {}
    if "Quota" in df_.columns: fmt_map["Quota"] = "{:.2f}"
    if "Fair"  in df_.columns: fmt_map["Fair"]  = _fmt_fair
    if "EV"    in df_.columns: fmt_map["EV"]    = "{:.3f}"
    if "Edge"  in df_.columns: fmt_map["Edge"]  = "{:.3f}"
    if "½-Kelly %" in df_.columns: fmt_map["½-Kelly %"] = "{:.1f}%"
    for col in df_.columns:
        if str(col).endswith("%") and col not in fmt_map:
            fmt_map[col] = "{:.1f}%"
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
        if c in df_.columns: sty = sty.apply(_bg_posneg, subset=[c])
    return sty

def _pct_str(x):
    try: return f"{float(x):.2f}%"
    except Exception:
        s = str(x); return s if s.endswith("%") else (f"{s}%" if s not in ("", "None") else "N/D")

def _safe_num(x):
    try: return f"{float(x):.2f}"
    except Exception: return "N/D"

def _card_macro(title: str, stats: dict, light=True):
    if not isinstance(stats, dict): stats = {}
    win  = _pct_str(stats.get("Win %", stats.get("Win%", stats.get("Win", ""))))
    draw = _pct_str(stats.get("Draw %", stats.get("Draw%", stats.get("Draw", ""))))
    loss = _pct_str(stats.get("Loss %", stats.get("Loss%", stats.get("Loss", ""))))
    avg_for = _safe_num(stats.get("Avg Goals Scored", stats.get("GF avg", stats.get("GF", ""))))
    avg_ag  = _safe_num(stats.get("Avg Goals Conceded", stats.get("GA avg", stats.get("GA", ""))))
    btts    = _pct_str(stats.get("BTTS %", stats.get("BTTS%", stats.get("BTTS", ""))))
    mp      = stats.get("Matches Played", stats.get("Matches", "N/D"))
    cls = "card light" if light else "card"
    st.markdown(f"""
    <div class="{cls}">
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
    pills = []
    try:
        if isinstance(pattern_obj, dict):
            for k, v in pattern_obj.items():
                label = f"{k}: {v}"
                cls = "pill"
                try:
                    score = float(v)
                    if score >= 0.15: cls = "pill good"
                    elif score >= 0.05: cls = "pill warn"
                except Exception:
                    pass
                pills.append((label, cls))
        elif isinstance(pattern_obj, list):
            for it in pattern_obj:
                pills.append((str(it), "pill"))
        else:
            pills.append((str(pattern_obj), "pill"))
    except Exception:
        pass
    return pills

def _pattern_effects(pattern_obj):
    eff = defaultdict(float)
    if not isinstance(pattern_obj, (dict, list)): return eff
    def add(markets, val):
        for m in markets: eff[m] += val
    items = []
    if isinstance(pattern_obj, dict): items = list(pattern_obj.items())
    elif isinstance(pattern_obj, list): items = [(str(x), 0.05) for x in pattern_obj]
    for k, v in items:
        s = str(k).lower()
        try: val = float(v)
        except Exception: val = 0.05
        val = max(-0.20, min(0.20, val))
        if any(w in s for w in ["over","late goal","second half","attack","shots","pressure"]):
            add(["Over 0.5","Over 1.5","Over 2.5","Over 3.5","BTTS (GG)"], 0.5*val)
        if "btts" in s: add(["BTTS (GG)"], 0.7*val)
        if "home" in s and any(w in s for w in ["pressure","attack","form","momentum"]):
            add(["1 (Home)"], 0.6*val)
        if "away" in s and any(w in s for w in ["pressure","attack","form","momentum"]):
            add(["2 (Away)"], 0.6*val)
        if any(w in s for w in ["draw","balanced","equilibrium"]):
            add(["X (Draw)"], 0.5*val)
    return eff

def _open_help_btn():
    if st.button("📘 Guida rapida — EV Advisor", help="Apri un pop-up con formule, definizioni e spiegazioni del Breakdown 1X2."):
        st.session_state["show_help_ev"] = True

def _render_modal_if_needed():
    if not st.session_state.get("show_help_ev"): return
    st.markdown("<div class='badge'>Guida rapida aperta</div>", unsafe_allow_html=True)
    if st.button("Chiudi", key="close_help_ev"): st.session_state["show_help_ev"] = False

def _legend_badges():
    st.markdown(
        """
        <div style="display:flex; flex-wrap:wrap; gap:.45rem; margin:.25rem 0 .5rem;">
          <span class="badge"><b>Fair</b></span>
          <span class="badge"><b>Edge</b></span>
          <span class="badge"><b>EV</b></span>
          <span class="badge"><b>½-Kelly</b></span>
          <span class="badge"><b>Δ vs prior</b></span>
          <span class="badge"><b>AI score</b></span>
        </div>
        """, unsafe_allow_html=True
    )

def run_live_minute_analysis(df: pd.DataFrame | None = None):
    st.set_page_config(page_title="Analisi Live Minuto — ProTrader", layout="wide")
    _inject_css()
    st.title("⏱️ Analisi Live — ProTrader Suite")

    # Fonte globale se df mancante/vuoto
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        gdf, meta = get_global_source_df()
        if isinstance(gdf, pd.DataFrame) and not gdf.empty:
            df = gdf.copy()

    if df is None or df.empty:
        st.warning("Nessun dataset disponibile. Seleziona campionato & stagioni dalla sidebar dell’Hub.")
        return

    # --- Applica filtri GLOBALI con fallback (non svuotare il df se mismatch) ---
    champ_global   = st.session_state.get(GLOBAL_CHAMP_KEY)
    seasons_global = st.session_state.get(GLOBAL_SEASONS_KEY)

    if champ_global and "country" in df.columns:
        df_tmp = df[df["country"].astype(str) == str(champ_global)]
        if not df_tmp.empty:
            df = df_tmp  # applico solo se produce righe

    if seasons_global:
        if "Stagione" in df.columns:
            df_tmp = df[df["Stagione"].astype(str).isin([str(s) for s in seasons_global])]
            if not df_tmp.empty:
                df = df_tmp
        elif "sezonul" in df.columns:
            df_tmp = df[df["sezonul"].astype(str).isin([str(s) for s in seasons_global])]
            if not df_tmp.empty:
                df = df_tmp

    # Label & normalizzazione minuti-gol
    if "Label" not in df.columns:
        df = df.copy()
        df["Label"] = df.apply(label_match, axis=1)
    try:
        df = unify_goal_minute_columns(df)
    except Exception:
        pass

    import altair as alt  # noqa: F401
    tab_setup, tab_ev, tab_camp, tab_team, tab_signals = st.tabs(
        ["🎛️ Setup", "🧠 EV Advisor", "🏆 Campionato (stesso stato)", "📈 Squadra focus", "🧩 Segnali"]
    )

    # ---------- SETUP ----------
    with tab_setup:
        if "country" not in df.columns:
            st.error("Il dataset non contiene la colonna 'country'."); return

        # opzioni campionato robuste
        camp_series = df["country"].astype(str)
        champ_options = sorted([x for x in camp_series.dropna().unique() if x and x.lower() != "nan"])

        # Fallback se lista vuota
        if not champ_options:
            fallback = (
                st.session_state.get("campionato_corrente")
                or (str(camp_series.iloc[0]) if len(camp_series) else "N/A")
            )
            champ_options = [fallback]

        champ_default = (
            st.session_state.get("campionato_corrente")
            or st.session_state.get(GLOBAL_CHAMP_KEY)
            or (champ_options[0] if champ_options else "N/A")
        )

        col0,col1 = st.columns([1.2,2])
        with col0:
            champ = st.selectbox(
                "🏆 Campionato",
                champ_options,
                index=champ_options.index(champ_default) if champ_default in champ_options else 0,
                key="champ_live",
                help="Seleziona il campionato da analizzare. Filtreremo lo storico a campionato+label."
            )
        with col1:
            if "Home" not in df.columns or "Away" not in df.columns:
                st.error("Il dataset non ha colonne 'Home' / 'Away'."); return
            teams_home = sorted(df["Home"].dropna().astype(str).unique())
            teams_away = sorted(df["Away"].dropna().astype(str).unique())
            c1,c2 = st.columns(2)
            with c1:
                home_team = st.selectbox("🏠 Casa", teams_home, key="home_live")
            with c2:
                away_team = st.selectbox("🚪 Trasferta", teams_away, key="away_live")

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
            st.error("Formato risultato non valido (esempio: 1-1)."); return
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

        for k,v in [("ov05",q_over05),("ov15",q_over15),("ov25",q_over25),("ov35",q_over35),("btts",q_btts)]:
            _set_shared_quote(k, v)

        st.session_state["_live_ctx"] = {
            "champ": champ, "home": home_team, "away": away_team,
            "odd_home": odd_home, "odd_draw": odd_draw, "odd_away": odd_away,
            "lay_home": lay_home, "lay_draw": lay_draw, "lay_away": lay_away,
            "minute": current_min, "score": (live_h, live_a),
            "commission": commission, "label": label_live,
            "q_over": {0.5:q_over05,1.5:q_over15,2.5:q_over25,3.5:q_over35}, "q_btts": q_btts,
            "show_ext": show_ext
        }

    if "_live_ctx" not in st.session_state: st.stop()
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

    st.session_state["_label_live"] = label_live

    df_league = get_df_league_cached(df, champ, label_live)
    df_matched, df_home_side, df_away_side = get_matched_cached(
        df_league, current_min, live_h, live_a, home_team, away_team, champ, label_live
    )

    st.caption(f"✅ Campione: {len(df_matched)} | {sample_badge(len(df_matched))} • Team focus: {home_team} / {away_team}")

    pH_L,pD_L,pA_L = _result_probs(df_matched)
    pH_H, pD_H, _   = _result_probs(df_home_side)
    _,    pD_A, pA_A= _result_probs(df_away_side)
    p_home = _blend(pH_L, len(df_matched), pH_H, len(df_home_side))
    p_away = _blend(pA_L, len(df_matched), pA_A, len(df_away_side))
    p_draw_side = _blend(pD_H, len(df_home_side), pD_A, len(df_away_side))
    p_draw = _blend(pD_L, len(df_matched), p_draw_side, len(df_home_side)+len(df_away_side))
    s = p_home + p_draw + p_away
    if s>0: p_home, p_draw, p_away = p_home/s, p_draw/s, p_away/s

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

    priors = league_priors(df_league, live_h, live_a, over_lines)

    def _ev_rows():
        rows = []
        def add_row(market, kind, price, p, prior_p):
            ev_b = ev_back(p, price, commission) if kind=="Back" else ev_lay(p, price, commission)
            fair = 1/max(p,1e-9); edge = (fair - price)/fair
            kelly = kelly_fraction(p, price) if kind=="Back" else None
            quality = len(df_matched); delta = abs(p - prior_p)
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
        add_row("1 (Home)", "Back", odd_home, p_home, priors["1"])
        add_row("X (Draw)", "Back", odd_draw, p_draw, priors["X"])
        add_row("2 (Away)", "Back", odd_away, p_away, priors["2"])
        add_row("1 (Home)", "Lay",  lay_home, p_home, priors["1"])
        add_row("X (Draw)", "Lay",  lay_draw, p_draw, priors["X"])
        add_row("2 (Away)", "Lay",  lay_away, p_away, priors["2"])
        for line, q in {0.5:q_over05, 1.5:q_over15, 2.5:q_over25, 3.5:q_over35}.items():
            add_row(f"Over {line}", "Back", q, probs_over[line], priors[f"Over {line}"])
            add_row(f"Over {line}", "Lay",  q, probs_over[line], priors[f"Over {line}"])
        add_row("BTTS (GG)", "Back", q_btts, p_btts, priors["BTTS"])
        add_row("BTTS (GG)", "Lay",  q_btts, p_btts, priors["BTTS"])
        return pd.DataFrame(rows)
    df_ev_full = _ev_rows()

    ext = get_external_signals(df_league, home_team, away_team) if show_ext else {}
    pat_eff = {}
    if show_ext and ext.get("pattern_signals"):
        # opzionale: converti pattern in boost/malus per l'AI score
        pat_eff = defaultdict(float)  # placeholder no-op

    # ---------- EV ----------
    with tab_ev:
        st.subheader("EV Advisor — ranking opportunità")
        st.caption(f"Contesto: **{champ} / {label_live}**, minuto **{current_min}'**, score **{live_h}-{live_a}** · campione **{len(df_matched)}**.")
        _open_help_btn(); _render_modal_if_needed()
        only_pos = st.checkbox("Solo EV+", value=True)
        thr = st.number_input("Soglia EV% min", -20.0, 20.0, 0.0, step=0.5)
        min_samp = st.number_input("Min campione", 0, 500, 30, step=10)
        order = st.selectbox("Ordina per", ["EV", "AI score", "Edge", "½-Kelly %"], index=0)
        view = df_ev_full.copy()
        if only_pos: view = view[view["EV"] > 0]
        view = view[view["EV %"] >= thr]
        view = view[view["Campione"] >= min_samp]
        view = view.sort_values(order, ascending=False).reset_index(drop=True)
        _legend_badges()
        cols_show = ["Mercato","Tipo","Quota","Prob %","Fair","Edge","EV","EV %","½-Kelly %","Campione","Δ vs prior","AI score"]
        st.dataframe(_style_table(view[cols_show]), use_container_width=True, height=430)
        st.download_button("⬇️ Esporta ranking EV (CSV)", data=view.to_csv(index=False).encode("utf-8"),
                           file_name="ev_advisor_snapshot.csv", mime="text/csv")

    # ---------- Campionato ----------
    with tab_camp:
        st.subheader("🏆 Campionato — stesso label & stato live")
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
            st.dataframe(compute_post_minute_stats(df_matched, current_min), use_container_width=True)
        else:
            st.info("Nessun match nel campione con questo stato.")

    # ---------- Squadra focus ----------
    with tab_team:
        st.subheader(f"📈 Squadra — {home_team} (Home) / {away_team} (Away)")
        # Scegliamo la squadra focus in base alla probabilità 1X2 stimata nello stato live
        focus_is_home = bool(p_home >= p_away)
        df_team_focus = df_home_side if focus_is_home else df_away_side
        team_name = home_team if focus_is_home else away_team
        role = "Home" if focus_is_home else "Away"

        if len(df_team_focus):
            # Badge esplicito su quale squadra stiamo usando e perché
            st.markdown(
                f"<span class='badge'>🎯 <b>Team focus:</b> {team_name} ({role})</span> "
                f"<span class='badge small'>criterio: p({role}) ≥ p({'Away' if focus_is_home else 'Home'})</span>",
                unsafe_allow_html=True
            )

            # Probabilità Over e quote eque (sotto-campione della squadra focus)
            rows = []
            for line in [0.5, 1.5, 2.5, 3.5]:
                p = _over_prob(df_team_focus, live_h, live_a, line)
                rows.append({"Mercato": f"Over {line}", "Prob %": p * 100, "Fair": 1 / max(p, 1e-9)})

            st.caption(f"Campione squadra: {len(df_team_focus)} ({sample_badge(len(df_team_focus))})")
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            # Post-minuto dedicata alla squadra focus
            st.write("**Post-minuto (squadra focus)**")
            st.dataframe(compute_post_minute_stats(df_team_focus, current_min), use_container_width=True)

            # Top Correct Score (squadra focus)
            lam_for_T, lam_against_T = estimate_remaining_lambdas(df_team_focus, current_min, focus_is_home)
            top_cs_T = final_cs_distribution(lam_for_T, lam_against_T, live_h, live_a, max_goals_delta=6)[:6]
            st.write("**Top Correct Score (squadra focus)**")
            st.table(pd.DataFrame([{"CS": k, "Prob %": v * 100} for k, v in top_cs_T]).style.format({"Prob %": "{:.2f}"}))
        else:
            st.info("Nessun match squadra con questo stato.")

    # ---------- Segnali ----------
    with tab_signals:
        st.subheader("🧩 Segnali esterni")
        st.caption("I segnali non alterano l’EV ma possono guidare la lettura.")
        st.info("Se abilitati nel Setup, verranno mostrati qui (macro KPI, pattern, bias).")

def run_live_minute_panel(df: pd.DataFrame | None = None):
    if df is None or (isinstance(df, pd.DataFrame) and df.empty):
        gdf, _ = get_global_source_df()
        df = gdf if isinstance(gdf, pd.DataFrame) else pd.DataFrame()
    return run_live_minute_analysis(df)

# alias utili per app.py
run_live_minuto_analysis  = run_live_minute_analysis
run_live_minuto           = run_live_minute_analysis
run_live                  = run_live_minute_analysis
def main(df: pd.DataFrame | None = None): return run_live_minute_analysis(df)
