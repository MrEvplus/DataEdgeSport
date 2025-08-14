import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# =========================================================
# Utilities
# =========================================================
def _ensure_str_with_unknown(s: pd.Series, fill="") -> pd.Series:
    return s.fillna(fill).astype(str)

def _today_year():
    try:
        return datetime.today().year
    except Exception:
        return 2025

# Intervalli cronologici usati in tutti i grafici
def timeframes():
    return [(0,15),(16,30),(31,45),(46,60),(61,75),(76,120)]

# --------------------------------------------------------
# Riconoscere se la riga ha info di match giocato
# --------------------------------------------------------
def is_match_played(row):
    # se ho minuti espliciti basta questo
    if pd.notna(row.get("minuti goal segnato home", "")) and str(row.get("minuti goal segnato home")).strip():
        return True
    if pd.notna(row.get("minuti goal segnato away", "")) and str(row.get("minuti goal segnato away")).strip():
        return True

    # fallback: se ho i gol FT, considero match giocato
    gh = row.get("Home Goal FT", None)
    ga = row.get("Away Goal FT", None)
    return (pd.notna(gh) and pd.notna(ga))

# --------------------------------------------------------
# Parse minuti "12;38;90+2" -> [12,38,92] (gestione +)
# --------------------------------------------------------
def parse_goal_times(val) -> list[int]:
    if pd.isna(val) or str(val).strip() == "":
        return []
    out = []
    for p in str(val).split(";"):
        p = p.strip()
        if p == "":
            continue
        if "+" in p:
            try:
                base, add = p.split("+", 1)
                out.append(int(base) + int(add))
            except Exception:
                continue
        else:
            try:
                out.append(int(p))
            except Exception:
                continue
    return out

# =========================================================
# Macro KPI (per venue)
# =========================================================
def compute_team_macro_stats(df: pd.DataFrame, team: str, venue: str) -> dict:
    if venue == "Home":
        data = df[df["Home"] == team]
        gf_col, ga_col = "Home Goal FT", "Away Goal FT"
    else:
        data = df[df["Away"] == team]
        gf_col, ga_col = "Away Goal FT", "Home Goal FT"

    data = data[data.apply(is_match_played, axis=1)]
    n = len(data)
    if n == 0:
        return {}

    if venue == "Home":
        w = (data["Home Goal FT"] > data["Away Goal FT"]).sum()
        d = (data["Home Goal FT"] == data["Away Goal FT"]).sum()
        l = (data["Home Goal FT"] < data["Away Goal FT"]).sum()
    else:
        w = (data["Away Goal FT"] > data["Home Goal FT"]).sum()
        d = (data["Away Goal FT"] == data["Home Goal FT"]).sum()
        l = (data["Away Goal FT"] < data["Home Goal FT"]).sum()

    btts = ((data["Home Goal FT"] > 0) & (data["Away Goal FT"] > 0)).mean() * 100

    return {
        "Matches": n,
        "Win %": round(w/n*100, 2),
        "Draw %": round(d/n*100, 2),
        "Loss %": round(l/n*100, 2),
        "Avg Goals Scored": round(data[gf_col].mean(), 2),
        "Avg Goals Conceded": round(data[ga_col].mean(), 2),
        "BTTS %": round(btts, 2),
    }

# =========================================================
# Goal patterns + distribuzione per time-frame
# =========================================================
def _goals_to_timeframes(mins: list[int]) -> dict:
    bins = {f"{a}-{b}":0 for a,b in timeframes()}
    for m in mins:
        for a,b in timeframes():
            if a <= m <= b:
                bins[f"{a}-{b}"] += 1
                break
    return bins

def compute_goal_patterns(df_team: pd.DataFrame, venue: str):
    """
    Ritorna:
      patterns: dict di percentuali (Win/Draw/Loss, 1-0, 0-1, ecc. + 0-0)
      tf_scored, tf_conceded: contatori per fascia minuti
    """
    if venue == "Home":
        gf_ft, ga_ft = "Home Goal FT", "Away Goal FT"
        mins_for  = df_team["minuti goal segnato home"].apply(parse_goal_times)
        mins_against = df_team["minuti goal segnato away"].apply(parse_goal_times)
    else:
        gf_ft, ga_ft = "Away Goal FT", "Home Goal FT"
        mins_for  = df_team["minuti goal segnato away"].apply(parse_goal_times)
        mins_against = df_team["minuti goal segnato home"].apply(parse_goal_times)

    n = len(df_team)
    if n == 0:
        # chiavi minime per non rompere la UI
        base = {"P":0,"Win %":0,"Draw %":0,"Loss %":0,"0-0 %":0,"2+ Goals %":0,
                "H 1st %":0,"D 1st %":0,"A 1st %":0,"H 2nd %":0,"D 2nd %":0,"A 2nd %":0}
        for a,b in timeframes():
            base[f"{a}-{b} Goals %"] = 0
        return base, {f"{a}-{b}":0 for a,b in timeframes()}, {f"{a}-{b}":0 for a,b in timeframes()}

    # Esiti FT
    w = (df_team[gf_ft] > df_team[ga_ft]).sum()
    d = (df_team[gf_ft] == df_team[ga_ft]).sum()
    l = (df_team[gf_ft] < df_team[ga_ft]).sum()
    zero_zero = ((df_team[gf_ft] == 0) & (df_team[ga_ft] == 0)).sum()

    # Primo/secondo tempo: proxy con gol 1T e FT se disponibili
    # Se non esistono le colonne 1T, non rompiamo
    h1, a1 = df_team.get("Home Goal 1T"), df_team.get("Away Goal 1T")
    if venue == "Home" and h1 is not None and a1 is not None:
        h1w = (h1 > a1).sum(); h1d = (h1 == a1).sum(); h1l = (h1 < a1).sum()
        s2w = ((df_team[gf_ft]-h1) > (df_team[ga_ft]-a1)).sum()
        s2d = ((df_team[gf_ft]-h1) == (df_team[ga_ft]-a1)).sum()
        s2l = ((df_team[gf_ft]-h1) < (df_team[ga_ft]-a1)).sum()
    elif venue == "Away" and h1 is not None and a1 is not None:
        h1w = (a1 < h1).sum(); h1d = (a1 == h1).sum(); h1l = (a1 > h1).sum()
        s2w = ((df_team[gf_ft]-a1) > (df_team[ga_ft]-h1)).sum()
        s2d = ((df_team[gf_ft]-a1) == (df_team[ga_ft]-h1)).sum()
        s2l = ((df_team[gf_ft]-a1) < (df_team[ga_ft]-h1)).sum()
    else:
        h1w=h1d=h1l=s2w=s2d=s2l = 0

    # Distribuzione minuti
    tf_scored = {f"{a}-{b}":0 for a,b in timeframes()}
    tf_conceded = {f"{a}-{b}":0 for a,b in timeframes()}
    for mins in mins_for:
        dct = _goals_to_timeframes(mins if mins else [])
        for k,v in dct.items(): tf_scored[k]+=v
    for mins in mins_against:
        dct = _goals_to_timeframes(mins if mins else [])
        for k,v in dct.items(): tf_conceded[k]+=v

    # % su time-frames
    tot_for = sum(tf_scored.values()) or 1
    tot_ag  = sum(tf_conceded.values()) or 1
    tf_scored_pct   = {k: round(v/tot_for*100,2) for k,v in tf_scored.items()}
    tf_conceded_pct = {k: round(v/tot_ag*100,2)  for k,v in tf_conceded.items()}

    patterns = {
        "P": n,
        "Win %": round(w/n*100,2),
        "Draw %": round(d/n*100,2),
        "Loss %": round(l/n*100,2),
        "0-0 %": round(zero_zero/n*100,2),
        "H 1st %": round(h1w/n*100,2),
        "D 1st %": round(h1d/n*100,2),
        "A 1st %": round(h1l/n*100,2),
        "H 2nd %": round(s2w/n*100,2),
        "D 2nd %": round(s2d/n*100,2),
        "A 2nd %": round(s2l/n*100,2),
        "2+ Goals %": round(((df_team[gf_ft]+df_team[ga_ft])>=2).mean()*100,2),
    }
    for a,b in timeframes():
        k = f"{a}-{b} Goals %"
        patterns[k] = tf_scored_pct[f"{a}-{b}"]  # mostriamo % segnati per fascia

    return patterns, tf_scored_pct, tf_conceded_pct

# Grafico a barre (grouped) per % nei time-frame
def plot_timeframe_goals(tf_scored_pct: dict, tf_conceded_pct: dict, team: str):
    data = []
    order = [f"{a}-{b}" for a,b in timeframes()]
    for k in order:
        data.append({"Intervallo":k,"Tipo":"Segnati","%":tf_scored_pct.get(k,0.0)})
        data.append({"Intervallo":k,"Tipo":"Subiti","%":tf_conceded_pct.get(k,0.0)})

    df_tf = pd.DataFrame(data)
    ch = alt.Chart(df_tf).mark_bar().encode(
        x=alt.X("Intervallo:N", sort=order, title="Minuti"),
        y=alt.Y("%:Q", title="Percentuale"),
        color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Segnati","Subiti"], range=["#22c55e","#ef4444"])),
        xOffset="Tipo:N",
        tooltip=["Tipo","Intervallo","%"]
    ).properties(height=280, title=f"Distribuzione goal per intervalli — {team}")
    return ch

# =========================================================
# UI di dettaglio
# =========================================================
def _macro_card(title: str, stats: dict):
    cols = st.columns(7)
    cols[0].metric("Match", stats.get("Matches",0))
    cols[1].metric("Win %", stats.get("Win %",0))
    cols[2].metric("Draw %", stats.get("Draw %",0))
    cols[3].metric("Loss %", stats.get("Loss %",0))
    cols[4].metric("Avg GF", stats.get("Avg Goals Scored",0))
    cols[5].metric("Avg GA", stats.get("Avg Goals Conceded",0))
    cols[6].metric("BTTS %", stats.get("BTTS %",0))

def _patterns_table(title: str, patterns: dict):
    st.markdown(f"#### {title}")
    dfp = pd.DataFrame([patterns])
    # spostiamo 0-0 vicino a Win/Draw/Loss
    cols_order = [c for c in ["P","Win %","Draw %","Loss %","0-0 %","2+ Goals %",
                              "H 1st %","D 1st %","A 1st %","H 2nd %","D 2nd %","A 2nd %"]
                  if c in dfp.columns]
    tf_cols = [c for c in dfp.columns if c.endswith("Goals %")]
    dfp = dfp[cols_order + tf_cols]
    st.dataframe(dfp.style.format({c:"{:.2f}%" for c in dfp.columns if c!="P"}), use_container_width=True)

# =========================================================
# Corpo pagina/tab
# =========================================================
def _render_setup_and_body(
    df: pd.DataFrame,
    db_selected: str,
    is_embedded: bool = False,
    squadra_casa: str | None = None,
    squadra_ospite: str | None = None,
):
    if "country" not in df.columns:
        st.error("Colonna 'country' mancante."); st.stop()
    if "Home" not in df.columns or "Away" not in df.columns:
        st.error("Colonne 'Home'/'Away' mancanti."); st.stop()

    df = df.copy()
    df["country"] = _ensure_str_with_unknown(df["country"]).str.upper().str.strip()
    db_selected = (db_selected or "").upper().strip()

    # filtro campionato
    if db_selected not in df["country"].unique():
        st.warning(f"Il campionato '{db_selected}' non è presente."); st.stop()
    df = df[df["country"] == db_selected].copy()
    df["Home"] = _ensure_str_with_unknown(df["Home"])
    df["Away"] = _ensure_str_with_unknown(df["Away"])

    # --- unico filtro stagioni per questa sezione (default = stagione corrente/max)
    if "Stagione" not in df.columns:
        st.error("Colonna 'Stagione' mancante."); st.stop()

    seasons_av = sorted(df["Stagione"].dropna().unique().tolist(), reverse=True)
    if not seasons_av:
        st.warning("Nessuna stagione disponibile per questa lega."); st.stop()

    # default: stagione più recente (corrente)
    default_sel = [seasons_av[0]]
    with st.expander("🧰 Filtro stagioni (solo per questa sezione)", expanded=True):
        seasons_sel = st.multiselect(
            "Seleziona le stagioni (se vuoto → tutte):",
            options=seasons_av,
            default=default_sel,
            key=f"teamstats_seasons_{db_selected}"
        )
    if seasons_sel:
        df = df[df["Stagione"].isin(seasons_sel)]

    # --- scelta squadre
    teams = sorted(set(df["Home"]).union(set(df["Away"])))
    if not teams:
        st.warning("Nessuna squadra trovata nel campionato selezionato."); st.stop()

    if is_embedded:
        # se arrivano dal pre_match li uso come default
        default_home = squadra_casa if squadra_casa in teams else teams[0]
        default_away = squadra_ospite if (squadra_ospite in teams and squadra_ospite != default_home) else ""
    else:
        default_home = teams[0]; default_away = ""

    c1, c2 = st.columns(2)
    with c1:
        home_team = st.selectbox("Squadra 1", teams, index=teams.index(default_home))
    with c2:
        away_team = st.selectbox("Squadra 2 (facoltativa)", [""]+teams,
                                 index=([""]+teams).index(default_away) if default_away else 0)

    # ------------------- macro e pattern home
    st.subheader(f"✅ Statistiche Macro — {home_team}")
    _macro_card(home_team, compute_team_macro_stats(df, home_team, "Home"))
    pat_h, tfh_sc, tfh_con = compute_goal_patterns(df[df["Home"] == home_team], "Home")
    _patterns_table(f"Goal pattern — {home_team} (Home)", pat_h)
    st.altair_chart(plot_timeframe_goals(tfh_sc, tfh_con, home_team), use_container_width=True)

    if away_team and away_team != home_team:
        st.subheader(f"✅ Statistiche Macro — {away_team}")
        _macro_card(away_team, compute_team_macro_stats(df, away_team, "Away"))
        pat_a, tfa_sc, tfa_con = compute_goal_patterns(df[df["Away"] == away_team], "Away")
        _patterns_table(f"Goal pattern — {away_team} (Away)", pat_a)
        st.altair_chart(plot_timeframe_goals(tfa_sc, tfa_con, away_team), use_container_width=True)

# =========================================================
# Entry points
# =========================================================
def run_team_stats(df: pd.DataFrame, db_selected: str):
    st.header("📊 Statistiche per Squadre")
    _render_setup_and_body(df, db_selected, is_embedded=False)

def render_team_stats_tab(df_league_all: pd.DataFrame, league_code: str, squadra_casa: str, squadra_ospite: str):
    """Wrapper usato dal pre_match (tab Statistiche Squadre)."""
    _render_setup_and_body(
        df=df_league_all,
        db_selected=league_code,
        is_embedded=True,
        squadra_casa=squadra_casa,
        squadra_ospite=squadra_ospite,
    )
