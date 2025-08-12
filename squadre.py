from __future__ import annotations

import re
from datetime import date
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


# =========================================================
# Helpers generali
# =========================================================
def _ensure_str_with_unknown(s: pd.Series, default: str = "Unknown") -> pd.Series:
    if s is None:
        return pd.Series(dtype="string")
    try:
        if pd.api.types.is_categorical_dtype(s.dtype):
            s = s.cat.add_categories([default]).fillna(default)
            s = s.astype("string")
        else:
            s = s.astype("string").fillna(default)
    except Exception:
        s = s.astype("string").fillna(default)
    return s.replace("", default)

def _coerce_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _season_sort_key(s: str) -> int:
    if not isinstance(s, str):
        s = str(s)
    yrs = [int(x) for x in re.findall(r"\d{4}", s)]
    return max(yrs) if yrs else -1

def _seasons_desc(unique_seasons: list) -> list[str]:
    arr = [str(x) for x in unique_seasons if pd.notna(x)]
    return sorted(arr, key=_season_sort_key, reverse=True)

def _limit_last_n(df_in: pd.DataFrame, n: int) -> pd.DataFrame:
    if n and n > 0 and "Data" in df_in.columns:
        s = pd.to_datetime(df_in["Data"], errors="coerce")
        tmp = df_in.copy()
        tmp["_data_"] = s
        tmp = tmp.sort_values("_data_", ascending=False).drop(columns=["_data_"])
        return tmp.head(n)
    return df_in

def _quality_label(n: int) -> str:
    if n >= 50:
        return "ALTO"
    if n >= 20:
        return "MEDIO"
    return "BASSO"

def _pct(x: float, y: float) -> float:
    return round((x / y) * 100, 2) if y else 0.0


# =========================================================
# Quote condivise (sincronizzate con pre_match)
# =========================================================
_SHARED_PREFIX = "prematch:shared:"

def _shared_key(name: str) -> str:
    return f"{_SHARED_PREFIX}{name}"

def _init_shared_quotes():
    defaults = {"ov15": 2.00, "ov25": 2.00, "ov35": 2.00, "btts": 2.00}
    for k, v in defaults.items():
        st.session_state.setdefault(_shared_key(k), v)

def _get_shared_quotes() -> dict:
    _init_shared_quotes()
    return {
        "ov15": float(st.session_state[_shared_key("ov15")]),
        "ov25": float(st.session_state[_shared_key("ov25")]),
        "ov35": float(st.session_state[_shared_key("ov35")]),
        "btts": float(st.session_state[_shared_key("btts")]),
    }


# =========================================================
# Entry points
# =========================================================
def run_team_stats(df: pd.DataFrame, db_selected: str):
    st.header("📊 Statistiche per Squadre")
    _render_setup_and_body(df, db_selected, is_embedded=False)

def render_team_stats_tab(
    df_league_all: pd.DataFrame,
    league_code: str,
    squadra_casa: str,
    squadra_ospite: str,
):
    _render_setup_and_body(
        df=df_league_all,
        db_selected=league_code,
        is_embedded=True,
        squadra_casa=squadra_casa,
        squadra_ospite=squadra_ospite,
    )


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
        st.error("Colonne 'Home' e/o 'Away' mancanti."); st.stop()

    _init_shared_quotes()

    df = df.copy()
    df["country"] = _ensure_str_with_unknown(df["country"], "Unknown").str.strip().str.upper()
    db_selected = (db_selected or "").strip().upper()

    if db_selected not in df["country"].unique():
        st.warning(f"⚠️ Il campionato selezionato '{db_selected}' non è presente nel database.")
        st.stop()

    df = df[df["country"] == db_selected].copy()
    df["Home"] = _ensure_str_with_unknown(df["Home"], "")
    df["Away"] = _ensure_str_with_unknown(df["Away"], "")

    if "Stagione" not in df.columns:
        st.error("Colonna 'Stagione' mancante."); st.stop()

    seasons_desc = _seasons_desc(df["Stagione"].dropna().unique().tolist())
    latest = seasons_desc[0] if seasons_desc else None

    with st.expander("⚙️ Filtro stagioni (solo per questa sezione)", expanded=True):
        seasons_selected = st.multiselect(
            "Scegli le stagioni da includere (se vuoto = tutte)",
            options=seasons_desc,
            default=[latest] if latest else [],
            key="teams:seasons_manual",
        )
        if seasons_selected:
            st.caption(f"Stagioni attive (Squadre): **{', '.join(seasons_selected)}**")
            df = df[df["Stagione"].astype("string").isin(seasons_selected)].copy()
        else:
            st.caption("Stagioni attive (Squadre): **Tutte**")

    if not is_embedded:
        teams_available = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))
        c1, c2 = st.columns(2)
        with c1: squadra_casa   = st.selectbox("Squadra Casa", options=teams_available, key="teams:home")
        with c2: squadra_ospite = st.selectbox("Squadra Ospite", options=[""] + teams_available, key="teams:away")
    else:
        if not squadra_casa or not squadra_ospite:
            st.info("Seleziona entrambe le squadre nella parte alta della pagina pre-match."); return
        if squadra_casa == squadra_ospite:
            st.warning("Casa e Ospite sono uguali."); return

    # KPI macro rapidi
    st.subheader("📌 KPI Macro (contesto corretto)")
    st.caption("Metriche in **Home@Casa** e **Away@Trasferta**. BTTS% = partite con gol di entrambe.")
    stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
    stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")
    if not stats_home or not stats_away:
        st.info("⚠️ Dati insufficienti nelle stagioni selezionate."); return
    k1, k2, k3, k4 = st.columns(4)
    with k1: st.metric(f"{squadra_casa} — P", stats_home.get("Matches Played", 0))
    with k2: st.metric(f"{squadra_casa} — Win%", f"{stats_home.get('Win %', 0):.1f}%")
    with k3: st.metric(f"{squadra_casa} — GF", f"{stats_home.get('Avg Goals Scored', 0):.2f}")
    with k4: st.metric(f"{squadra_casa} — BTTS", f"{stats_home.get('BTTS %', 0):.1f}%")
    d1, d2, d3, d4 = st.columns(4)
    with d1: st.metric(f"{squadra_ospite} — P", stats_away.get("Matches Played", 0))
    with d2: st.metric(f"{squadra_ospite} — Win%", f"{stats_away.get('Win %', 0):.1f}%")
    with d3: st.metric(f"{squadra_ospite} — GF", f"{stats_away.get('Avg Goals Scored', 0):.2f}")
    with d4: st.metric(f"{squadra_ospite} — BTTS", f"{stats_away.get('BTTS %', 0):.1f}%")

    # === BOARD compatta stile screenshot ===
    render_compact_match_board(df, db_selected, squadra_casa, squadra_ospite)
    st.divider()

    # === GOAL TIMES estesi + PATTERNS e grafici ===
    st.subheader("🕒 Goal Times — distribuzione e dettagli")

    with st.expander(f"Dettaglio {squadra_casa} (Home)", expanded=True):
        render_goal_times_table(df, squadra_casa, venue="Home")
    with st.expander(f"Dettaglio {squadra_ospite} (Away)", expanded=True):
        render_goal_times_table(df, squadra_ospite, venue="Away")

    st.subheader("🎯 Goal patterns e distribuzione per fasce minuto")
    df_home = df[df["Home"].astype("string") == squadra_casa].copy()
    if not df_home.empty:
        mask_played_home = df_home.apply(is_match_played, axis=1)
        df_home = df_home[mask_played_home]
    total_home = len(df_home)

    df_away = df[df["Away"].astype("string") == squadra_ospite].copy()
    if not df_away.empty:
        mask_played_away = df_away.apply(is_match_played, axis=1)
        df_away = df_away[mask_played_away]
    total_away = len(df_away)

    if total_home > 0:
        patterns_home, tf_scored_home, tf_conceded_home = compute_goal_patterns(df_home, "Home", total_home)
        tf_scored_home_pct = _tf_to_pct(tf_scored_home)
        tf_conceded_home_pct = _tf_to_pct(tf_conceded_home)
    else:
        patterns_home, tf_scored_home, tf_conceded_home = {}, {}, {}
        tf_scored_home_pct, tf_conceded_home_pct = {}, {}

    if total_away > 0:
        patterns_away, tf_scored_away, tf_conceded_away = compute_goal_patterns(df_away, "Away", total_away)
        tf_scored_away_pct = _tf_to_pct(tf_scored_away)
        tf_conceded_away_pct = _tf_to_pct(tf_conceded_away)
    else:
        patterns_away, tf_scored_away, tf_conceded_away = {}, {}, {}
        tf_scored_away_pct, tf_conceded_away_pct = {}, {}

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**{squadra_casa} (Home)**")
        if patterns_home:
            st.markdown(_build_goal_pattern_html(patterns_home, squadra_casa, "#16a34a"), unsafe_allow_html=True)
        else:
            st.info("Dati insufficienti.")
    with col2:
        st.markdown(f"**{squadra_ospite} (Away)**")
        if patterns_away:
            st.markdown(_build_goal_pattern_html(patterns_away, squadra_ospite, "#dc2626"), unsafe_allow_html=True)
        else:
            st.info("Dati insufficienti.")
    with col3:
        if patterns_home or patterns_away:
            total = compute_goal_patterns_total(patterns_home or {}, patterns_away or {}, total_home, total_away)
            st.markdown("**Totale**")
            st.markdown(
                _build_goal_pattern_html({k: total.get(k, 0) for k in goal_pattern_keys_without_tf()},
                                         "Totale", "#2563eb"),
                unsafe_allow_html=True
            )
        else:
            st.info("—")

    if patterns_home:
        st.markdown(f"**Distribuzione Goal – {squadra_casa} (Home)**")
        ch = plot_timeframe_goals(tf_scored_home, tf_conceded_home, tf_scored_home_pct, tf_conceded_home_pct, squadra_casa)
        st.altair_chart(ch, use_container_width=True)
    if patterns_away:
        st.markdown(f"**Distribuzione Goal – {squadra_ospite} (Away)**")
        ca = plot_timeframe_goals(tf_scored_away, tf_conceded_away, tf_scored_away_pct, tf_conceded_away_pct, squadra_ospite)
        st.altair_chart(ca, use_container_width=True)


# =========================================================
# Board compatta (stile screenshot)
# =========================================================
def _cell_bar(pct: float, color: str = "#16a34a") -> str:
    width = max(0, min(100, float(pct)))
    return f"""
    <div style="display:flex;flex-direction:column;gap:4px;align-items:center;">
      <div style="height:12px;width:100%;background:#e5e7eb;border-radius:6px;overflow:hidden;">
        <div style="height:12px;width:{width}%;background:{color};"></div>
      </div>
      <div style="font-size:11px">{pct:.0f}%</div>
    </div>"""

def _cell_dual_bar(gf: float, ga: float) -> str:
    tot = max(gf + ga, 0.0001)
    w_g = (gf / tot) * 100.0
    w_a = (ga / tot) * 100.0
    return f"""
    <div style="display:flex;flex-direction:column;gap:4px;align-items:center;">
      <div style="height:12px;width:100%;background:#e5e7eb;border-radius:6px;overflow:hidden;display:flex">
        <div style="height:12px;width:{w_g:.1f}%;background:#16a34a;"></div>
        <div style="height:12px;width:{w_a:.1f}%;background:#dc2626;"></div>
      </div>
      <div style="font-size:11px">GF {gf:.2f} · GA {ga:.2f}</div>
    </div>"""

def _compute_board_kpis(df_ctx: pd.DataFrame, venue: str) -> dict:
    if df_ctx.empty:
        return dict(P=0, Z0=0, O15=0, O25=0, O35=0,
                    SC=0, CC=0, GF=0.0, GA=0.0, BTTS=0, FH1=0, FH2=0, SH1=0, SH2=0)

    df = df_ctx.copy()
    h = pd.to_numeric(df["Home Goal FT"], errors="coerce").fillna(0).astype(int)
    a = pd.to_numeric(df["Away Goal FT"], errors="coerce").fillna(0).astype(int)
    tot = (h + a).astype(int)

    if "Home Goal 1T" in df.columns and "Away Goal 1T" in df.columns:
        h1 = pd.to_numeric(df["Home Goal 1T"], errors="coerce").fillna(0).astype(int)
        a1 = pd.to_numeric(df["Away Goal 1T"], errors="coerce").fillna(0).astype(int)
    else:
        h1 = pd.Series([0]*len(df), index=df.index)
        a1 = pd.Series([0]*len(df), index=df.index)
    h2 = (h - h1).clip(lower=0)
    a2 = (a - a1).clip(lower=0)

    P   = int(len(df))
    Z0  = _pct(((h == 0) & (a == 0)).sum(), P)
    O15 = _pct((tot > 1).sum(), P)
    O25 = _pct((tot > 2).sum(), P)
    O35 = _pct((tot > 3).sum(), P)
    BT  = _pct(((h > 0) & (a > 0)).sum(), P)

    if venue == "Home":
        SC = _pct((h > 0).sum(), P)   # Scored in match (%)
        CC = _pct((a > 0).sum(), P)   # Conceded in match (%)
        GF = float(h.mean()); GA = float(a.mean())
    else:
        SC = _pct((a > 0).sum(), P)
        CC = _pct((h > 0).sum(), P)
        GF = float(a.mean()); GA = float(h.mean())

    FH1 = _pct(((h1 + a1) >= 1).sum(), P)
    FH2 = _pct(((h1 + a1) >= 2).sum(), P)
    SH1 = _pct(((h2 + a2) >= 1).sum(), P)
    SH2 = _pct(((h2 + a2) >= 2).sum(), P)

    return dict(P=P, Z0=Z0, O15=O15, O25=O25, O35=O35,
                SC=SC, CC=CC, GF=GF, GA=GA, BTTS=BT,
                FH1=FH1, FH2=FH2, SH1=SH1, SH2=SH2)

def _fh_sh_combined_prob(df_home_ctx: pd.DataFrame, df_away_ctx: pd.DataFrame) -> dict:
    """Percentuali 'team scored in half' e probabilità combinata di almeno un gol nella frazione."""
    def team_scored_half(df_ctx, half="FH"):
        if df_ctx.empty: return 0.0
        h = _coerce_num(df_ctx["Home Goal 1T"]).fillna(0).astype(int) if "Home Goal 1T" in df_ctx.columns else pd.Series([0]*len(df_ctx))
        a = _coerce_num(df_ctx["Away Goal 1T"]).fillna(0).astype(int) if "Away Goal 1T" in df_ctx.columns else pd.Series([0]*len(df_ctx))
        if half == "FH":
            tot = h + a
        else:
            hft = _coerce_num(df_ctx["Home Goal FT"]).fillna(0).astype(int)
            aft = _coerce_num(df_ctx["Away Goal FT"]).fillna(0).astype(int)
            tot = (hft - h).clip(lower=0) + (aft - a).clip(lower=0)
        return round((tot >= 1).mean() * 100, 2)

    ph_fh = team_scored_half(df_home_ctx, "FH")
    pa_fh = team_scored_half(df_away_ctx, "FH")
    ph_sh = team_scored_half(df_home_ctx, "SH")
    pa_sh = team_scored_half(df_away_ctx, "SH")

    prob_fh = round(100 * (1 - (1 - ph_fh/100) * (1 - pa_fh/100)), 2)
    prob_sh = round(100 * (1 - (1 - ph_sh/100) * (1 - pa_sh/100)), 2)
    return dict(FH_home=ph_fh, FH_away=pa_fh, FH_prob=prob_fh,
                SH_home=ph_sh, SH_away=pa_sh, SH_prob=prob_sh)

def render_compact_match_board(df: pd.DataFrame, league_code: str, squadra_casa: str, squadra_ospite: str):
    if not all(col in df.columns for col in ["Home", "Away", "Home Goal FT", "Away Goal FT"]):
        st.info("Colonne minime non disponibili per la match board."); return

    # Context corretti
    df_home_ctx = df[df["Home"].astype("string") == squadra_casa].copy()
    if not df_home_ctx.empty:
        mask_home = df_home_ctx.apply(is_match_played, axis=1)
        df_home_ctx = df_home_ctx[mask_home]
    df_away_ctx = df[df["Away"].astype("string") == squadra_ospite].copy()
    if not df_away_ctx.empty:
        mask_away = df_away_ctx.apply(is_match_played, axis=1)
        df_away_ctx = df_away_ctx[mask_away]

    k_home = _compute_board_kpis(df_home_ctx, venue="Home")
    k_away = _compute_board_kpis(df_away_ctx, venue="Away")
    fhsh = _fh_sh_combined_prob(df_home_ctx, df_away_ctx)

    # CSS e header
    theme_css = """
    <style>
      .board { border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;margin-top:.5rem }
      .board .hdr { background:#e8f0fe;padding:8px 12px;font-weight:600;display:flex;gap:8px;align-items:center; flex-wrap:wrap }
      .grid { display:grid;grid-template-columns: 220px 60px repeat(3, 80px) 100px 100px 90px repeat(4, 80px); gap:8px; padding:12px; background:white }
      .grid .cell { background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:6px;display:flex;align-items:center;justify-content:center;min-height:54px }
      .grid .team { justify-content:flex-start;padding-left:10px;font-weight:600 }
      .grid .head { font-size:12px;font-weight:600;background:#f1f5f9 }
      .legend { display:flex;gap:14px;align-items:center;margin-left:auto;font-size:12px }
      .legend span::before { content:""; display:inline-block; width:10px;height:10px; border-radius:2px; margin-right:6px; vertical-align:middle }
      .l-sc::before { background:#16a34a }
      .l-cc::before { background:#dc2626 }
      .l-btts::before { background:#10b981 }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="board">
          <div class="hdr">
            <div>Match Board — <b>{squadra_casa}</b> vs <b>{squadra_ospite}</b> · <span style="opacity:.7">{league_code}</span></div>
            <div class="legend">
              <span class="l-sc">Scored%</span>
              <span class="l-cc">Conceded%</span>
              <span class="l-btts">BTTS%</span>
            </div>
          </div>
        """,
        unsafe_allow_html=True
    )

    header = """
      <div class="gb head">Team (context)</div>
      <div class="gb head">P</div>
      <div class="gb head">0-0</div>
      <div class="gb head">O1.5</div>
      <div class="gb head">O2.5</div>
      <div class="gb head">O3.5</div>
      <div class="gb head">Scored%</div>
      <div class="gb head">Conceded%</div>
      <div class="gb head">Average</div>
      <div class="gb head">BTTS%</div>
      <div class="gb head">1+ 1st</div>
      <div class="gb head">2+ 1st</div>
      <div class="gb head">1+ 2nd</div>
      <div class="gb head">2+ 2nd</div>
    """

    def _row(team: str, vlabel: str, k: dict, bg: str) -> str:
        return f"""
          <div class="gb cell team" style="background:{bg}">{team} <span style="opacity:.7;font-weight:400">({vlabel})</span></div>
          <div class="gb cell">{k['P']}</div>
          <div class="gb cell">{_cell_bar(k['Z0'], '#9ca3af')}</div>
          <div class="gb cell">{_cell_bar(k['O15'], '#16a34a')}</div>
          <div class="gb cell">{_cell_bar(k['O25'], '#65a30d')}</div>
          <div class="gb cell">{_cell_bar(k['O35'], '#84cc16')}</div>
          <div class="gb cell">{_cell_bar(k['SC'], '#16a34a')}</div>
          <div class="gb cell">{_cell_bar(k['CC'], '#dc2626')}</div>
          <div class="gb cell">{_cell_dual_bar(k['GF'], k['GA'])}</div>
          <div class="gb cell">{_cell_bar(k['BTTS'], '#10b981')}</div>
          <div class="gb cell">{_cell_bar(k['FH1'], '#0ea5e9')}</div>
          <div class="gb cell">{_cell_bar(k['FH2'], '#0284c7')}</div>
          <div class="gb cell">{_cell_bar(k['SH1'], '#a855f7')}</div>
          <div class="gb cell">{_cell_bar(k['SH2'], '#7c3aed')}</div>
        """

    row_home = _row(squadra_casa, "Home", k_home, "#f0fdf4")
    row_away = _row(squadra_ospite, "Away", k_away, "#fff7ed")

    st.markdown(f'<div class="grid">{header}{row_home}{row_away}</div>', unsafe_allow_html=True)

    # Riga riassuntiva FH/SH goals scored
    st.markdown(
        f"""
        <div style="padding:10px 12px;background:#f8fafc;border-top:1px solid #eef2f7">
          <b>FH goals scored:</b> Home {fhsh['FH_home']:.0f}% · Away {fhsh['FH_away']:.0f}% · Probability {fhsh['FH_prob']:.0f}% &nbsp;&nbsp;|&nbsp;&nbsp;
          <b>SH goals scored:</b> Home {fhsh['SH_home']:.0f}% · Away {fhsh['SH_away']:.0f}% · Probability {fhsh['SH_prob']:.0f}%
        </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# Goal Times table (estesa)
# =========================================================
_BINS = [(0,15), (16,30), (31,45), (46,60), (61,75), (76,90)]

def parse_goal_times(val):
    if pd.isna(val) or val == "":
        return []
    out = []
    for part in str(val).strip().split(";"):
        p = part.strip()
        if p.isdigit():
            out.append(int(p))
    return out

def is_match_played(row) -> bool:
    m_home = str(row.get("minuti goal segnato home", "") or "").strip()
    m_away = str(row.get("minuti goal segnato away", "") or "").strip()
    if m_home != "" or m_away != "":
        return True
    hg = row.get("Home Goal FT", None)
    ag = row.get("Away Goal FT", None)
    return pd.notna(hg) and pd.notna(ag)

def _minute_bin(minute: int) -> str:
    for a,b in _BINS:
        if a < minute <= b:
            return f"{a}-{b}"
    return "90+"

def _goal_times_stats(df_team: pd.DataFrame, venue: str) -> dict:
    """Restituisce distribuzioni per 'gol fatti' e 'gol subiti' per bin minuti + riepiloghi half."""
    if df_team.empty:
        return dict(P=0, gf=0, ga=0, bins_for={}, bins_against={}, gs1=0, gs2=0, gc1=0, gc2=0, g40p=0, g85p=0)

    if venue == "Home":
        mins_for   = df_team["minuti goal segnato home"].apply(parse_goal_times)
        mins_again = df_team["minuti goal segnato away"].apply(parse_goal_times)
        gf_series = _coerce_num(df_team["Home Goal FT"]).fillna(0).astype(int)
        ga_series = _coerce_num(df_team["Away Goal FT"]).fillna(0).astype(int)
        h1_for = _coerce_num(df_team.get("Home Goal 1T", 0)).fillna(0).astype(int)
        h1_agn = _coerce_num(df_team.get("Away Goal 1T", 0)).fillna(0).astype(int)
    else:
        mins_for   = df_team["minuti goal segnato away"].apply(parse_goal_times)
        mins_again = df_team["minuti goal segnato home"].apply(parse_goal_times)
        gf_series = _coerce_num(df_team["Away Goal FT"]).fillna(0).astype(int)
        ga_series = _coerce_num(df_team["Home Goal FT"]).fillna(0).astype(int)
        h1_for = _coerce_num(df_team.get("Away Goal 1T", 0)).fillna(0).astype(int)
        h1_agn = _coerce_num(df_team.get("Home Goal 1T", 0)).fillna(0).astype(int)

    bins_for = {f"{a}-{b}": 0 for a,b in _BINS}
    bins_against = {f"{a}-{b}": 0 for a,b in _BINS}
    g40p = 0; g85p = 0

    for arr in mins_for:
        for m in arr:
            if m > 40: g40p += 1
            if m > 85: g85p += 1
            key = _minute_bin(m)
            if key in bins_for: bins_for[key] += 1
    for arr in mins_again:
        for m in arr:
            key = _minute_bin(m)
            if key in bins_against: bins_against[key] += 1

    gs1 = int(h1_for.sum())
    gs2 = int(gf_series.sum() - gs1)
    gc1 = int(h1_agn.sum())
    gc2 = int(ga_series.sum() - gc1)

    return dict(
        P=int(len(df_team)),
        gf=int(gf_series.sum()),
        ga=int(ga_series.sum()),
        bins_for=bins_for,
        bins_against=bins_against,
        gs1=gs1, gs2=gs2, gc1=gc1, gc2=gc2,
        g40p=g40p, g85p=g85p
    )

def render_goal_times_table(df: pd.DataFrame, team: str, venue: str):
    df_team = df[(df["Home" if venue=="Home" else "Away"].astype("string") == team)].copy()
    if df_team.empty:
        st.info("Nessun match utile."); return
    mask = df_team.apply(is_match_played, axis=1)
    df_team = df_team[mask]

    stats = _goal_times_stats(df_team, venue)
    if stats["P"] == 0:
        st.info("Dati insufficienti."); return

    # Costruzione tabella HTML compatta
    # Colonne: P | G+ | G- | 0-15 | 16-30 | 31-45 | 46-60 | 61-75 | 76-90 | GS1 | GS2 | GC1 | GC2 | 40+ | 85+
    cols = ["P", "G+", "G-", "0-15", "16-30", "31-45", "46-60", "61-75", "76-90", "GS1", "GS2", "GC1", "GC2", "40+", "85+"]
    head = "".join([f"<th>{c}</th>" for c in cols])

    def bar_small(val, denom, color="#16a34a"):
        pct = _pct(val, denom)
        w = int(max(0, min(100, pct)))
        return f"""
        <div style="width:100%;background:#eef2f7;height:10px;border-radius:4px;overflow:hidden">
          <div style="height:10px;width:{w}%;background:{color}"></div>
        </div>
        <div style="font-size:11px;text-align:center">{pct:.0f}%</div>
        """

    row_vals = [
        str(stats["P"]),
        str(stats["gf"]),
        str(stats["ga"]),
        bar_small(stats["bins_for"]["0-15"],  max(1, stats["gf"])),
        bar_small(stats["bins_for"]["16-30"], max(1, stats["gf"])),
        bar_small(stats["bins_for"]["31-45"], max(1, stats["gf"])),
        bar_small(stats["bins_for"]["46-60"], max(1, stats["gf"])),
        bar_small(stats["bins_for"]["61-75"], max(1, stats["gf"])),
        bar_small(stats["bins_for"]["76-90"], max(1, stats["gf"])),
        str(stats["gs1"]),
        str(stats["gs2"]),
        str(stats["gc1"]),
        str(stats["gc2"]),
        bar_small(stats["g40p"], max(1, stats["gf"])),
        bar_small(stats["g85p"], max(1, stats["gf"]))
    ]
    body = "".join([f"<td style='vertical-align:middle'>{v}</td>" for v in row_vals])

    html = f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px">
      <thead>
        <tr style="background:#f1f5f9">{head}</tr>
      </thead>
      <tbody>
        <tr>{body}</tr>
      </tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# Macro stats per singola squadra
# =========================================================
def compute_team_macro_stats(df: pd.DataFrame, team: str, venue: str) -> dict:
    if venue == "Home":
        data = df[df["Home"].astype("string") == team]
        gf_col, ga_col = "Home Goal FT", "Away Goal FT"
    else:
        data = df[df["Away"].astype("string") == team]
        gf_col, ga_col = "Away Goal FT", "Home Goal FT"

    mask_played = data.apply(is_match_played, axis=1)
    data = data[mask_played]

    n = len(data)
    if n == 0:
        return {}
    if venue == "Home":
        wins = int((data["Home Goal FT"] > data["Away Goal FT"]).sum())
        draws = int((data["Home Goal FT"] == data["Away Goal FT"]).sum())
        losses= int((data["Home Goal FT"] < data["Away Goal FT"]).sum())
    else:
        wins = int((data["Away Goal FT"] > data["Home Goal FT"]).sum())
        draws = int((data["Away Goal FT"] == data["Home Goal FT"]).sum())
        losses= int((data["Away Goal FT"] < data["Home Goal FT"]).sum())
    gf = float(_coerce_num(data[gf_col]).mean())
    ga = float(_coerce_num(data[ga_col]).mean())
    btts = float(((_coerce_num(data["Home Goal FT"]) > 0) & (_coerce_num(data["Away Goal FT"]) > 0)).mean() * 100)

    return {"Matches Played": n,
            "Win %": round((wins / n) * 100, 2),
            "Draw %": round((draws / n) * 100, 2),
            "Loss %": round((losses / n) * 100, 2),
            "Avg Goals Scored": round(gf, 2),
            "Avg Goals Conceded": round(ga, 2),
            "BTTS %": round(btts, 2)}


# =========================================================
# Goal pattern computation + grafici (come prima)
# =========================================================
def build_timeline(row, venue):
    try:
        h_goals = parse_goal_times(row.get("minuti goal segnato home", ""))
        a_goals = parse_goal_times(row.get("minuti goal segnato away", ""))
        tl = [("H", m) for m in h_goals] + [("A", m) for m in a_goals]
        if tl:
            tl.sort(key=lambda x: x[1]); return tl
        hg_raw = row.get("Home Goal FT", 0); ag_raw = row.get("Away Goal FT", 0)
        hg = int(hg_raw) if pd.notna(hg_raw) else 0; ag = int(ag_raw) if pd.notna(ag_raw) else 0
        return [("H", 90)] * hg + [("A", 91)] * ag
    except Exception:
        return []

def timeframes():
    return [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 120)]

def compute_goal_patterns(df_team: pd.DataFrame, venue: str, total_matches: int):
    if total_matches == 0:
        return {key: 0 for key in goal_pattern_keys()}, {}, {}
    def pct(x):       return round((x / total_matches) * 100, 2) if total_matches > 0 else 0
    def pct_sub(x,y): return round((x / y) * 100, 2) if y > 0 else 0

    if venue == "Home":
        wins  = int((df_team["Home Goal FT"] > df_team["Away Goal FT"]).sum())
        draws = int((df_team["Home Goal FT"] == df_team["Away Goal FT"]).sum())
        losses= int((df_team["Home Goal FT"] < df_team["Away Goal FT"]).sum())
        zero_zero_count = int(((df_team["Home Goal FT"] == 0) & (df_team["Away Goal FT"] == 0)).sum())
    else:
        wins  = int((df_team["Away Goal FT"] > df_team["Home Goal FT"]).sum())
        draws = int((df_team["Away Goal FT"] == df_team["Home Goal FT"]).sum())
        losses= int((df_team["Away Goal FT"] < df_team["Home Goal FT"]).sum())
        zero_zero_count = int(((df_team["Away Goal FT"] == 0) & (df_team["Home Goal FT"] == 0)).sum())

    zero_zero_pct = pct(zero_zero_count)
    tf_scored   = {f"{a}-{b}": 0 for a, b in timeframes()}
    tf_conceded = {f"{a}-{b}": 0 for a, b in timeframes()}
    first_goal = last_goal = 0
    one_zero = one_one_after_one_zero = two_zero_after_one_zero = 0
    zero_one = one_one_after_zero_one = zero_two_after_zero_one = 0

    for _, row in df_team.iterrows():
        tl = build_timeline(row, venue)
        if not tl: continue
        first = tl[0][0]; last = tl[-1][0]
        if venue == "Home":
            if first == "H": first_goal += 1
            if last  == "H": last_goal  += 1
        else:
            if first == "A": first_goal += 1
            if last  == "A": last_goal  += 1

        score_home = score_away = 0
        for team_char, minute in tl:
            if team_char == "H": score_home += 1
            else:                score_away += 1
            for start, end in timeframes():
                if start < minute <= end:
                    if venue == "Home":
                        if team_char == "H": tf_scored[f"{start}-{end}"] += 1
                        else:                tf_conceded[f"{start}-{end}"] += 1
                    else:
                        if team_char == "A": tf_scored[f"{start}-{end}"] += 1
                        else:                tf_conceded[f"{start}-{end}"] += 1

        if venue == "Home":
            if first == "H":
                one_zero += 1; sH, sA = 1, 0
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 2 and sA == 0: two_zero_after_one_zero += 1; break
                    if sH == 1 and sA == 1: one_one_after_one_zero += 1; break
            elif first == "A":
                zero_one += 1; sH, sA = 0, 1
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 1 and sA == 1: one_one_after_zero_one += 1; break
                    if sH == 0 and sA == 2: zero_two_after_zero_one += 1; break
        else:
            if first == "H":
                one_zero += 1; sH, sA = 1, 0
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 2 and sA == 0: two_zero_after_one_zero += 1; break
                    if sH == 1 and sA == 1: one_one_after_one_zero += 1; break
            elif first == "A":
                zero_one += 1; sH, sA = 0, 1
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 1 and sA == 1: one_one_after_zero_one += 1; break
                    if sH == 0 and sA == 2: zero_two_after_zero_one += 1; break

    two_up = int((np.abs(_coerce_num(df_team["Home Goal FT"]) - _coerce_num(df_team["Away Goal FT"])) >= 2).sum())
    ht_home_win = int((_coerce_num(df_team["Home Goal 1T"]) > _coerce_num(df_team["Away Goal 1T"])).sum())
    ht_draw     = int((_coerce_num(df_team["Home Goal 1T"]) == _coerce_num(df_team["Away Goal 1T"])).sum())
    ht_away_win = int((_coerce_num(df_team["Home Goal 1T"]) < _coerce_num(df_team["Away Goal 1T"])).sum())
    sh_home_win = int(((_coerce_num(df_team["Home Goal FT"]) - _coerce_num(df_team["Home Goal 1T"])) >
                       (_coerce_num(df_team["Away Goal FT"]) - _coerce_num(df_team["Away Goal 1T"]))).sum())
    sh_draw     = int(((_coerce_num(df_team["Home Goal FT"]) - _coerce_num(df_team["Home Goal 1T"])) ==
                       (_coerce_num(df_team["Away Goal FT"]) - _coerce_num(df_team["Away Goal 1T"]))).sum())
    sh_away_win = int(((_coerce_num(df_team["Home Goal FT"]) - _coerce_num(df_team["Home Goal 1T"])) <
                       (_coerce_num(df_team["Away Goal FT"]) - _coerce_num(df_team["Away Goal 1T"]))).sum())

    tf_scored_pct   = _tf_to_pct(tf_scored)
    tf_conceded_pct = _tf_to_pct(tf_conceded)
    patterns = {
        "P": total_matches,
        "Win %": pct(wins), "Draw %": pct(draws), "Loss %": pct(losses),
        "First Goal %": pct(first_goal), "Last Goal %": pct(last_goal),
        "1-0 %": pct(one_zero), "1-1 after 1-0 %": pct_sub(one_one_after_one_zero, one_zero),
        "2-0 after 1-0 %": pct_sub(two_zero_after_one_zero, one_zero),
        "0-1 %": pct(zero_one), "1-1 after 0-1 %": pct_sub(one_one_after_zero_one, zero_one),
        "0-2 after 0-1 %": pct_sub(zero_two_after_zero_one, zero_one),
        "2+ Goals %": pct(two_up),
        "H 1st %": pct(ht_home_win), "D 1st %": pct(ht_draw), "A 1st %": pct(ht_away_win),
        "H 2nd %": pct(sh_home_win), "D 2nd %": pct(sh_draw), "A 2nd %": pct(sh_away_win),
        "0-0 %": zero_zero_pct,
    }
    return patterns, tf_scored, tf_conceded

def _tf_to_pct(tf_dict: dict[str, int]) -> dict[str, float]:
    tot = sum(tf_dict.values())
    return {k: round((v / tot) * 100, 2) if tot > 0 else 0 for k, v in tf_dict.items()}

def _build_goal_pattern_html(patterns: dict, team: str, color_hex: str) -> str:
    def bar_html(value: float, color: str, width_max: int = 90) -> str:
        width = int(width_max * float(value) / 100.0) if isinstance(value, (int, float)) else 0
        return (
            "<div style='display:flex;align-items:center;gap:6px'>"
            f"<div style='height:10px;width:{width}px;background:{color};border-radius:3px'></div>"
            f"<span style='font-size:12px'>{value:.1f}%</span>"
            "</div>"
        )
    rows = "<tr><th>Statistica</th><th>Valore</th></tr>"
    for key, val in patterns.items():
        label = key.replace("%", "").strip()
        cell = f"<b>{int(val)}</b>" if key == "P" else bar_html(val, color_hex)
        rows += f"<tr><td>{label}</td><td>{cell}</td></tr>"
    return "<table style='border-collapse:collapse;width:100%;font-size:12px'>" + rows + "</table>"

def plot_timeframe_goals(tf_scored, tf_conceded, tf_scored_pct, tf_conceded_pct, team):
    data = []
    keys = list(tf_scored.keys())
    for tf in keys:
        data.append({"Time Frame": tf, "Tipo": "Segnati", "Perc": tf_scored_pct.get(tf, 0), "Count": tf_scored.get(tf, 0)})
        data.append({"Time Frame": tf, "Tipo": "Subiti",  "Perc": tf_conceded_pct.get(tf, 0), "Count": tf_conceded.get(tf, 0)})
    df_tf = pd.DataFrame(data)
    chart = alt.Chart(df_tf).mark_bar().encode(
        x=alt.X("Time Frame:N", title="Minuti", sort=keys),
        y=alt.Y("Perc:Q", title="Percentuale (%)"),
        color=alt.Color("Tipo:N", scale=alt.Scale(domain=["Segnati", "Subiti"], range=["#16a34a", "#dc2626"])),
        xOffset="Tipo:N",
        tooltip=["Tipo", "Time Frame", alt.Tooltip("Perc:Q", format=".1f"), "Count"],
    ).properties(height=300, title=f"Distribuzione gol per intervalli – {team}")
    text = alt.Chart(df_tf).mark_text(align="center", baseline="middle", dy=-5).encode(
        x=alt.X("Time Frame:N", sort=keys),
        y="Perc:Q",
        detail="Tipo:N",
        text=alt.Text("Count:Q", format=".0f"),
    )
    return chart + text

def goal_pattern_keys():
    keys = [
        "P", "Win %", "Draw %", "Loss %", "First Goal %", "Last Goal %",
        "1-0 %", "1-1 after 1-0 %", "2-0 after 1-0 %",
        "0-1 %", "1-1 after 0-1 %", "0-2 after 0-1 %",
        "2+ Goals %", "H 1st %", "D 1st %", "A 1st %",
        "H 2nd %", "D 2nd %", "A 2nd %", "0-0 %",
    ]
    for a, b in timeframes():
        keys.append(f"{a}-{b} Goals %")
    return keys

def goal_pattern_keys_without_tf():
    return [
        "P", "Win %", "Draw %", "Loss %", "0-0 %",
        "1-0 %", "1-1 after 1-0 %", "2-0 after 1-0 %",
        "0-1 %", "1-1 after 0-1 %", "0-2 after 0-1 %",
        "2+ Goals %", "H 1st %", "D 1st %", "A 1st %",
        "H 2nd %", "D 2nd %", "A 2nd %",
    ]

def compute_goal_patterns_total(patterns_home, patterns_away, total_home_matches, total_away_matches):
    total_matches = total_home_matches + total_away_matches
    total = {}
    for key in goal_pattern_keys():
        if key == "P":
            total["P"] = total_matches
        elif key in ["Win %", "Draw %", "Loss %"]:
            if key == "Win %":
                val = (patterns_home.get("Win %", 0) + patterns_away.get("Loss %", 0)) / 2
            elif key == "Draw %":
                val = (patterns_home.get("Draw %", 0) + patterns_away.get("Draw %", 0)) / 2
            else:
                val = (patterns_home.get("Loss %", 0) + patterns_away.get("Win %", 0)) / 2
            total[key] = round(val, 2)
        elif key in ["First Goal %", "Last Goal %"]:
            continue
        else:
            hv = patterns_home.get(key, 0)
            av = patterns_away.get(key, 0)
            val = ((hv * total_home_matches) + (av * total_away_matches)) / total_matches if total_matches > 0 else 0
            total[key] = round(val, 2)
    return total


# =========================================================
# Stand-alone helpers (compatibilità)
# =========================================================
def show_team_macro_stats(df, team, venue):
    stats = compute_team_macro_stats(df, team, venue)
    if not stats:
        st.info(f"⚠️ Nessuna partita utile per {team} ({venue})."); return
    df_stats = pd.DataFrame([stats]).set_index(pd.Index([venue]))
    st.dataframe(df_stats, use_container_width=True)

def show_goal_patterns(df, team1, team2, country, stagione):
    df = df.copy()
    df = df[(df["country"] == country) & (df["Stagione"].astype("string") == str(stagione))]
    df_team1_home = df[df["Home"].astype("string") == team1]
    df_team2_away = df[df["Away"].astype("string") == team2]
    mask_played_home = df_team1_home.apply(is_match_played, axis=1)
    mask_played_away = df_team2_away.apply(is_match_played, axis=1)
    df_team1_home = df_team1_home[mask_played_home]
    df_team2_away = df_team2_away[mask_played_away]
    total_home = len(df_team1_home); total_away = len(df_team2_away)
    if total_home == 0 and total_away == 0:
        st.info("Nessun match utile per calcolare i pattern."); return
    patterns_home, tf_scored_home, tf_conceded_home = compute_goal_patterns(df_team1_home, "Home", total_home) if total_home else ({}, {}, {})
    patterns_away, tf_scored_away, tf_conceded_away = compute_goal_patterns(df_team2_away, "Away", total_away) if total_away else ({}, {}, {})
    tf_scored_home_pct = _tf_to_pct(tf_scored_home) if tf_scored_home else {}
    tf_conceded_home_pct = _tf_to_pct(tf_conceded_home) if tf_conceded_home else {}
    tf_scored_away_pct = _tf_to_pct(tf_scored_away) if tf_scored_away else {}
    tf_conceded_away_pct = _tf_to_pct(tf_conceded_away) if tf_conceded_away else {}
    patterns_total = compute_goal_patterns_total(patterns_home or {}, patterns_away or {}, total_home, total_away)
    html_home = _build_goal_pattern_html(patterns_home, team1, "#16a34a") if patterns_home else None
    html_away = _build_goal_pattern_html(patterns_away, team2, "#dc2626") if patterns_away else None
    html_total = _build_goal_pattern_html({k: patterns_total.get(k, 0) for k in goal_pattern_keys_without_tf()}, "Totale", "#2563eb") if (patterns_home or patterns_away) else None
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown(f"### {team1} (Home)");  st.markdown(html_home or "—", unsafe_allow_html=True)
    with col2: st.markdown(f"### {team2} (Away)");  st.markdown(html_away or "—", unsafe_allow_html=True)
    with col3: st.markdown(f"### Totale");          st.markdown(html_total or "—", unsafe_allow_html=True)
    if patterns_home:
        ch = plot_timeframe_goals(tf_scored_home, tf_conceded_home, tf_scored_home_pct, tf_conceded_home_pct, team1)
        st.altair_chart(ch, use_container_width=True)
    if patterns_away:
        ca = plot_timeframe_goals(tf_scored_away, tf_conceded_away, tf_scored_away_pct, tf_conceded_away_pct, team2)
        st.altair_chart(ca, use_container_width=True)
