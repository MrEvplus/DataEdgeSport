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
# Entry point classico (facoltativo)
# =========================================================
def run_team_stats(df: pd.DataFrame, db_selected: str):
    st.header("📊 Statistiche per Squadre")
    _render_setup_and_body(df, db_selected, is_embedded=False)


# =========================================================
# Entry point per TAB su Confronto pre-match
# =========================================================
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
        st.error("Colonna 'country' mancante.")
        st.stop()
    if "Home" not in df.columns or "Away" not in df.columns:
        st.error("Colonne 'Home' e/o 'Away' mancanti.")
        st.stop()

    _init_shared_quotes()  # ➜ quote condivise pronte

    df = df.copy()
    df["country"] = _ensure_str_with_unknown(df["country"], "Unknown").str.strip().str.upper()
    db_selected = (db_selected or "").strip().upper()

    if db_selected not in df["country"].unique():
        st.warning(f"⚠️ Il campionato selezionato '{db_selected}' non è presente nel database.")
        st.stop()

    df = df[df["country"] == db_selected].copy()
    df["Home"] = _ensure_str_with_unknown(df["Home"], "")
    df["Away"] = _ensure_str_with_unknown(df["Away"], "")

    # --- Filtro stagioni (solo manuale per questa sezione)
    if "Stagione" not in df.columns:
        st.error("Colonna 'Stagione' mancante.")
        st.stop()

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
            st.caption(f"Stagioni attive (sezione Squadre): **{', '.join(seasons_selected)}**")
            df = df[df["Stagione"].astype("string").isin(seasons_selected)].copy()
        else:
            st.caption("Stagioni attive (sezione Squadre): **Tutte**")

    # --- Selettori squadre (se pagina autonoma)
    if not is_embedded:
        teams_available = sorted(
            set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique())
        )
        col1, col2 = st.columns(2)
        with col1:
            squadra_casa = st.selectbox("Seleziona Squadra Casa", options=teams_available, key="teams:home")
        with col2:
            squadra_ospite = st.selectbox("Seleziona Squadra Ospite (facoltativa)", options=[""] + teams_available, key="teams:away")
    else:
        if not squadra_casa or not squadra_ospite:
            st.info("Seleziona **entrambe** le squadre nella parte alta della pagina pre-match.")
            return
        if squadra_casa == squadra_ospite:
            st.warning("Casa e Ospite sono uguali: modifica la selezione.")
            return

    # ==========================
    # KPI CARDS (macro)
    # ==========================
    st.subheader("📌 KPI Macro (contesto corretto)")
    st.markdown(
        "Metriche calcolate su **Home@Casa** per la squadra di casa e **Away@Trasferta** per la squadra ospite. "
        "BTTS% = percentuale partite con gol di entrambe."
    )

    stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
    stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")

    if not stats_home or not stats_away:
        st.info("⚠️ Una delle due squadre non ha match disponibili nelle stagioni selezionate.")
        return

    def _card(value, label, help_txt=""):
        st.metric(label=label, value=value, help=help_txt)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card(stats_home.get("Matches Played", 0), f"{squadra_casa} – Partite")
    with c2:
        _card(f"{stats_home.get('Win %', 0):.1f}%", f"{squadra_casa} – Win%")
    with c3:
        _card(f"{stats_home.get('Avg Goals Scored', 0):.2f}", f"{squadra_casa} – GF")
    with c4:
        _card(f"{stats_home.get('BTTS %', 0):.1f}%", f"{squadra_casa} – BTTS%")

    d1, d2, d3, d4 = st.columns(4)
    with d1:
        _card(stats_away.get("Matches Played", 0), f"{squadra_ospite} – Partite")
    with d2:
        _card(f"{stats_away.get('Win %', 0):.1f}%", f"{squadra_ospite} – Win%")
    with d3:
        _card(f"{stats_away.get('Avg Goals Scored', 0):.2f}", f"{squadra_ospite} – GF")
    with d4:
        _card(f"{stats_away.get('BTTS %', 0):.1f}%", f"{squadra_ospite} – BTTS%")

    st.divider()

    # ==========================
    # Goal Patterns & Fasce Minuto
    # ==========================
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
                _build_goal_pattern_html(
                    {k: total.get(k, 0) for k in goal_pattern_keys_without_tf()},
                    "Totale", "#2563eb"
                ),
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

    st.divider()

# ======== COMPACT MATCH BOARD (griglia in stile screenshot) ========
def _pct(x: float, y: float) -> float:
    return round((x / y) * 100, 2) if y else 0.0

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
    """
    df_ctx: partite della squadra già nel contesto giusto:
            Home@Casa oppure Away@Trasferta
    """
    if df_ctx.empty:
        return dict(P=0, Z0=0, O15=0, O25=0, O35=0, GF=0.0, GA=0.0,
                    BTTS=0, FH1=0, FH2=0, SH1=0, SH2=0)

    df = df_ctx.copy()
    # ft
    h = pd.to_numeric(df["Home Goal FT"], errors="coerce").fillna(0).astype(int)
    a = pd.to_numeric(df["Away Goal FT"], errors="coerce").fillna(0).astype(int)
    tot = (h + a).astype(int)

    # 1° tempo (se disponibile)
    if "Home Goal 1T" in df.columns and "Away Goal 1T" in df.columns:
        h1 = pd.to_numeric(df["Home Goal 1T"], errors="coerce").fillna(0).astype(int)
        a1 = pd.to_numeric(df["Away Goal 1T"], errors="coerce").fillna(0).astype(int)
    else:
        h1 = pd.Series([0]*len(df), index=df.index)
        a1 = pd.Series([0]*len(df), index=df.index)
    # 2° tempo = FT - 1T
    h2 = (h - h1).clip(lower=0)
    a2 = (a - a1).clip(lower=0)

    # metriche base
    P   = int(len(df))
    Z0  = _pct(((h == 0) & (a == 0)).sum(), P)
    O15 = _pct((tot > 1).sum(), P)
    O25 = _pct((tot > 2).sum(), P)
    O35 = _pct((tot > 3).sum(), P)
    BT  = _pct(((h > 0) & (a > 0)).sum(), P)

    # GF/GA medi nel contesto giusto
    if venue == "Home":
        GF = float(h.mean())
        GA = float(a.mean())
    else:
        GF = float(a.mean())
        GA = float(h.mean())

    # 1+ e 2+ per tempi
    FH1 = _pct(((h1 + a1) >= 1).sum(), P)
    FH2 = _pct(((h1 + a1) >= 2).sum(), P)
    SH1 = _pct(((h2 + a2) >= 1).sum(), P)
    SH2 = _pct(((h2 + a2) >= 2).sum(), P)

    return dict(P=P, Z0=Z0, O15=O15, O25=O25, O35=O35,
                GF=GF, GA=GA, BTTS=BT, FH1=FH1, FH2=FH2, SH1=SH1, SH2=SH2)

def _render_board_row(team: str, venue_label: str, k: dict, row_color: str) -> str:
    return f"""
      <div class="gb cell team" style="background:{row_color}">{team} <span style="opacity:.7;font-weight:400">({venue_label})</span></div>
      <div class="gb cell">{k['P']}</div>
      <div class="gb cell">{_cell_bar(k['Z0'], '#9ca3af')}</div>
      <div class="gb cell">{_cell_bar(k['O15'], '#16a34a')}</div>
      <div class="gb cell">{_cell_bar(k['O25'], '#65a30d')}</div>
      <div class="gb cell">{_cell_bar(k['O35'], '#84cc16')}</div>
      <div class="gb cell">{_cell_dual_bar(k['GF'], k['GA'])}</div>
      <div class="gb cell">{_cell_bar(k['BTTS'], '#10b981')}</div>
      <div class="gb cell">{_cell_bar(k['FH1'], '#0ea5e9')}</div>
      <div class="gb cell">{_cell_bar(k['FH2'], '#0284c7')}</div>
      <div class="gb cell">{_cell_bar(k['SH1'], '#a855f7')}</div>
      <div class="gb cell">{_cell_bar(k['SH2'], '#7c3aed')}</div>
    """

def render_compact_match_board(df: pd.DataFrame, league_code: str, squadra_casa: str, squadra_ospite: str):
    """
    Mostra la griglia compatta stile screenshot per le due squadre selezionate.
    Usa il df già filtrato per le stagioni (in questa sezione lo fai manualmente).
    """
    if not all(col in df.columns for col in ["Home", "Away", "Home Goal FT", "Away Goal FT"]):
        st.info("Colonne minime non disponibili per la match board.")
        return

    theme_css = """
    <style>
      .board { border:1px solid #e5e7eb;border-radius:12px;overflow:hidden;margin-top:.5rem }
      .board .hdr { background:#e8f0fe;padding:8px 12px;font-weight:600;display:flex;gap:8px;align-items:center; }
      .grid { display:grid;grid-template-columns: 220px 70px repeat(10, 1fr);gap:8px;padding:12px;background:white }
      .grid .cell { background:#f8fafc;border:1px solid #eef2f7;border-radius:8px;padding:6px;display:flex;align-items:center;justify-content:center;min-height:48px }
      .grid .team { justify-content:flex-start;padding-left:10px;font-weight:600 }
      .grid .head { font-size:12px;font-weight:600;background:#f1f5f9 }
      .legend { display:flex;gap:14px;align-items:center;margin-left:auto;font-size:12px }
      .legend span::before { content:""; display:inline-block; width:10px;height:10px; border-radius:2px; margin-right:6px; vertical-align:middle }
      .l-ov15::before { background:#16a34a }
      .l-ov25::before { background:#65a30d }
      .l-ov35::before { background:#84cc16 }
      .l-btts::before { background:#10b981 }
      .l-fh::before   { background:#0ea5e9 }
      .l-sh::before   { background:#a855f7 }
    </style>
    """

    st.markdown(theme_css, unsafe_allow_html=True)
    st.markdown('<div class="board">', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="hdr">
          <div>Match Board – <b>{squadra_casa}</b> vs <b>{squadra_ospite}</b> · <span style="opacity:.7">{league_code}</span></div>
          <div class="legend">
            <span class="l-ov15">O1.5</span>
            <span class="l-ov25">O2.5</span>
            <span class="l-ov35">O3.5</span>
            <span class="l-btts">BTTS</span>
            <span class="l-fh">1° T</span>
            <span class="l-sh">2° T</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Scope corretti
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

    header = """
      <div class="gb head">Team (context)</div>
      <div class="gb head">P</div>
      <div class="gb head">0-0</div>
      <div class="gb head">O1.5</div>
      <div class="gb head">O2.5</div>
      <div class="gb head">O3.5</div>
      <div class="gb head">Scrd / Concd</div>
      <div class="gb head">BTTS</div>
      <div class="gb head">1+ 1st</div>
      <div class="gb head">2+ 1st</div>
      <div class="gb head">1+ 2nd</div>
      <div class="gb head">2+ 2nd</div>
    """

    row_home = _render_board_row(squadra_casa, "Home", k_home, "#f0fdf4")
    row_away = _render_board_row(squadra_ospite, "Away", k_away, "#fff7ed")

    st.markdown(f'<div class="grid">{header}{row_home}{row_away}</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
# ======== /COMPACT MATCH BOARD ========


    # ==========================
    # EV Consigliato per il match (quote condivise)
    # ==========================
    st.subheader("🏅 EV consigliato (storico)")
    st.caption("Le quote Over/BTTS utilizzate sono quelle **condivise** inserite nella parte alta della pagina Pre-Match.")

    last_n = st.slider("Ultimi N match (0=tutti)", min_value=0, max_value=50, value=0, key="teams:lastn")

    df_home_ctx = _limit_last_n(df_home, last_n)
    df_away_ctx = _limit_last_n(df_away, last_n)
    df_h2h = df[
        ((df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)) |
        ((df["Home"] == squadra_ospite) & (df["Away"] == squadra_casa))
    ].copy()
    df_h2h = _limit_last_n(df_h2h, last_n)

    shared = _get_shared_quotes()
    q15, q25, q35, qgg = shared["ov15"], shared["ov25"], shared["ov35"], shared["btts"]

    df_ev, best = _build_ev_table(
        df_home_ctx, df_away_ctx, df_h2h,
        squadra_casa, squadra_ospite,
        q15, q25, q35, qgg
    )

    if best and best["ev"] > 0:
        _best_ev_card(best)
    else:
        st.info("Nessun EV > 0 con le quote inserite (nei campioni disponibili).")

    st.dataframe(
        df_ev,
        use_container_width=True,
        height=360,
        column_config={
            "Quota": st.column_config.NumberColumn(format="%.2f"),
            f"{squadra_casa} @Casa %": st.column_config.NumberColumn(format="%.2f"),
            f"{squadra_ospite} @Trasferta %": st.column_config.NumberColumn(format="%.2f"),
            "Blended %": st.column_config.NumberColumn(format="%.2f"),
            "Head-to-Head %": st.column_config.NumberColumn(format="%.2f"),
            "EV Blended": st.column_config.NumberColumn(format="%.2f"),
            "EV H2H": st.column_config.NumberColumn(format="%.2f"),
        },
    )


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
        losses = int((data["Home Goal FT"] < data["Away Goal FT"]).sum())
    else:
        wins = int((data["Away Goal FT"] > data["Home Goal FT"]).sum())
        draws = int((data["Away Goal FT"] == data["Home Goal FT"]).sum())
        losses = int((data["Away Goal FT"] < data["Home Goal FT"]).sum())

    gf = float(_coerce_num(data[gf_col]).mean())
    ga = float(_coerce_num(data[ga_col]).mean())
    btts = float(((_coerce_num(data["Home Goal FT"]) > 0) & (_coerce_num(data["Away Goal FT"]) > 0)).mean() * 100)

    return {
        "Matches Played": n,
        "Win %": round((wins / n) * 100, 2),
        "Draw %": round((draws / n) * 100, 2),
        "Loss %": round((losses / n) * 100, 2),
        "Avg Goals Scored": round(gf, 2),
        "Avg Goals Conceded": round(ga, 2),
        "BTTS %": round(btts, 2),
    }


# =========================================================
# Match giocato?
# =========================================================
def is_match_played(row) -> bool:
    m_home = str(row.get("minuti goal segnato home", "") or "").strip()
    m_away = str(row.get("minuti goal segnato away", "") or "").strip()
    if m_home != "" or m_away != "":
        return True
    hg = row.get("Home Goal FT", None)
    ag = row.get("Away Goal FT", None)
    return pd.notna(hg) and pd.notna(ag)


# =========================================================
# Timeline / parse minuti
# =========================================================
def parse_goal_times(val):
    if pd.isna(val) or val == "":
        return []
    out = []
    for part in str(val).strip().split(";"):
        p = part.strip()
        if p.isdigit():
            out.append(int(p))
    return out

def build_timeline(row, venue):
    try:
        h_goals = parse_goal_times(row.get("minuti goal segnato home", ""))
        a_goals = parse_goal_times(row.get("minuti goal segnato away", ""))

        tl = [("H", m) for m in h_goals] + [("A", m) for m in a_goals]
        if tl:
            tl.sort(key=lambda x: x[1])
            return tl

        hg_raw = row.get("Home Goal FT", 0)
        ag_raw = row.get("Away Goal FT", 0)
        hg = int(hg_raw) if pd.notna(hg_raw) else 0
        ag = int(ag_raw) if pd.notna(ag_raw) else 0
        tl = [("H", 90)] * hg + [("A", 91)] * ag
        return tl
    except Exception:
        return []


# =========================================================
# Timeframes
# =========================================================
def timeframes():
    return [(0, 15), (16, 30), (31, 45), (46, 60), (61, 75), (76, 120)]


# =========================================================
# Goal pattern computation
# =========================================================
def compute_goal_patterns(df_team: pd.DataFrame, venue: str, total_matches: int):
    if total_matches == 0:
        return {key: 0 for key in goal_pattern_keys()}, {}, {}

    def pct(x):      return round((x / total_matches) * 100, 2) if total_matches > 0 else 0
    def pct_sub(x,y):return round((x / y) * 100, 2) if y > 0 else 0

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
        if not tl:
            continue

        first = tl[0][0]
        last  = tl[-1][0]
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
                one_zero += 1
                sH, sA = 1, 0
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 2 and sA == 0: two_zero_after_one_zero += 1; break
                    if sH == 1 and sA == 1: one_one_after_one_zero += 1; break
            elif first == "A":
                zero_one += 1
                sH, sA = 0, 1
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 1 and sA == 1: one_one_after_zero_one += 1; break
                    if sH == 0 and sA == 2: zero_two_after_zero_one += 1; break
        else:
            if first == "H":
                one_zero += 1
                sH, sA = 1, 0
                for ch,_ in tl[1:]:
                    if ch == "H": sH += 1
                    else:         sA += 1
                    if sH == 2 and sA == 0: two_zero_after_one_zero += 1; break
                    if sH == 1 and sA == 1: one_one_after_one_zero += 1; break
            elif first == "A":
                zero_one += 1
                sH, sA = 0, 1
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
        "Win %": pct(wins),
        "Draw %": pct(draws),
        "Loss %": pct(losses),
        "First Goal %": pct(first_goal),
        "Last Goal %": pct(last_goal),
        "1-0 %": pct(one_zero),
        "1-1 after 1-0 %": pct_sub(one_one_after_one_zero, one_zero),
        "2-0 after 1-0 %": pct_sub(two_zero_after_one_zero, one_zero),
        "0-1 %": pct(zero_one),
        "1-1 after 0-1 %": pct_sub(one_one_after_zero_one, zero_one),
        "0-2 after 0-1 %": pct_sub(zero_two_after_zero_one, zero_one),
        "2+ Goals %": pct(two_up),
        "H 1st %": pct(ht_home_win),
        "D 1st %": pct(ht_draw),
        "A 1st %": pct(ht_away_win),
        "H 2nd %": pct(sh_home_win),
        "D 2nd %": pct(sh_draw),
        "A 2nd %": pct(sh_away_win),
        "0-0 %": zero_zero_pct,
    }

    return patterns, tf_scored, tf_conceded

def _tf_to_pct(tf_dict: dict[str, int]) -> dict[str, float]:
    tot = sum(tf_dict.values())
    return {k: round((v / tot) * 100, 2) if tot > 0 else 0 for k, v in tf_dict.items()}


# =========================================================
# HTML goal pattern table
# =========================================================
def _build_goal_pattern_html(patterns: dict, team: str, color_hex: str) -> str:
    def bar_html(value: float, color: str, width_max: int = 90) -> str:
        width = int(width_max * float(value) / 100.0) if isinstance(value, (int, float)) else 0
        return (
            "<div style='display:flex;align-items:center;'>"
            f"<div style='height:10px;width:{width}px;background:{color};margin-right:6px;border-radius:3px;'></div>"
            f"<span style='font-size:12px'>{value:.1f}%</span>"
            "</div>"
        )

    rows = "<tr><th>Statistica</th><th>Valore</th></tr>"
    for key, val in patterns.items():
        label = key.replace("%", "").strip()
        if key == "P":
            cell = f"<b>{int(val)}</b>"
        else:
            cell = bar_html(val, color_hex)
        rows += f"<tr><td>{label}</td><td>{cell}</td></tr>"

    html = (
        "<table style='border-collapse:collapse;width:100%;font-size:12px;'>"
        f"{rows}"
        "</table>"
    )
    return html


# =========================================================
# Grafico fasce minuto
# =========================================================
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
    ).properties(
        height=300,
        title=f"Distribuzione gol per intervalli – {team}",
    )

    text = alt.Chart(df_tf).mark_text(align="center", baseline="middle", dy=-5).encode(
        x=alt.X("Time Frame:N", sort=keys),
        y="Perc:Q",
        detail="Tipo:N",
        text=alt.Text("Count:Q", format=".0f"),
    )
    return chart + text


# =========================================================
# Goal pattern keys
# =========================================================
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


# =========================================================
# Totale patterns (Home+Away pesati)
# =========================================================
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
# EV storico: tabella e best
# =========================================================
def _market_prob(df: pd.DataFrame, market: str, line: float | None) -> float:
    if df.empty:
        return 0.0
    hg = _coerce_num(df["Home Goal FT"]).fillna(0)
    ag = _coerce_num(df["Away Goal FT"]).fillna(0)
    goals = hg + ag
    if market == "BTTS":
        ok = ((hg > 0) & (ag > 0)).mean()
    else:
        ok = (goals > float(line)).mean() if line is not None else 0.0
    return round(float(ok) * 100, 2)

def _build_ev_table(
    df_home_ctx: pd.DataFrame,
    df_away_ctx: pd.DataFrame,
    df_h2h: pd.DataFrame,
    squadra_casa: str,
    squadra_ospite: str,
    q15: float,
    q25: float,
    q35: float,
    qgg: float,
):
    markets = [
        ("Over 1.5", 1.5, q15),
        ("Over 2.5", 2.5, q25),
        ("Over 3.5", 3.5, q35),
        ("BTTS", None, qgg),
    ]
    rows = []
    candidates = []

    for name, line, q in markets:
        p_home = _market_prob(df_home_ctx, name, line)
        p_away = _market_prob(df_away_ctx, name, line)
        p_blnd = round((p_home + p_away) / 2, 2) if (p_home > 0 or p_away > 0) else 0.0
        p_h2h  = _market_prob(df_h2h, name, line)

        ev_home = round(q * (p_home / 100) - 1, 2)
        ev_away = round(q * (p_away / 100) - 1, 2)
        ev_blnd = round(q * (p_blnd / 100) - 1, 2)
        ev_h2h  = round(q * (p_h2h  / 100) - 1, 2)

        n_h, n_a, n_h2h = len(df_home_ctx), len(df_away_ctx), len(df_h2h)
        qual_blnd = _quality_label(n_h + n_a)
        qual_h2h  = _quality_label(n_h2h)

        rows.append({
            "Mercato": name,
            "Quota": q,
            f"{squadra_casa} @Casa %": p_home,
            f"EV {squadra_casa}": ev_home,
            f"{squadra_ospite} @Trasferta %": p_away,
            f"EV {squadra_ospite}": ev_away,
            "Blended %": p_blnd,
            "EV Blended": ev_blnd,
            "Qualità Blended": qual_blnd,
            "Head-to-Head %": p_h2h,
            "EV H2H": ev_h2h,
            "Qualità H2H": qual_h2h,
            "Match H": n_h,
            "Match A": n_a,
            "Match H2H": n_h2h,
        })

        candidates += [
            {"scope": "Blended", "mercato": name, "quota": q, "prob": p_blnd, "ev": ev_blnd, "campione": n_h + n_a, "qualita": qual_blnd},
            {"scope": "Head-to-Head", "mercato": name, "quota": q, "prob": p_h2h, "ev": ev_h2h, "campione": n_h2h, "qualita": qual_h2h},
        ]

    df_ev = pd.DataFrame(rows)

    best = None
    for c in sorted(candidates, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c
            break

    return df_ev, best

def _best_ev_card(best: dict):
    bg = "#052e16"
    st.markdown(
        f"""
        <div style="border:1px solid #16a34a;border-radius:12px;padding:14px;background:{bg};color:#e5fff0;">
            <div style="font-size:14px;opacity:.9;">Miglior opportunità (storico)</div>
            <div style="display:flex;gap:20px;align-items:baseline;flex-wrap:wrap;">
                <div style="font-size:28px;font-weight:700;">EV {best['ev']:+.2f}</div>
                <div style="font-size:16px;">Mercato: <b>{best['mercato']}</b></div>
                <div style="font-size:16px;">Scope: <b>{best['scope']}</b></div>
                <div style="font-size:16px;">Prob: <b>{best['prob']:.1f}%</b></div>
                <div style="font-size:16px;">Quota: <b>{best['quota']:.2f}</b></div>
                <div style="font-size:16px;">Campione: <b>{best['campione']}</b> ({best['qualita']})</div>
            </div>
            <div style="font-size:12px;opacity:.8;margin-top:6px;">
                Nota: EV stimato su storico; verifica sempre la coerenza con il contesto attuale.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# Stand-alone di compatibilità (se usati altrove)
# =========================================================
def show_team_macro_stats(df, team, venue):
    stats = compute_team_macro_stats(df, team, venue)
    if not stats:
        st.info(f"⚠️ Nessuna partita utile per {team} ({venue}).")
        return
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

    total_home = len(df_team1_home)
    total_away = len(df_team2_away)

    if total_home == 0 and total_away == 0:
        st.info("Nessun match utile per calcolare i pattern.")
        return

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
    with col1:
        st.markdown(f"### {team1} (Home)")
        st.markdown(html_home or "—", unsafe_allow_html=True)
    with col2:
        st.markdown(f"### {team2} (Away)")
        st.markdown(html_away or "—", unsafe_allow_html=True)
    with col3:
        st.markdown(f"### Totale")
        st.markdown(html_total or "—", unsafe_allow_html=True)

    if patterns_home:
        ch = plot_timeframe_goals(tf_scored_home, tf_conceded_home, tf_scored_home_pct, tf_conceded_home_pct, team1)
        st.altair_chart(ch, use_container_width=True)
    if patterns_away:
        ca = plot_timeframe_goals(tf_scored_away, tf_conceded_away, tf_scored_away_pct, tf_conceded_away_pct, team2)
        st.altair_chart(ca, use_container_width=True)

