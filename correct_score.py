# correct_score.py — pannello Correct Score (Poisson + Dixon–Coles) con parametri salvati per lega
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# Persisitenza parametri CS per lega
from cs_prefs import get_params as cs_get, save_params as cs_save


# =========================
# Altair theme (dark friendly)
# =========================
def _alt_theme():
    return {
        "config": {
            "view": {"stroke": "transparent"},
            "axis": {"labelFontSize": 12, "titleFontSize": 12, "gridColor": "#223", "labelColor": "#e5e7eb", "titleColor": "#cbd5e1"},
            "legend": {"labelFontSize": 12, "titleFontSize": 12, "labelColor": "#e5e7eb", "titleColor": "#cbd5e1"},
            "title": {"fontSize": 14, "color": "#e5e7eb"},
            "background": "transparent",
        }
    }
try:
    alt.themes.register("app_theme_cs", _alt_theme)
    alt.themes.enable("app_theme_cs")
except Exception:
    pass


# =========================
# Utils
# =========================
def _to_num(s: pd.Series) -> pd.Series:
    if s is None:
        return pd.Series(dtype="float")
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _last_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if not n or n <= 0 or "Data" not in df.columns:
        return df
    tmp = df.copy()
    tmp["_d"] = pd.to_datetime(tmp["Data"], errors="coerce")
    tmp = tmp.sort_values("_d", ascending=False).drop(columns=["_d"])
    return tmp.head(n)


def _league_avgs(df_league: pd.DataFrame) -> Tuple[float, float]:
    """Media di lega per GF home e GF away."""
    hg = _to_num(df_league.get("Home Goal FT")).fillna(0)
    ag = _to_num(df_league.get("Away Goal FT")).fillna(0)
    if len(hg) == 0:
        return 1.2, 1.2
    return float(hg.mean()), float(ag.mean())


@dataclass
class XG:
    home: float
    away: float


# =========================
# Stima λ con shrink (attack/def multipliers) + forma recente
# =========================
def _team_ctx(df: pd.DataFrame, team: str, venue: str) -> pd.DataFrame:
    if venue == "Home":
        return df[df["Home"].astype(str) == str(team)]
    return df[df["Away"].astype(str) == str(team)]


def _safemean(s: pd.Series, default: float = 0.0) -> float:
    s = _to_num(s)
    return float(s.mean()) if len(s) else float(default)


def _attack_def_multipliers(
    df_team: pd.DataFrame,
    venue: str,
    league_home_avg: float,
    league_away_avg: float,
) -> Tuple[float, float, int]:
    """
    Ritorna (attack_mult, defense_mult, n_partite)
    - attack_mult è rispetto alla media di lega corretta per venue
    - defense_mult usa la media dell'avversario in quel venue (es. difesa Away confrontata con gol Home di lega)
    """
    n = len(df_team)
    if n == 0:
        return 1.0, 1.0, 0

    if venue == "Home":
        gf = _safemean(df_team.get("Home Goal FT"), 0.0)  # segnati da Home
        ga = _safemean(df_team.get("Away Goal FT"), 0.0)  # subiti da Home
        atk = (gf / league_home_avg) if league_home_avg > 0 else 1.0
        dfn = (ga / league_away_avg) if league_away_avg > 0 else 1.0
    else:
        gf = _safemean(df_team.get("Away Goal FT"), 0.0)  # segnati da Away
        ga = _safemean(df_team.get("Home Goal FT"), 0.0)  # subiti da Away
        atk = (gf / league_away_avg) if league_away_avg > 0 else 1.0
        dfn = (ga / league_home_avg) if league_home_avg > 0 else 1.0

    # bound morbidi per evitare estremi
    atk = float(np.clip(atk, 0.4, 2.5))
    dfn = float(np.clip(dfn, 0.4, 2.5))
    return atk, dfn, n


def _shrink_to_one(mult: float, n: int, kappa: float) -> float:
    """
    Shrink verso 1 con forza kappa (interpreta kappa come 'match equivalenti' in prior).
    """
    k = max(0.0, float(kappa))
    return float((k * 1.0 + n * mult) / (k + n)) if (k + n) > 0 else 1.0


def _blend_recent(base_mult: float, recent_mult: float, weight_recent: float) -> float:
    w = float(np.clip(weight_recent, 0.0, 1.0))
    return float((1.0 - w) * base_mult + w * recent_mult)


def estimate_expected_goals(
    df: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    seasons: Optional[list[str]] = None,
    kappa: float = 3.0,
    recent_weight: float = 0.25,
    recent_n: int = 6,
) -> XG:
    """
    Calcola λ_home e λ_away con:
      - moltiplicatori attacco/difesa rispetto alla media di lega
      - shrink verso 1 con forza kappa
      - blend con forma recente (ultime N gare, weight recent_weight)
    """
    d = df.copy()

    if "country" in d.columns:
        d = d[d["country"].astype(str).str.upper() == str(league).upper()]

    if seasons:
        d = d[d["Stagione"].astype(str).isin([str(s) for s in seasons])]

    # media di lega
    league_home_avg, league_away_avg = _league_avgs(d)

    # contesti pieni
    df_home_ctx = _team_ctx(d, home_team, "Home")
    df_away_ctx = _team_ctx(d, away_team, "Away")

    # forma recente
    df_home_recent = _last_n(df_home_ctx, int(recent_n))
    df_away_recent = _last_n(df_away_ctx, int(recent_n))

    # moltiplicatori base
    atkH, defH, nH = _attack_def_multipliers(df_home_ctx, "Home", league_home_avg, league_away_avg)
    atkA, defA, nA = _attack_def_multipliers(df_away_ctx, "Away", league_home_avg, league_away_avg)

    # moltiplicatori recenti
    atkH_r, defH_r, nH_r = _attack_def_multipliers(df_home_recent, "Home", league_home_avg, league_away_avg)
    atkA_r, defA_r, nA_r = _attack_def_multipliers(df_away_recent, "Away", league_home_avg, league_away_avg)

    # shrink verso 1
    atkH_s = _shrink_to_one(atkH, nH, kappa=kappa)
    defH_s = _shrink_to_one(defH, nH, kappa=kappa)
    atkA_s = _shrink_to_one(atkA, nA, kappa=kappa)
    defA_s = _shrink_to_one(defA, nA, kappa=kappa)

    atkH_r_s = _shrink_to_one(atkH_r, nH_r, kappa=kappa)
    defH_r_s = _shrink_to_one(defH_r, nH_r, kappa=kappa)
    atkA_r_s = _shrink_to_one(atkA_r, nA_r, kappa=kappa)
    defA_r_s = _shrink_to_one(defA_r, nA_r, kappa=kappa)

    # blend con forma recente
    atkH_b = _blend_recent(atkH_s, atkH_r_s, recent_weight)
    defH_b = _blend_recent(defH_s, defH_r_s, recent_weight)
    atkA_b = _blend_recent(atkA_s, atkA_r_s, recent_weight)
    defA_b = _blend_recent(defA_s, defA_r_s, recent_weight)

    # λ finali
    lam_home = float(np.clip(league_home_avg * atkH_b * defA_b, 0.05, 6.0))
    lam_away = float(np.clip(league_away_avg * atkA_b * defH_b, 0.05, 6.0))

    return XG(home=lam_home, away=lam_away)


# =========================
# Dixon–Coles correction
# =========================
def _dc_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Correzione di Dixon–Coles per risultati a basso punteggio.
    Riferimento pratico: τ(0,0)=1-λhλaρ; τ(0,1)=1+λhρ; τ(1,0)=1+λaρ; τ(1,1)=1-ρ; altrimenti 1.
    """
    if x == 0 and y == 0:
        return 1.0 - lam_home * lam_away * rho
    if x == 0 and y == 1:
        return 1.0 + lam_home * rho
    if x == 1 and y == 0:
        return 1.0 + lam_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def correct_score_matrix(lam_home: float, lam_away: float, rho: float = -0.05, max_goals: int = 6) -> np.ndarray:
    """
    Genera matrice P(X=x, Y=y) con Poisson indipendenti corretti da Dixon–Coles.
    Rinormalizza per sommare a 1.
    """
    mx = int(max(1, int(max_goals)))
    xs = np.arange(0, mx + 1)
    ys = np.arange(0, mx + 1)

    # Poisson pmf
    px = np.exp(-lam_home) * np.power(lam_home, xs) / np.array([math.factorial(i) for i in xs], dtype=float)
    py = np.exp(-lam_away) * np.power(lam_away, ys) / np.array([math.factorial(i) for i in ys], dtype=float)

    mat = np.outer(px, py)

    # DC correction on low scores
    R = float(rho)
    for x in range(0, min(2, mx) + 1):
        for y in range(0, min(2, mx) + 1):
            mat[x, y] *= _dc_tau(x, y, lam_home, lam_away, R)

    # Rinormalizza
    s = mat.sum()
    if s > 0:
        mat = mat / s
    return mat


# =========================
# Derivati dalla matrice CS
# =========================
def _derived_from_matrix(pmat: np.ndarray) -> Dict[str, Any]:
    gmax = pmat.shape[0] - 1
    # 1X2
    p_home = float(np.tril(pmat, -1).sum())  # x>y
    p_draw = float(np.trace(pmat))
    p_away = float(np.triu(pmat, 1).sum())   # y>x

    # Over/Under 2.5
    over25 = 0.0
    for x in range(gmax + 1):
        for y in range(gmax + 1):
            if (x + y) > 2.5:
                over25 += float(pmat[x, y])
    under25 = 1.0 - over25

    # BTTS
    btts = float(pmat[1:, 1:].sum())
    nobtts = 1.0 - btts

    return {
        "1": p_home, "X": p_draw, "2": p_away,
        "Over2.5": over25, "Under2.5": under25,
        "BTTS": btts, "NoBTTS": nobtts,
    }


def _top_scorelines(pmat: np.ndarray, topn: int = 10) -> pd.DataFrame:
    gmax = pmat.shape[0] - 1
    rows = []
    for i in range(gmax + 1):
        for j in range(gmax + 1):
            rows.append({"Score": f"{i}-{j}", "Prob %": 100 * float(pmat[i, j])})
    df = pd.DataFrame(rows).sort_values("Prob %", ascending=False).head(int(topn))
    df["Prob %"] = df["Prob %"].map(lambda v: round(v, 2))
    return df.reset_index(drop=True)


# =========================
# UI helpers
# =========================
def _num(v: float, pct: bool = False) -> str:
    if pct:
        return f"{v*100:.1f}%"
    return f"{v:.2f}"


def _prob_card(title: str, value: float):
    st.metric(label=title, value=_num(value, pct=True))


def _render_heatmap(pmat: np.ndarray, home: str, away: str):
    gmax = pmat.shape[0] - 1
    data = []
    for i in range(gmax + 1):
        for j in range(gmax + 1):
            data.append({"Home": i, "Away": j, "Prob": 100 * float(pmat[i, j])})
    dfh = pd.DataFrame(data)

    ch = (
        alt.Chart(dfh)
        .mark_rect()
        .encode(
            x=alt.X("Away:O", title=f"{away} goals"),
            y=alt.Y("Home:O", title=f"{home} goals"),
            color=alt.Color("Prob:Q", title="%", scale=alt.Scale(scheme="blues")),
            tooltip=[alt.Tooltip("Home:Q"), alt.Tooltip("Away:Q"), alt.Tooltip("Prob:Q", format=".2f")],
        )
        .properties(height=360, width="container", title="Correct Score – Heatmap (%)")
    )
    st.altair_chart(ch, use_container_width=True)


# =========================
# Entry point
# =========================
def run_correct_score_panel(
    df: pd.DataFrame,
    league_code: str,
    home_team: str,
    away_team: str,
    seasons: Optional[list[str]] = None,
    default_rho: float = -0.05,
    default_kappa: float = 3.0,
    default_recent_weight: float = 0.25,
    default_recent_n: int = 6,
    default_max_goals: int = 6,
):
    st.markdown("### 🎯 Correct Score — Poisson + Dixon–Coles")

    # Parametri salvati per lega
    prefs = cs_get(league_code) if league_code else {}
    rho_init = float(prefs.get("rho", default_rho))
    kappa_init = float(prefs.get("kappa", default_kappa))
    r_w_init = float(prefs.get("recent_weight", default_recent_weight))
    r_n_init = int(prefs.get("recent_n", default_recent_n))
    gmax_init = int(prefs.get("max_goals", default_max_goals))

    with st.expander("Parametri modello", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            rho = st.number_input("Rho (Dixon-Coles)", value=rho_init, step=0.01, min_value=-0.5, max_value=0.5, key="cs_rho")
        with c2:
            kappa = st.number_input("Kappa (shrink)", value=kappa_init, step=0.1, min_value=0.0, max_value=50.0, key="cs_kappa")
        with c3:
            recent_weight = st.slider("Peso forma recente", 0.0, 1.0, r_w_init, 0.05, key="cs_recent_w")
        with c4:
            recent_n = st.slider("N gare recenti", 0, 20, r_n_init, 1, key="cs_recent_n")
        with c5:
            max_goals = st.slider("Max goals", 4, 10, gmax_init, 1, key="cs_gmax")

        if st.button("💾 Salva impostazioni per questa lega", use_container_width=True):
            cs_save(league_code, {
                "rho": float(rho),
                "kappa": float(kappa),
                "recent_weight": float(recent_weight),
                "recent_n": int(recent_n),
                "max_goals": int(max_goals),
            })
            st.success("Parametri salvati per la lega.")

    # Stima λ
    with st.spinner("Calcolo attacchi/difese e λ attesi…"):
        xg = estimate_expected_goals(
            df=df,
            league=league_code,
            home_team=home_team,
            away_team=away_team,
            seasons=seasons,
            kappa=float(kappa),
            recent_weight=float(recent_weight),
            recent_n=int(recent_n),
        )

    st.caption(f"λ stimati — **{home_team}**: {xg.home:.2f} | **{away_team}**: {xg.away:.2f}")

    # Matrice Correct Score
    pmat = correct_score_matrix(xg.home, xg.away, rho=float(rho), max_goals=int(max_goals))

    # Heatmap + Top scoreline
    _render_heatmap(pmat, home_team, away_team)

    st.subheader("Top correct score")
    topn = st.slider("Visualizza i primi N punteggi", 5, 20, 10, 1, key="cs_topn")
    df_top = _top_scorelines(pmat, topn)
    st.dataframe(
        df_top,
        use_container_width=True,
        height=36 + 28 * min(len(df_top), 12),
        column_config={"Prob %": st.column_config.NumberColumn(format="%.2f")},
    )

    # Mercati derivati (1X2, O/U, BTTS)
    st.subheader("Derivati dal modello")
    d = _derived_from_matrix(pmat)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: _prob_card("1", d["1"])
    with c2: _prob_card("X", d["X"])
    with c3: _prob_card("2", d["2"])
    with c4: _prob_card("Over 2.5", d["Over2.5"])
    with c5: _prob_card("Under 2.5", d["Under2.5"])
    with c6: _prob_card("BTTS", d["BTTS"])

    # CSV export
    st.download_button(
        "⬇️ Esporta matrice CS (CSV)",
        data=pd.DataFrame(pmat).to_csv(index=False).encode("utf-8"),
        file_name=f"correct_score_matrix_{home_team}_vs_{away_team}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    with st.expander("Dettagli calcolo λ (debug)", expanded=False):
        st.write(
            "λ calcolati combinando media di lega, moltiplicatori attacco/difesa "
            "per contesto corretto (Home@Casa, Away@Trasferta), shrink verso 1 (kappa) "
            "e blend con forma recente (peso recent_weight, ultime recent_n gare)."
        )
