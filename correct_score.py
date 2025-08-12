# correct_score.py
# ------------------------------------------------------------
# Correct Score Prediction con Poisson + Dixon-Coles
# + Calibrazione automatica dei parametri su storico di lega:
#   - rho (Dixon-Coles)
#   - kappa (shrinkage verso media di lega)
#   - recent_weight (peso forma recente) e recent_n
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Streamlit/Altair opzionali: import condizionale
try:
    import streamlit as st
    import altair as alt
except Exception:  # pragma: no cover
    st = None
    alt = None


# ===========================
# Utility numeriche
# ===========================
def _poisson_pmf(mu: float, k: int) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1))


def _dixon_coles_tau(x: int, y: int, lam_home: float, lam_away: float, rho: float) -> float:
    """Correzione Dixon-Coles per celle a bassa segnatura."""
    if rho == 0:
        return 1.0
    if x == 0 and y == 0:
        return 1.0 - lam_home * lam_away * rho
    if x == 0 and y == 1:
        return 1.0 + lam_home * rho
    if x == 1 and y == 0:
        return 1.0 + lam_away * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def _safe_mean(a: Iterable[float]) -> float:
    a = list(a)
    return float(np.nanmean(a)) if len(a) > 0 else 0.0


def _shrink_mean(sum_vals: float, n: int, league_mean: float, kappa: float) -> float:
    """mean* = (sum + kappa * league_mean) / (n + kappa)"""
    n = max(int(n), 0)
    return (sum_vals + kappa * league_mean) / (n + kappa) if (n + kappa) > 0 else league_mean


# ===========================
# Estrazione statistiche base
# ===========================
@dataclass
class LeagueAverages:
    home_goals: float
    away_goals: float


def _league_averages(df: pd.DataFrame) -> LeagueAverages:
    hg = pd.to_numeric(df["Home Goal FT"], errors="coerce")
    ag = pd.to_numeric(df["Away Goal FT"], errors="coerce")
    return LeagueAverages(home_goals=float(hg.mean(skipna=True)), away_goals=float(ag.mean(skipna=True)))


@dataclass
class TeamContextStats:
    n: int
    scored_mean: float
    conceded_mean: float
    scored_sum: float
    conceded_sum: float


def _team_context_stats(df: pd.DataFrame, team: str, venue: str) -> TeamContextStats:
    if venue == "Home":
        d = df[df["Home"].astype("string") == team]
        scored = pd.to_numeric(d["Home Goal FT"], errors="coerce").fillna(0)
        conceded = pd.to_numeric(d["Away Goal FT"], errors="coerce").fillna(0)
    else:
        d = df[df["Away"].astype("string") == team]
        scored = pd.to_numeric(d["Away Goal FT"], errors="coerce").fillna(0)
        conceded = pd.to_numeric(d["Home Goal FT"], errors="coerce").fillna(0)

    return TeamContextStats(
        n=int(len(d)),
        scored_mean=float(scored.mean()) if len(d) else 0.0,
        conceded_mean=float(conceded.mean()) if len(d) else 0.0,
        scored_sum=float(scored.sum()),
        conceded_sum=float(conceded.sum()),
    )


def _recent_context_stats(df: pd.DataFrame, team: str, venue: str, last_n: int = 6) -> TeamContextStats:
    if venue == "Home":
        d = df[df["Home"].astype("string") == team].copy()
        sc_col, cc_col = "Home Goal FT", "Away Goal FT"
    else:
        d = df[df["Away"].astype("string") == team].copy()
        sc_col, cc_col = "Away Goal FT", "Home Goal FT"

    if "Data" in d.columns:
        d["_d_"] = pd.to_datetime(d["Data"], errors="coerce")
        d = d.sort_values("_d_", ascending=False).drop(columns=["_d_"])

    d = d.head(int(last_n))
    scored = pd.to_numeric(d[sc_col], errors="coerce").fillna(0)
    conceded = pd.to_numeric(d[cc_col], errors="coerce").fillna(0)

    return TeamContextStats(
        n=int(len(d)),
        scored_mean=float(scored.mean()) if len(d) else 0.0,
        conceded_mean=float(conceded.mean()) if len(d) else 0.0,
        scored_sum=float(scored.sum()),
        conceded_sum=float(conceded.sum()),
    )


# ===========================
# Stima λ attesi
# ===========================
@dataclass
class ExpectedGoals:
    home: float
    away: float


def estimate_expected_goals(
    df: pd.DataFrame,
    league: str,
    home_team: str,
    away_team: str,
    seasons: Optional[Iterable[str]] = None,
    kappa: float = 3.0,
    recent_weight: float = 0.25,
    recent_n: int = 6,
) -> ExpectedGoals:
    df = df.copy()
    df["country"] = df["country"].astype("string").str.upper().str.strip()
    league_code = (league or "").upper().strip()
    df = df[df["country"] == league_code]

    if seasons is not None and "Stagione" in df.columns:
        seasons = [str(s) for s in seasons]
        df = df[df["Stagione"].astype("string").isin(seasons)]

    if df.empty:
        return ExpectedGoals(home=1.2, away=1.0)

    L = _league_averages(df)

    h_home = _team_context_stats(df, home_team, "Home")
    a_away = _team_context_stats(df, away_team, "Away")

    home_sc_rate = _shrink_mean(h_home.scored_sum, h_home.n, L.home_goals, kappa)
    home_cc_rate = _shrink_mean(h_home.conceded_sum, h_home.n, L.away_goals, kappa)
    away_sc_rate = _shrink_mean(a_away.scored_sum, a_away.n, L.away_goals, kappa)
    away_cc_rate = _shrink_mean(a_away.conceded_sum, a_away.n, L.home_goals, kappa)

    home_attack = home_sc_rate / max(L.home_goals, 1e-9)
    away_defence = away_cc_rate / max(L.home_goals, 1e-9)
    away_attack = away_sc_rate / max(L.away_goals, 1e-9)
    home_defence = home_cc_rate / max(L.away_goals, 1e-9)

    lam_home_base = L.home_goals * home_attack * away_defence
    lam_away_base = L.away_goals * away_attack * home_defence

    if recent_weight > 0:
        h_recent = _recent_context_stats(df, home_team, "Home", last_n=recent_n)
        a_recent = _recent_context_stats(df, away_team, "Away", last_n=recent_n)

        h_att_r = h_recent.scored_mean / max(L.home_goals, 1e-9) if h_recent.n > 0 else home_attack
        a_def_r = a_recent.conceded_mean / max(L.home_goals, 1e-9) if a_recent.n > 0 else away_defence

        a_att_r = a_recent.scored_mean / max(L.away_goals, 1e-9) if a_recent.n > 0 else away_attack
        h_def_r = h_recent.conceded_mean / max(L.away_goals, 1e-9) if h_recent.n > 0 else home_defence

        lam_h_r = L.home_goals * h_att_r * a_def_r
        lam_a_r = L.away_goals * a_att_r * h_def_r

        lam_home = (1 - recent_weight) * lam_home_base + recent_weight * lam_h_r
        lam_away = (1 - recent_weight) * lam_away_base + recent_weight * lam_a_r
    else:
        lam_home, lam_away = lam_home_base, lam_away_base

    lam_home = float(np.clip(lam_home, 0.05, 6.0))
    lam_away = float(np.clip(lam_away, 0.05, 6.0))
    return ExpectedGoals(home=lam_home, away=lam_away)


# ===========================
# Matrice punteggi esatti
# ===========================
@dataclass
class ScoreGrid:
    matrix: pd.DataFrame
    home_lambda: float
    away_lambda: float
    rho: float
    max_goals: int


def correct_score_matrix(lam_home: float, lam_away: float, rho: float = -0.05, max_goals: int = 6) -> ScoreGrid:
    max_goals = int(max(1, max_goals))
    px = np.array([_poisson_pmf(lam_home, k) for k in range(max_goals + 1)], dtype=float)
    py = np.array([_poisson_pmf(lam_away, k) for k in range(max_goals + 1)], dtype=float)
    mat = np.outer(px, py)
    for x in (0, 1):
        for y in (0, 1):
            mat[x, y] *= _dixon_coles_tau(x, y, lam_home, lam_away, rho)
    s = mat.sum()
    if s > 0:
        mat = mat / s
    df = pd.DataFrame(mat, index=[f"H{x}" for x in range(max_goals + 1)],
                      columns=[f"A{y}" for y in range(max_goals + 1)])
    return ScoreGrid(matrix=df, home_lambda=lam_home, away_lambda=lam_away, rho=rho, max_goals=max_goals)


# ===========================
# Helper punteggi
# ===========================
def top_correct_scores(grid: ScoreGrid, top_n: int = 10) -> List[Tuple[str, float, float]]:
    mat = grid.matrix.values
    out = []
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            p = float(mat[i, j])
            out.append((f"{i}-{j}", p, (1.0 / p) if p > 0 else float("inf")))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_n]


def market_ev_for_scores(grid: ScoreGrid, market_prices: Dict[str, float], commission: float = 0.0) -> pd.DataFrame:
    rows = []
    for s, price in market_prices.items():
        try:
            h, a = s.split("-"); h = int(h); a = int(a)
        except Exception:
            continue
        if h <= grid.max_goals and a <= grid.max_goals:
            p = float(grid.matrix.values[h, a])
            ev = price * p * (1 - commission) - 1
            rows.append({"Score": s, "Prob %": round(p * 100, 2), "Quota": price, "EV": round(ev, 3)})
    df = pd.DataFrame(rows).sort_values("EV", ascending=False)
    return df.reset_index(drop=True)


# ===========================
# Calibrazione automatica parametri
# ===========================
def _score_prob(hg: int, ag: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Probabilità del punteggio osservato (hg, ag) con DC correction."""
    p = _poisson_pmf(lam_h, hg) * _poisson_pmf(lam_a, ag)
    if (hg in (0, 1)) and (ag in (0, 1)):
        p *= _dixon_coles_tau(hg, ag, lam_h, lam_a, rho)
    return max(p, 1e-12)


def _prepare_league(df: pd.DataFrame, league_code: str, seasons: Optional[Iterable[str]] = None) -> pd.DataFrame:
    d = df.copy()
    d["country"] = d["country"].astype("string").str.upper().str.strip()
    d = d[d["country"] == (league_code or "").upper().strip()]
    if seasons is not None and "Stagione" in d.columns:
        seasons = [str(s) for s in seasons]
        d = d[d["Stagione"].astype("string").isin(seasons)]
    # ordina cronologicamente se possibile
    if "Data" in d.columns:
        d["_d_"] = pd.to_datetime(d["Data"], errors="coerce")
        d = d.sort_values("_d_").drop(columns=["_d_"])
    return d.reset_index(drop=True)


def _rolling_nll_for_grid(
    df_league: pd.DataFrame,
    kappa: float,
    recent_weight: float,
    recent_n: int,
    rho: float,
    warmup: int = 80,
    step: int = 2,
    max_eval: int = 800,
) -> Tuple[float, int]:
    """
    Negative Log-Likelihood "rolling": per ogni match i usa solo le partite precedenti.
    Ritorna (NLL aggregata, numero match valutati).
    """
    n = len(df_league)
    if n <= warmup + 5:
        return float("inf"), 0

    nll = 0.0
    evaluated = 0

    # per risparmiare, limitiamo #valutazioni
    end = n
    start = min(warmup, n - 1)
    idxs = list(range(start, end, max(1, step)))[:max_eval]

    for i in idxs:
        row = df_league.iloc[i]
        hg = row.get("Home Goal FT", np.nan); ag = row.get("Away Goal FT", np.nan)
        if pd.isna(hg) or pd.isna(ag):
            continue
        hg = int(hg); ag = int(ag)

        # dataset passato (niente look-ahead)
        past = df_league.iloc[:i].copy()
        home = str(row["Home"]); away = str(row["Away"])

        xg = estimate_expected_goals(
            df=past,
            league=str(row["country"]),
            home_team=home,
            away_team=away,
            seasons=None,  # già filtrato
            kappa=float(kappa),
            recent_weight=float(recent_weight),
            recent_n=int(recent_n),
        )
        p = _score_prob(hg, ag, xg.home, xg.away, float(rho))
        nll += -math.log(p)
        evaluated += 1

    return (nll, evaluated)


def recommend_dc_params(
    df: pd.DataFrame,
    league_code: str,
    seasons: Optional[Iterable[str]] = None,
    *,
    kappa_grid: Iterable[float] = (1.5, 3.0, 4.5, 6.0, 8.0),
    recent_w_grid: Iterable[float] = (0.0, 0.15, 0.25, 0.35),
    recent_n_grid: Iterable[int] = (4, 6, 8),
    rho_grid: Iterable[float] = (-0.12, -0.08, -0.05, -0.02, 0.00, 0.03),
    warmup: int = 80,
    step: int = 2,
    max_eval: int = 800,
) -> Dict[str, float]:
    """
    Cerca i parametri che minimizzano la Negative Log-Likelihood rolling.
    Ritorna un dizionario con i valori consigliati.
    """
    d = _prepare_league(df, league_code, seasons)
    if len(d) < warmup + 10:
        # poco storico: suggerimenti prudenti
        return {"rho": -0.05, "kappa": 3.0, "recent_weight": 0.25, "recent_n": 6, "evaluated": 0, "nll_per_match": np.nan}

    best = {"nll": float("inf"), "evaluated": 0, "rho": -0.05, "kappa": 3.0, "recent_weight": 0.25, "recent_n": 6}

    # griglia
    for kappa in kappa_grid:
        for rw in recent_w_grid:
            for rn in recent_n_grid:
                for rho in rho_grid:
                    nll, ev = _rolling_nll_for_grid(d, kappa, rw, rn, rho, warmup=warmup, step=step, max_eval=max_eval)
                    if ev == 0:
                        continue
                    if nll < best["nll"]:
                        best.update({"nll": nll, "evaluated": ev, "rho": rho, "kappa": kappa, "recent_weight": rw, "recent_n": rn})

    if best["evaluated"] > 0:
        best["nll_per_match"] = round(best["nll"] / best["evaluated"], 6)
    else:
        best["nll_per_match"] = np.nan
    return {k: best[k] for k in ("rho", "kappa", "recent_weight", "recent_n", "evaluated", "nll_per_match")}


# ===========================
# Pannello Streamlit (UI)
# ===========================
def run_correct_score_panel(
    df: pd.DataFrame,
    league_code: str,
    home_team: str,
    away_team: str,
    seasons: Optional[Iterable[str]] = None,
    default_rho: float = -0.05,
    default_kappa: float = 3.0,
    default_recent_weight: float = 0.25,
    default_recent_n: int = 6,
    default_max_goals: int = 6,
):
    if st is None:
        raise RuntimeError("Streamlit non disponibile in questo ambiente.")

    st.subheader("🎯 Correct Score Prediction (Poisson + Dixon–Coles)")

    with st.expander("Parametri modello", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            rho = st.number_input("Rho (Dixon-Coles)", key="cs_rho", value=float(default_rho),
                                  step=0.01, min_value=-0.5, max_value=0.5)
        with c2:
            kappa = st.number_input("Kappa shrinkage", key="cs_kappa", value=float(default_kappa),
                                    step=0.5, min_value=0.0, max_value=50.0)
        with c3:
            recent_w = st.slider("Peso forma recente", key="cs_recent_w",
                                 min_value=0.0, max_value=1.0, value=float(default_recent_weight))
        with c4:
            recent_n = st.number_input("N partite recenti", key="cs_recent_n",
                                       value=int(default_recent_n), min_value=1, step=1)
        with c5:
            gmax = st.number_input("Max gol tabella", key="cs_gmax",
                                   value=int(default_max_goals), min_value=3, max_value=10, step=1)

    # ---- SUGGERITORE PARAMETRI SU STORICO
    with st.expander("🧪 Suggerisci parametri dai dati di lega", expanded=False):
        st.caption("Calibrazione rolling su storico della lega: minimizza la verosimiglianza del punteggio esatto (no look-ahead).")
        colb1, colb2, colb3 = st.columns([1, 1, 2])
        with colb1:
            warmup = st.number_input("Warmup iniziale", value=80, min_value=40, step=10, help="Numero di match iniziali ignorati per stimare i tassi.")
        with colb2:
            max_eval = st.number_input("Max valutazioni", value=800, min_value=200, step=100, help="Per contenere i tempi.")
        with colb3:
            step = st.slider("Step (salta match)", min_value=1, max_value=5, value=2, help="1 = valuta tutti i match (più lento)")

        if st.button("🔎 Calcola parametri consigliati", use_container_width=True):
            with st.spinner("Calibrazione in corso..."):
                # facciamo una cache semplice per non ripetere calcoli identici
                @st.cache_data(show_spinner=False)
                def _cached_reco(_df, _league, _seasons, _warmup, _step, _max_eval):
                    return recommend_dc_params(_df, _league, _seasons, warmup=_warmup, step=_step, max_eval=_max_eval)
                rec = _cached_reco(df, league_code, seasons, warmup, step, max_eval)

            st.success(f"Consigliato → ρ: **{rec['rho']}**, κ: **{rec['kappa']}**, "
                       f"peso recente: **{rec['recent_weight']}**, N recente: **{rec['recent_n']}** "
                       f"(eval: {rec['evaluated']} match, NLL/match: {rec['nll_per_match']})")

            apply = st.checkbox("Applica questi valori ai controlli", value=True)
            if apply:
                st.session_state["cs_rho"] = float(rec["rho"])
                st.session_state["cs_kappa"] = float(rec["kappa"])
                st.session_state["cs_recent_w"] = float(rec["recent_weight"])
                st.session_state["cs_recent_n"] = int(rec["recent_n"])
                st.info("Parametri applicati. Puoi ricalcolare la matrice con i nuovi valori.")

    # ---- Stima λ e matrice
    xg = estimate_expected_goals(
        df=df,
        league=league_code,
        home_team=home_team,
        away_team=away_team,
        seasons=seasons,
        kappa=float(st.session_state.get("cs_kappa", kappa)),
        recent_weight=float(st.session_state.get("cs_recent_w", recent_w)),
        recent_n=int(st.session_state.get("cs_recent_n", recent_n)),
    )

    st.caption(f"λ attesi → Home: **{xg.home:.3f}**, Away: **{xg.away:.3f}** (league-adjusted + recent form)")
    grid = correct_score_matrix(xg.home, xg.away, rho=float(st.session_state.get("cs_rho", rho)), max_goals=int(gmax))

    # ---- Top score
    top = top_correct_scores(grid, top_n=12)
    df_top = pd.DataFrame([{"Score": s, "Prob %": round(p * 100, 2), "Quota implicita": round(q, 2)} for s, p, q in top])
    st.markdown("#### 📈 Top correct score (probabilità più alte)")
    st.dataframe(df_top, use_container_width=True, hide_index=True)

    # ---- Heatmap
    st.markdown("#### 🔥 Heatmap probabilità punteggio")
    df_heat = grid.matrix.copy()
    df_heat.index = [int(x[1:]) for x in df_heat.index]
    df_heat.columns = [int(x[1:]) for x in df_heat.columns]
    df_heat_long = df_heat.reset_index(names="Home").melt("Home", var_name="Away", value_name="Prob")
    df_heat_long["Prob %"] = df_heat_long["Prob"] * 100

    if alt is not None:
        heat = (
            alt.Chart(df_heat_long)
            .mark_rect()
            .encode(
                x=alt.X("Away:O", title="Gol Away"),
                y=alt.Y("Home:O", title="Gol Home"),
                color=alt.Color("Prob %:Q", title="Prob %", scale=alt.Scale(scheme="greens")),
                tooltip=[alt.Tooltip("Home:O", title="Home"),
                         alt.Tooltip("Away:O", title="Away"),
                         alt.Tooltip("Prob %:Q", title="Prob %", format=".2f")],
            ).properties(height=300)
        )
        text = alt.Chart(df_heat_long).mark_text(baseline="middle", fontSize=11)\
            .encode(x="Away:O", y="Home:O", text=alt.Text("Prob %:Q", format=".1f"))
        st.altair_chart(heat + text, use_container_width=True)
    else:
        st.dataframe(df_heat.style.format("{:.2%}"), use_container_width=True)

    # ---- EV correct score (facoltativo)
    st.markdown("#### 💰 EV corretto punteggio (facoltativo)")
    with st.expander("Inserisci qualche quota CS per valutare EV", expanded=False):
        examples = "1-0:7.5, 2-0:12.0, 2-1:9.2, 1-1:6.8, 0-0:10.0"
        txt = st.text_input("Score:Quota separati da virgola", value=examples, help="es: 1-0:7.5, 2-1:9.2")
        commission = st.number_input("Commissione (exchange)", value=0.0, min_value=0.0, max_value=0.1, step=0.01)
        if st.button("Calcola EV", use_container_width=True):
            mp = {}
            for chunk in txt.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    s, q = chunk.split(":")
                    mp[s.strip()] = float(q)
                except Exception:
                    pass
            if mp:
                df_ev = market_ev_for_scores(grid, mp, commission=float(commission))
                st.dataframe(df_ev, use_container_width=True, hide_index=True)
            else:
                st.info("Inserisci almeno una coppia score:quota valida.")
