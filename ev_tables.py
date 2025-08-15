# ev_tables.py
from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
import pandas as pd

try:
    import streamlit as st  # opzionale: il modulo funziona anche senza Streamlit
except Exception:  # pragma: no cover
    st = None  # type: ignore


# ---------------------------
# Helpers interni
# ---------------------------
def _market_prob(df: pd.DataFrame, market: str, line: float | None) -> float:
    """Probabilità (%) storica del verificarsi del mercato su df leggero (solo FT goal)."""
    if df is None or df.empty:
        return 0.0
    hg = pd.to_numeric(df.get("Home Goal FT"), errors="coerce").fillna(0)
    ag = pd.to_numeric(df.get("Away Goal FT"), errors="coerce").fillna(0)
    if market == "BTTS":
        ok = ((hg > 0) & (ag > 0)).mean()
    else:
        ok = (hg.add(ag) > float(line)).mean() if line is not None else 0.0
    return round(float(ok) * 100, 2)


def _quality_label(n: int) -> str:
    if n >= 50: return "ALTO"
    if n >= 20: return "MEDIO"
    return "BASSO"


# ---------------------------
# API pubblica
# ---------------------------
def build_ev_table(
    df_home_ctx: pd.DataFrame,
    df_away_ctx: pd.DataFrame,
    df_h2h: pd.DataFrame,
    squadra_casa: str,
    squadra_ospite: str,
    quota_ov15: float,
    quota_ov25: float,
    quota_ov35: float,
    quota_btts: float,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Ritorna (df_ev, best) con le stesse colonne/semantica usate in pre_match/squadre.
    EV = quota × (p/100) − 1
    """
    markets = [
        ("Over 1.5", 1.5, float(quota_ov15)),
        ("Over 2.5", 2.5, float(quota_ov25)),
        ("Over 3.5", 3.5, float(quota_ov35)),
        ("BTTS",    None, float(quota_btts)),
    ]

    rows, candidates = [], []
    n_h, n_a, n_h2h = len(df_home_ctx), len(df_away_ctx), len(df_h2h)
    qual_blnd = _quality_label(n_h + n_a)
    qual_h2h  = _quality_label(n_h2h)

    for name, line, q in markets:
        p_home = _market_prob(df_home_ctx, name, line)
        p_away = _market_prob(df_away_ctx, name, line)
        p_blnd = round((p_home + p_away) / 2, 2) if (p_home > 0 or p_away > 0) else 0.0
        p_h2h  = _market_prob(df_h2h, name, line)

        ev_home = round(q * (p_home / 100) - 1, 2)
        ev_away = round(q * (p_away / 100) - 1, 2)
        ev_blnd = round(q * (p_blnd / 100) - 1, 2)
        ev_h2h  = round(q * (p_h2h  / 100) - 1, 2)

        rows.append({
            "Mercato": name, "Quota": q,
            f"{squadra_casa} @Casa %": p_home, f"EV {squadra_casa}": ev_home,
            f"{squadra_ospite} @Trasferta %": p_away, f"EV {squadra_ospite}": ev_away,
            "Blended %": p_blnd, "EV Blended": ev_blnd, "Qualità Blended": qual_blnd,
            "Head-to-Head %": p_h2h, "EV H2H": ev_h2h, "Qualità H2H": qual_h2h,
            "Match H": n_h, "Match A": n_a, "Match H2H": n_h2h,
        })

        candidates.extend([
            {"scope": "Blended",      "mercato": name, "quota": q, "prob": p_blnd, "ev": ev_blnd, "campione": n_h + n_a, "qualita": qual_blnd},
            {"scope": "Head-to-Head", "mercato": name, "quota": q, "prob": p_h2h,  "ev": ev_h2h,  "campione": n_h2h,    "qualita": qual_h2h},
        ])

    df_ev = pd.DataFrame(rows)

    # best = EV più alto positivo; a parità di EV preferisci Blended
    best = None
    for c in sorted(candidates, key=lambda x: (x["ev"], 1 if x["scope"] == "Blended" else 0), reverse=True):
        if c["ev"] > 0:
            best = c
            break

    return df_ev, best


# Variante cache-friendly (stesse firme)
if st is not None:
    @st.cache_data(show_spinner=False, ttl=900)
    def build_ev_table_cached(
        df_home_ctx: pd.DataFrame,
        df_away_ctx: pd.DataFrame,
        df_h2h: pd.DataFrame,
        squadra_casa: str,
        squadra_ospite: str,
        quota_ov15: float,
        quota_ov25: float,
        quota_ov35: float,
        quota_btts: float,
    ):
        return build_ev_table(
            df_home_ctx, df_away_ctx, df_h2h,
            squadra_casa, squadra_ospite,
            quota_ov15, quota_ov25, quota_ov35, quota_btts
        )
else:  # fallback senza Streamlit
    def build_ev_table_cached(*args, **kwargs):
        return build_ev_table(*args, **kwargs)
