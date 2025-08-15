# minutes.py — normalizzazione colonne minuti-gol + parser robusto "90+2"
from __future__ import annotations
import pandas as pd

_HOME_ALIASES = [
    "minuti goal segnato home", "Minuti Goal Home", "Minuti Gol Casa",
    "minuti_gol_home", "minuti_gol_casa",
]
_AWAY_ALIASES = [
    "minuti goal segnato away", "Minuti Goal Away", "Minuti Gol Trasferta",
    "minuti_gol_away", "minuti_gol_trasferta",
]

def _first_present(cols: list[str], frame_cols: pd.Index) -> str | None:
    for c in cols:
        if c in frame_cols:
            return c
    return None

def unify_goal_minute_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Garantisce la presenza delle colonne standard:
        - "minuti goal segnato home"
        - "minuti goal segnato away"
    Usando alias noti e, se assenti, ricostruendo dai campi gh1..gh9 / ga1..ga9.
    Ritorna SEMPRE stringhe (anche se vuote).
    """
    d = df.copy()

    # HOME
    ch = _first_present(_HOME_ALIASES, d.columns)
    if ch is not None:
        d["minuti goal segnato home"] = d[ch].fillna("").astype(str)
    else:
        # fallback da gh1..gh9
        gh_cols = [c for c in d.columns if c.lower().startswith("gh")]
        if gh_cols:
            d["minuti goal segnato home"] = (
                d[gh_cols]
                .astype(str)
                .apply(lambda r: ";".join(v for v in r.tolist() if isinstance(v, str) and v.strip() not in ("", "nan", "NaN")), axis=1)
            )
        else:
            d["minuti goal segnato home"] = ""

    # AWAY
    ca = _first_present(_AWAY_ALIASES, d.columns)
    if ca is not None:
        d["minuti goal segnato away"] = d[ca].fillna("").astype(str)
    else:
        # fallback da ga1..ga9
        ga_cols = [c for c in d.columns if c.lower().startswith("ga")]
        if ga_cols:
            d["minuti goal segnato away"] = (
                d[ga_cols]
                .astype(str)
                .apply(lambda r: ";".join(v for v in r.tolist() if isinstance(v, str) and v.strip() not in ("", "nan", "NaN")), axis=1)
            )
        else:
            d["minuti goal segnato away"] = ""

    return d

def parse_goal_times(val) -> list[int]:
    """
    Converte stringhe/list/array di minuti in lista di int (1..130).
    Supporta formati: "12;45+1; 90+2", "12, 34", [12, '45+2'], ecc.
    Ignora valori non numerici.
    """
    import numpy as np

    if val is None or (isinstance(val, float) and np.isnan(val)):  # type: ignore
        return []
    if isinstance(val, (list, tuple)):
        parts = [str(x) for x in val]
    else:
        s = str(val).strip()
        if not s:
            return []
        s = s.replace(",", ";").replace("[", "").replace("]", "").replace("’", "+").replace("′", "+")
        parts = [p.strip() for p in s.split(";") if p.strip()]

    out: list[int] = []
    for p in parts:
        if "+" in p:
            try:
                a, b = p.split("+", 1)
                out.append(int(float(a)) + int(float(b)))
            except Exception:
                continue
        else:
            try:
                out.append(int(float(p)))
            except Exception:
                continue
    # limiti ragionevoli
    out = [m for m in out if 0 < m <= 130]
    return sorted(out)
