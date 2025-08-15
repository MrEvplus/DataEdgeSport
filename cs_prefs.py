# cs_prefs.py — persistenza parametri Correct Score per lega
from __future__ import annotations
import json, os
from typing import Dict, Any

DEFAULTS = {
    "rho": -0.05,
    "kappa": 3.0,
    "recent_weight": 0.25,
    "recent_n": 6,
    "max_goals": 6,
}

_STORE = os.environ.get("CS_PREFS_PATH", "data/cs_params.json")

def _ensure_dir(path: str):
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def _load_all() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(_STORE):
        return {}
    try:
        with open(_STORE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_all(obj: Dict[str, Dict[str, Any]]):
    _ensure_dir(_STORE)
    with open(_STORE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def get_params(league: str) -> Dict[str, Any]:
    data = _load_all()
    rec = data.get(league.upper(), {})
    out = DEFAULTS.copy()
    out.update({k: rec.get(k, v) for k, v in DEFAULTS.items()})
    return out

def save_params(league: str, params: Dict[str, Any]) -> None:
    data = _load_all()
    cur = data.get(league.upper(), {})
    cur.update({k: params.get(k, cur.get(k, DEFAULTS[k])) for k in DEFAULTS})
    data[league.upper()] = cur
    _save_all(data)
