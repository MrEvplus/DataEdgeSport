# app.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import date

# --- Import progetto (robusti) ---
try:
    from supabase import create_client  # opzionale: se assente non blocca l'app
except Exception:
    create_client = None

# utils centralizza secrets, loader e label
from utils import (
    SUPABASE_URL,
    SUPABASE_KEY,
    load_data_from_supabase,
    load_data_from_file,
    label_match,
)

# Moduli funzionali (manteniamo la struttura esistente)
from macros import run_macro_stats
from squadre import run_team_stats
from pre_match import run_pre_match
from correct_score_ev_sezione import run_correct_score_ev
from analisi_live_minuto import run_live_minuto_analysis
from partite_del_giorno import run_partite_del_giorno
from reverse_engineering import run_reverse_engineering

# Moduli opzionali (se non ci sono, ignoriamo)
try:
    from api_football_utils import get_fixtures_today_for_countries  # noqa: F401
except Exception:
    pass

try:
    from ai_inference import run_ai_inference  # noqa: F401
except Exception:
    pass

try:
    from mappa_leghe_supabase import run_mappa_leghe_supabase  # noqa: F401
except Exception:
    pass


# -------------------------------------------------------
# FUNZIONI DI SUPPORTO
# -------------------------------------------------------
def get_league_mapping() -> dict:
    """
    Recupera (opzionalmente) la mappa codice->nome lega da Supabase.
    Se supabase client o tabella non disponibili, ritorna dict vuoto.
    """
    try:
        if not create_client or not SUPABASE_URL or not SUPABASE_KEY:
            return {}
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        data = supabase.table("league_mapping").select("*").execute().data
        mapping = {r["code"]: r["league_name"] for r in data}
        mapping["Tutti"] = "Tutti i Campionati"
        return mapping
    except Exception:
        return {}


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce")
    except Exception:
        return pd.to_datetime(series.astype(str), errors="coerce")


# -------------------------------------------------------
# CONFIGURAZIONE PAGINA
# -------------------------------------------------------
st.set_page_config(page_title="Trading Dashboard", page_icon="⚽", layout="wide")
st.sidebar.title("📊 Trading Dashboard")

# -------------------------------------------------------
# MENU PRINCIPALE
# -------------------------------------------------------
menu_option = st.sidebar.radio(
    "Naviga tra le sezioni:",
    [
        "Macro Stats per Campionato",
        "Statistiche per Squadre",
        "Confronto Pre Match",
        "Correct Score EV",
        "Analisi Live da Minuto",
        "Partite del Giorno",
        "🧠 Reverse Engineering EV+",
    ],
    key="menu_principale",
)

# -------------------------------------------------------
# SELEZIONE ORIGINE DATI
# -------------------------------------------------------
origine_dati = st.sidebar.radio("Seleziona origine dati:", ["Supabase", "Upload Manuale"], key="origine_dati")

if origine_dati == "Supabase":
    df, db_selected = load_data_from_supabase(selectbox_key="campionato_supabase")
else:
    df, db_selected = load_data_from_file()

# Mappatura (opzionale) dei codici campionato in nomi leggibili
league_dict = get_league_mapping()
db_selected = league_dict.get(db_selected, db_selected)

# Stato persistente per selezione squadre/campionato
if "squadra_casa" not in st.session_state:
    st.session_state["squadra_casa"] = ""
if "squadra_ospite" not in st.session_state:
    st.session_state["squadra_ospite"] = ""
if "campionato_corrente" not in st.session_state:
    st.session_state["campionato_corrente"] = db_selected
else:
    if st.session_state["campionato_corrente"] != db_selected:
        # reset selezioni squadra quando cambi campionato
        st.session_state["squadra_casa"] = ""
        st.session_state["squadra_ospite"] = ""
        st.session_state["campionato_corrente"] = db_selected

# -------------------------------------------------------
# MAPPING COLONNE COMPLETO E PULIZIA
# (nota: allineiamo "Odd home" minuscolo per coerenza con label_match in utils.py)
# -------------------------------------------------------
col_map = {
    "country": "country",
    "sezonul": "Stagione",
    "datameci": "Data",
    "orameci": "Orario",
    "etapa": "Round",
    "txtechipa1": "Home",
    "txtechipa2": "Away",
    "scor1": "Home Goal FT",
    "scor2": "Away Goal FT",
    "scorp1": "Home Goal 1T",
    "scorp2": "Away Goal 1T",
    "place1": "Posizione Classifica Generale",
    "place1a": "Posizione Classifica Home",
    "place2": "Posizione Classifica Away Generale",
    "place2d": "Posizione classifica away",
    # *** IMPORTANTE: usare "Odd home" coerente con utils.label_match ***
    "cotaa": "Odd home",
    "cotad": "Odd Away",
    "cotae": "Odd Draw",
    "cotao0": "Odd Over 0.5",
    "cotao1": "Odd Over 1.5",
    "cotao": "Odd Over 2.5",
    "cotao3": "Odd Over 3.5",
    "cotao4": "Odd Over 4.5",
    "cotau0": "Odd Under 0.5",
    "cotau1": "Odd Under 1.5",
    "cotau": "Odd Under 2.5",
    "cotau3": "Odd Under 3.5",
    "cotau4": "Odd Under 4.5",
    "gg": "GG",
    "ng": "NG",
    "elohomeo": "ELO Home",
    "eloawayo": "ELO Away",
    "formah": "Form Home",
    "formaa": "Form Away",
    "suth": "Tiri Totali Home FT",
    "suth1": "Tiri Home 1T",
    "suth2": "Tiri Home 2T",
    "suta": "Tiri Totali Away FT",
    "suta1": "Tiri Away 1T",
    "suta2": "Tiri Away 2T",
    "sutht": "Tiri in Porta Home FT",
    "sutht1": "Tiri in Porta Home 1T",
    "sutht2": "Tiri in Porta Home 2T",
    "sutat": "Tiri in Porta Away FT",
    "sutat1": "Tiri in Porta Away 1T",
    "sutat2": "Tiri in Porta Away 2T",
    "mgolh": "Minuti Goal Home",
    "gh1": "Home Goal 1 (min)",
    "gh2": "Home Goal 2 (min)",
    "gh3": "Home Goal 3 (min)",
    "gh4": "Home Goal 4 (min)",
    "gh5": "Home Goal 5 (min)",
    "gh6": "Home Goal 6 (min)",
    "gh7": "Home Goal 7 (min)",
    "gh8": "Home Goal 8 (min)",
    "gh9": "Home Goal 9 (min)",
    "mgola": "Minuti Goal Away",
    "ga1": "Away Goal 1 (min)",
    "ga2": "Away Goal 2 (min)",
    "ga3": "Away Goal 3 (min)",
    "ga4": "Away Goal 4 (min)",
    "ga5": "Away Goal 5 (min)",
    "ga6": "Away Goal 6 (min)",
    "ga7": "Away Goal 7 (min)",
    "ga8": "Away Goal 8 (min)",
    "ga9": "Away Goal 9 (min)",
    "stare": "Stare",
    "codechipa1": "CodeChipa1",
    "codechipa2": "CodeChipa2",
}

# rename sicuro (se alcune colonne non esistono, pandas ignora)
df = df.rename(columns=col_map)

# normalizzazione leggera dei nomi
df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(r"[\n\r\t]", "", regex=True)
    .str.replace(r"\s+", " ", regex=True)
)

# Etichetta "Label" (se manca). Usa label_match riga per riga (coerenza con progetti esistenti).
if "Label" not in df.columns:
    # Per evitare errori se le quote mancano:
    if not set(["Odd home", "Odd Away"]).issubset(df.columns):
        df["Label"] = "Others"
    else:
        df["Label"] = df.apply(label_match, axis=1)

# -------------------------------------------------------
# FILTRO STAGIONI (Scorciatoie + Personalizza)
# -------------------------------------------------------
if "Stagione" in df.columns:
    stagioni_disponibili = sorted(df["Stagione"].dropna().astype(str).unique())

    opzione_range = st.sidebar.selectbox(
        "Seleziona un intervallo stagioni predefinito:",
        ["Tutte", "Ultime 3", "Ultime 5", "Ultime 10", "Personalizza"],
        key="selettore_range_stagioni",
    )

    if opzione_range == "Tutte":
        stagioni_scelte = stagioni_disponibili
    elif opzione_range == "Ultime 3":
        stagioni_scelte = stagioni_disponibili[-3:]
    elif opzione_range == "Ultime 5":
        stagioni_scelte = stagioni_disponibili[-5:]
    elif opzione_range == "Ultime 10":
        stagioni_scelte = stagioni_disponibili[-10:]
    else:
        stagioni_scelte = st.sidebar.multiselect(
            "Seleziona manualmente le stagioni da includere:",
            options=stagioni_disponibili,
            default=stagioni_disponibili,
            key="multiselect_stagioni_personalizzate",
        )

    if stagioni_scelte:
        df = df[df["Stagione"].astype(str).isin(stagioni_scelte)]

# -------------------------------------------------------
# INFO DATASET E GUARD-RAIL
# -------------------------------------------------------
with st.expander("✅ Colonne presenti nel dataset", expanded=False):
    st.write(list(df.columns))

if df.empty:
    st.error("⚠️ Nessun dato disponibile dopo i filtri applicati.")
    st.stop()

if "Home" not in df.columns or "Away" not in df.columns:
    st.error("⚠️ Colonne 'Home' e/o 'Away' mancanti nel dataset selezionato.")
    st.stop()

# Se c'è "Data", ripulisci e mostra solo partite fino ad oggi (niente future)
if "Data" in df.columns:
    df["Data"] = _safe_to_datetime(df["Data"])
    today = pd.Timestamp.today().normalize()
    df = df[(df["Data"].isna()) | (df["Data"] <= today)]

# -------------------------------------------------------
# KPI RAPIDI (facoltativi, non invasivi)
# -------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Partite DB", f"{len(df):,}")
if "country" in df: c2.metric("Campionati", df["country"].nunique())
if "Home" in df: c3.metric("Squadre", pd.concat([df["Home"], df["Away"]]).nunique())
if "Stagione" in df: c4.metric("Stagioni", df["Stagione"].nunique())

st.caption(f"Campionato selezionato: **{db_selected}**")

# -------------------------------------------------------
# ROUTING SEZIONI
# -------------------------------------------------------
if menu_option == "Macro Stats per Campionato":
    run_macro_stats(df, db_selected)

elif menu_option == "Statistiche per Squadre":
    run_team_stats(df, db_selected)

elif menu_option == "Confronto Pre Match":
    run_pre_match(df, db_selected)

elif menu_option == "Correct Score EV":
    run_correct_score_ev(df, db_selected)

elif menu_option == "Analisi Live da Minuto":
    run_live_minuto_analysis(df)

elif menu_option == "Partite del Giorno":
    run_partite_del_giorno(df, db_selected)

elif menu_option == "🧠 Reverse Engineering EV+":
    run_reverse_engineering(df)
