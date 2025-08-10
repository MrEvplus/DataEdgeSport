"""
config.py - Configurazioni di base per il progetto Trading Dashboard V2.0
⚠️ Questo file NON deve contenere credenziali reali.
    Le credenziali vanno inserite in .streamlit/secrets.toml o in variabili d'ambiente.
"""

# -------------------------------------------------------
# 🔹 Credenziali (da st.secrets o .env)
# -------------------------------------------------------
SUPABASE_URL = None      # Es: "https://xxxx.supabase.co"
SUPABASE_KEY = None      # Es: "eyJhbGciOiJIUzI1..."

# -------------------------------------------------------
# 🔹 Impostazioni Generali
# -------------------------------------------------------
COMMISS_BETFAIR = 0.045  # Commissione standard Betfair (4.5%)
DATA_FORMAT = "%Y-%m-%d" # Formato data standard
TIMEZONE = "Europe/Rome" # Fuso orario di default

# -------------------------------------------------------
# 🔹 Mapping Colonne (DB → Nome leggibile)
# -------------------------------------------------------
COLUMN_MAPPING = {
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
    "cotaa": "Odd Home",
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
    "codechipa2": "CodeChipa2"
}

# -------------------------------------------------------
# 🔹 Campionati Manual Mapping (opzionale)
# -------------------------------------------------------
LEAGUE_MAPPING = {
    "Tutti": "Tutti i Campionati",
    # Esempio: "ITA1": "Serie A", "ENG1": "Premier League"
}
