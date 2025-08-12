import streamlit as st
import pandas as pd
from utils import load_data_from_supabase, load_data_from_file

st.set_page_config(page_title="Mappatura Leghe", layout="wide")
st.title("🗺️ Mappatura Manuale Campionati (Codici → Nomi reali)")

# Origine dati
origine = st.radio("Origine dati:", ["Supabase", "Upload Manuale"])

if origine == "Supabase":
    df, _ = load_data_from_supabase()
else:
    df, _ = load_data_from_file()

# Controllo colonna "country"
if "country" not in df.columns:
    st.error("❌ Il file caricato non contiene la colonna 'country'.")
    st.stop()

# Codici campionato unici
codici = sorted(df["country"].dropna().unique().tolist())

# Prova a caricare una mappatura esistente
mapping_file = "league_mapping.csv"
try:
    df_map = pd.read_csv(mapping_file)
    existing_mapping = dict(zip(df_map["code"], df_map["league_name"]))
except:
    existing_mapping = {}

# Costruisci dataframe editabile
rows = []
for code in codici:
    league_name = existing_mapping.get(code, "")
    rows.append({"code": code, "league_name": league_name})

df_editor = pd.DataFrame(rows)

st.markdown("### ✏️ Modifica il nome reale per ogni codice campionato")
df_edited = st.data_editor(df_editor, num_rows="dynamic", use_container_width=True)

# Bottone per salvare
if st.button("💾 Salva Mappatura"):
    try:
        df_edited.to_csv(mapping_file, index=False)
        st.success("✅ Mappatura salvata con successo in 'league_mapping.csv'")
    except Exception as e:
        st.error(f"❌ Errore nel salvataggio: {e}")

# Mostra mappatura attuale
if st.checkbox("👁️ Visualizza mappatura salvata"):
    try:
        df_map = pd.read_csv(mapping_file)
        st.dataframe(df_map, use_container_width=True)
    except:
        st.warning("⚠️ Nessuna mappatura salvata trovata.")
