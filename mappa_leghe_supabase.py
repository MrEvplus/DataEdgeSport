import streamlit as st
import pandas as pd
from supabase import create_client
from utils import SUPABASE_URL, SUPABASE_KEY

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def run_mappa_leghe_supabase():
    st.header("🗾️ Mappatura Manuale Campionati su Supabase")

    origine = st.radio("Origine dati:", ["Supabase", "Upload Manuale"], index=0)

    if origine == "Supabase":
        try:
            response = supabase.table("league_mapping").select("*").execute()
            records = response.data
            df = pd.DataFrame(records)

            if df.empty or not all(col in df.columns for col in ["excel_country", "excel_league", "db_league_code"]):
                st.error("❌ La tabella deve contenere le colonne 'excel_country', 'excel_league' e 'db_league_code'.")
                return

            df["lookup_key"] = df["excel_country"].astype(str).str.strip().str.upper() + "__" + df["excel_league"].astype(str).str.strip().str.upper()

            st.success("✅ Dati correttamente caricati da Supabase.")
            st.dataframe(df)

        except Exception as e:
            st.error(f"Errore durante il recupero dei dati da Supabase: {e}")

    else:
        uploaded_file = st.file_uploader(
            "Carica un file CSV con colonne 'excel_country', 'excel_league', 'db_league_code':", type="csv")
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file)
                required_cols = ["excel_country", "excel_league", "db_league_code"]
                if not all(col in df.columns for col in required_cols):
                    st.error("❌ Il file deve contenere le colonne 'excel_country', 'excel_league', 'db_league_code'.")
                    return

                df["excel_country"] = df["excel_country"].astype(str).str.strip().str.upper()
                df["excel_league"] = df["excel_league"].astype(str).str.strip().str.upper()
                df["lookup_key"] = df["excel_country"] + "__" + df["excel_league"]

                st.success("✅ File caricato correttamente.")
                st.dataframe(df)

                if st.button("📄 Salva su Supabase"):
                    supabase.table("league_mapping").delete().neq("id", 0).execute()
                    insert_data = df.to_dict(orient="records")
                    for chunk in [insert_data[i:i + 50] for i in range(0, len(insert_data), 50)]:
                        supabase.table("league_mapping").insert(chunk).execute()

                    st.success("✅ Dati salvati correttamente su Supabase.")

            except Exception as e:
                st.error(f"Errore durante la lettura o il salvataggio del file: {e}")
