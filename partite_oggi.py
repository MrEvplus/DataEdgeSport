# partite_oggi.py

import streamlit as st
from api_football_utils import get_fixtures_today

def run_partite_oggi():
    st.title("📅 Partite in programma oggi")

    # API Key già definita
    api_key = "f3c8f9d1e309ec5c56b571990678e563"

    country_filter = st.text_input(
        "Filtra per Nazione (opzionale, es. Italy)", ""
    )

    if st.button("Carica Partite di Oggi"):
        with st.spinner("Recupero partite..."):
            df = get_fixtures_today(api_key, country_filter if country_filter else None)

        if df.empty:
            st.info("Nessuna partita trovata.")
        else:
            st.success(f"Trovate {len(df)} partite.")
            st.dataframe(df, use_container_width=True)

            # Facoltativo: Download CSV
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="💾 Scarica CSV",
                data=csv,
                file_name="partite_oggi.csv",
                mime="text/csv"
            )
