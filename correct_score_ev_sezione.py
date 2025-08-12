
import streamlit as st
import pandas as pd
from utils import label_match

def run_correct_score_ev(df, db_selected):
    st.title("🎯 Correct Score EV")

    # Assicuriamoci che Label sia presente
    if "Label" not in df.columns:
        df["Label"] = df.apply(label_match, axis=1)

    # Recupera squadre e quote dalla sessione
    squadra_casa = st.session_state.get("squadra_casa", "")
    squadra_ospite = st.session_state.get("squadra_ospite", "")
    odd_home = st.session_state.get("quota_home", 2.00)
    odd_away = st.session_state.get("quota_away", 3.00)

    if not squadra_casa or not squadra_ospite or squadra_casa == squadra_ospite:
        st.warning("⚠️ Seleziona prima le squadre in Confronto Pre Match.")
        return

    # Mostra i dati selezionati
    st.markdown(f"**Partita selezionata:** {squadra_casa} vs {squadra_ospite}")
    st.markdown(f"**Quote selezionate:** Casa {odd_home} | Ospite {odd_away}")

    # Calcola il Label dal match odds
    fake_row = {"Odd home": odd_home, "Odd Away": odd_away}
    label = label_match(fake_row)
    st.markdown(f"**Label identificato:** `{label}`")

    df_filtered = df[df["Label"] == label]
    df_filtered = df_filtered.dropna(subset=["Home Goal FT", "Away Goal FT"])

    if df_filtered.empty:
        st.warning("⚠️ Nessuna partita trovata per questo Label.")
        return

    df_filtered["CS"] = df_filtered["Home Goal FT"].astype(int).astype(str) + "-" + df_filtered["Away Goal FT"].astype(int).astype(str)
    cs_counts = df_filtered["CS"].value_counts().reset_index()
    cs_counts.columns = ["Correct Score", "Count"]
    total = cs_counts["Count"].sum()
    cs_counts["%"] = cs_counts["Count"] / total * 100

    st.markdown("## 🔢 Top 6 Risultati Esatti più frequenti")
    cs_top = cs_counts.head(6).copy()
    cs_top["Quota Inserita"] = [st.number_input(f"Quota per {row['Correct Score']}", min_value=1.01, step=0.01, key=f"quota_cs_{i}") for i, row in cs_top.iterrows()]
    cs_top["Probabilità"] = cs_top["%"] / 100
    cs_top["EV"] = (cs_top["Probabilità"] * cs_top["Quota Inserita"]) - 1
    cs_top["EV"] = cs_top["EV"].apply(lambda x: f"🟢 {x:.2%}" if x > 0 else f"🔴 {x:.2%}" if x < 0 else "0.00%")

    st.dataframe(cs_top[["Correct Score", "Count", "%", "Quota Inserita", "EV"]], use_container_width=True)
