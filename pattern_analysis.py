
import streamlit as st
import pandas as pd

def run_pattern_analysis(uploaded_df=None):
    st.title("🔍 Pattern Ricorrenti EV+")

    if uploaded_df is None:
        uploaded_file = st.file_uploader("📤 Carica il file CSV generato dal Reverse Batch:", type="csv")
        if uploaded_file is None:
            st.info("Carica un file per iniziare.")
            return
        df = pd.read_csv(uploaded_file)
    else:
        df = uploaded_df.copy()

    st.success(f"✅ {len(df)} righe caricate per l'analisi.")

    # Pulisci EV
    df["EV Over %"] = pd.to_numeric(df["EV Over %"], errors="coerce")
    df["Profitto"] = pd.to_numeric(df["Profitto"], errors="coerce")
    df["Label"] = df["Label"].fillna("Unknown")

    # Raggruppa per label
    group = df.groupby("Label").agg(
        Partite=("Match", "count"),
        EV_medio=("EV Over %", "mean"),
        ROI_totale=("Profitto", "sum"),
        ROI_medio=("Profitto", "mean"),
        %Win_Over=("Esito", lambda x: (x == "✅").mean() * 100)
    ).reset_index()

    group = group.round(2)
    group = group.sort_values(by="EV_medio", ascending=False)

    st.markdown("### 🧠 Pattern per Label")
    st.dataframe(group, use_container_width=True)

    # Filtra pattern EV+ forti
    st.markdown("### 🟢 Pattern EV+ con ROI medio positivo e almeno 5 partite")
    df_filtered = group[(group["Partite"] >= 5) & (group["ROI_medio"] > 0) & (group["EV_medio"] > 0)]
    st.dataframe(df_filtered, use_container_width=True)

    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button("💾 Scarica Pattern EV+ CSV", data=csv, file_name="pattern_ev_plus.csv", mime="text/csv")
