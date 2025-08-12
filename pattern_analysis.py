# pattern_analysis.py
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt


def _coerce_num(s: pd.Series) -> pd.Series:
    """Converte una Serie in numerico gestendo virgole/percentuali."""
    if s is None:
        return pd.Series(dtype="float")
    out = (
        s.astype(str)
         .str.replace("%", "", regex=False)
         .str.replace(",", ".", regex=False)
    )
    return pd.to_numeric(out, errors="coerce")


def _win_flag(esito: pd.Series) -> pd.Series:
    """Converte la colonna 'Esito' in 1/0 per il calcolo % win Over."""
    if esito is None:
        return pd.Series(dtype="float")
    s = esito.astype(str).str.strip().str.lower()
    return s.isin({"✅", "win", "v", "true", "1", "si", "yes", "y", "over", "won"}).astype("float")


def _pick_profit_col(df: pd.DataFrame) -> str:
    """Trova la colonna profitto migliore fra alcune varianti comuni."""
    candidates = ["Profitto", "Profit", "ROI", "PnL", "P&L"]
    for c in candidates:
        if c in df.columns:
            return c
    # se nulla, crea una colonna fittizia a zero
    df["Profitto"] = 0.0
    return "Profitto"


def run_pattern_analysis(uploaded_df: pd.DataFrame | None = None) -> None:
    st.title("🔍 Pattern Ricorrenti EV+")

    # 1) Input
    if uploaded_df is None:
        uploaded_file = st.file_uploader("📤 Carica il file CSV generato dal Reverse Batch:", type=["csv"])
        if uploaded_file is None:
            st.info("Carica un file per iniziare.")
            return
        df = pd.read_csv(uploaded_file)
    else:
        df = uploaded_df.copy()

    if df.empty:
        st.warning("Il file caricato non contiene righe.")
        return

    st.success(f"✅ {len(df):,} righe caricate per l'analisi.")

    # 2) Normalizzazioni minime/robustezza
    # Label sempre presente
    if "Label" not in df.columns:
        df["Label"] = "Unknown"
    df["Label"] = df["Label"].fillna("Unknown").astype(str)

    # EV Over %
    ev_col = "EV Over %"
    if ev_col not in df.columns:
        st.error("Colonna richiesta mancante: 'EV Over %'.")
        st.stop()
    df[ev_col] = _coerce_num(df[ev_col])

    # Profitto / Profit
    prof_col = _pick_profit_col(df)
    df[prof_col] = _coerce_num(df[prof_col])

    # Esito → flag win (1/0) per % di vittoria Over
    win_col = "_win_over_flag"
    if "Esito" in df.columns:
        df[win_col] = _win_flag(df["Esito"])
    else:
        # Se manca 'Esito', lascio NaN (verrà ignorato nella media)
        df[win_col] = np.nan

    # 3) Raggruppamento per Label
    # Partite: uso size su 'Label' così non dipendo da una colonna 'Match'
    group = df.groupby("Label").agg(
        Partite=("Label", "size"),
        EV_medio=(ev_col, "mean"),
        ROI_totale=(prof_col, "sum"),
        ROI_medio=(prof_col, "mean"),
        **{"Win_Over_%": (win_col, lambda x: float(np.nanmean(x) * 100) if x.notna().any() else np.nan)}
    ).reset_index()

    # Pulizia/ordinamento
    group = group.round(2)
    group = group.sort_values(by=["EV_medio", "ROI_medio"], ascending=[False, False])

    st.markdown("### 🧠 Pattern per Label")
    st.dataframe(group, use_container_width=True)

    # 4) Filtro pattern “forti”
    st.markdown("### 🟢 Pattern EV+ con ROI medio positivo e almeno 5 partite")
    min_partite = st.slider("Minimo partite per label", 1, 50, 5)
    df_filtered = group[
        (group["Partite"] >= min_partite) &
        (group["ROI_medio"] > 0) &
        (group["EV_medio"] > 0)
    ].copy()
    st.dataframe(df_filtered, use_container_width=True)

    # 5) Grafico rapido (EV medio e ROI medio per label)
    if not group.empty:
        top_n = st.slider("Mostra top N label per EV medio", 5, min(50, len(group)), min(10, len(group)))
        to_plot = group.nlargest(top_n, "EV_medio")
        chart = alt.Chart(to_plot).transform_fold(
            ["EV_medio", "ROI_medio"],
            as_=["Metrica", "Valore"]
        ).mark_bar().encode(
            x=alt.X("Label:N", sort="-y", title="Label"),
            y=alt.Y("Valore:Q", title="Valore"),
            color=alt.Color("Metrica:N"),
            column=alt.Column("Metrica:N", header=alt.Header(title="")),
            tooltip=["Label", "Partite", "EV_medio", "ROI_medio", "ROI_totale", "Win_Over_%"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    # 6) Export CSV
    csv = df_filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        "💾 Scarica Pattern EV+ CSV",
        data=csv,
        file_name="pattern_ev_plus.csv",
        mime="text/csv"
    )
