"""
analysis/pattern_analysis.py - Analisi pattern ricorrenti nei risultati
"""

import streamlit as st
import pandas as pd
import numpy as np


# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_pattern_analysis(df: pd.DataFrame, db_selected: str):
    st.title(f"📈 Analisi Pattern Risultati - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Selezione stagione
    stagioni_disponibili = sorted(df["Stagione"].dropna().unique()) if "Stagione" in df.columns else []
    if stagioni_disponibili:
        stagione_scelta = st.selectbox("Seleziona stagione", options=stagioni_disponibili, index=len(stagioni_disponibili) - 1)
        df = df[df["Stagione"] == stagione_scelta]

    # Analisi cosa succede dopo il primo gol
    st.subheader("⚽ Pattern dopo il Primo Gol")
    df_pattern = analizza_dopo_primo_gol(df)
    if not df_pattern.empty:
        st.dataframe(df_pattern, use_container_width=True)
    else:
        st.info("❌ Nessun dato disponibile per pattern dopo primo gol.")


# -------------------------------------------------------
# 🔹 Analizza cosa succede dopo il primo gol
# -------------------------------------------------------
def analizza_dopo_primo_gol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola le percentuali di pareggio o raddoppio dopo il primo gol.
    """
    risultati = []

    for _, row in df.iterrows():
        hg = row.get("Home Goal FT", np.nan)
        ag = row.get("Away Goal FT", np.nan)

        if np.isnan(hg) or np.isnan(ag):
            continue

        # Identifica primo gol (assumendo colonne 'minuti goal segnato home/away')
        minuti_home = estrai_minuti(row.get("minuti goal segnato home", ""))
        minuti_away = estrai_minuti(row.get("minuti goal segnato away", ""))

        if not minuti_home and not minuti_away:
            continue

        primo_minuto_home = min(minuti_home) if minuti_home else None
        primo_minuto_away = min(minuti_away) if minuti_away else None

        if primo_minuto_home is not None and (primo_minuto_away is None or primo_minuto_home < primo_minuto_away):
            primo_gol = "1-0"
        elif primo_minuto_away is not None and (primo_minuto_home is None or primo_minuto_away < primo_minuto_home):
            primo_gol = "0-1"
        else:
            continue

        # Calcola esito dopo primo gol
        if primo_gol == "1-0":
            if hg > ag:
                esito = "Raddoppio 2-0"
            elif hg == ag:
                esito = "Pareggio 1-1"
            else:
                esito = "Rimonta avversaria"
        else:  # 0-1
            if ag > hg:
                esito = "Raddoppio 0-2"
            elif ag == hg:
                esito = "Pareggio 1-1"
            else:
                esito = "Rimonta avversaria"

        risultati.append({
            "Home": row.get("Home", ""),
            "Away": row.get("Away", ""),
            "Primo Gol": primo_gol,
            "Esito Finale": esito
        })

    if not risultati:
        return pd.DataFrame()

    df_out = pd.DataFrame(risultati)
    tabella = df_out.groupby(["Primo Gol", "Esito Finale"]).size().reset_index(name="Partite")
    tabella["%"] = (tabella["Partite"] / tabella.groupby("Primo Gol")["Partite"].transform("sum")) * 100
    return tabella.sort_values(by=["Primo Gol", "%"], ascending=[True, False])


# -------------------------------------------------------
# 🔹 Funzione utilità per estrarre minuti
# -------------------------------------------------------
def estrai_minuti(val):
    if pd.isna(val) or val == "":
        return []
    if isinstance(val, list):
        return [int(v) for v in val if isinstance(v, (int, float))]
    if isinstance(val, str):
        parti = val.replace(",", ";").split(";")
        return [int(float(p)) for p in parti if p.strip().replace(".", "", 1).isdigit()]
    return []
