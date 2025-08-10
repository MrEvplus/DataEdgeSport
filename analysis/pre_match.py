"""
analysis/pre_match.py - Confronto pre-match tra due squadre
"""

import streamlit as st
import pandas as pd
from utils.label import label_match

# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_pre_match(df: pd.DataFrame, db_selected: str):
    st.title(f"⚔️ Confronto Pre Match - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Lista squadre
    squadre_disponibili = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))

    # -----------------------------
    # Selezione squadre persistente
    # -----------------------------
    col1, col2 = st.columns(2)
    with col1:
        squadra_casa = st.selectbox(
            "Seleziona Squadra Casa",
            options=[""] + squadre_disponibili,
            index=0 if not st.session_state.get("squadra_casa") else squadre_disponibili.index(st.session_state["squadra_casa"]) + 1,
            key=f"squadra_casa_{db_selected}"
        )
    with col2:
        squadra_ospite = st.selectbox(
            "Seleziona Squadra Ospite",
            options=[""] + squadre_disponibili,
            index=0 if not st.session_state.get("squadra_ospite") else squadre_disponibili.index(st.session_state["squadra_ospite"]) + 1,
            key=f"squadra_ospite_{db_selected}"
        )

    # Salvataggio stato
    st.session_state["squadra_casa"] = squadra_casa
    st.session_state["squadra_ospite"] = squadra_ospite

    if not squadra_casa or not squadra_ospite:
        st.info("Seleziona entrambe le squadre per procedere.")
        return

    # -----------------------------
    # Filtra partite delle squadre
    # -----------------------------
    df_casa = df[(df["Home"] == squadra_casa) | (df["Away"] == squadra_casa)]
    df_ospite = df[(df["Home"] == squadra_ospite) | (df["Away"] == squadra_ospite)]

    # -----------------------------
    # Statistiche riepilogo
    # -----------------------------
    st.subheader(f"📌 Statistiche Generali - {squadra_casa}")
    mostra_statistiche_generali(df_casa, squadra_casa)

    st.subheader(f"📌 Statistiche Generali - {squadra_ospite}")
    mostra_statistiche_generali(df_ospite, squadra_ospite)

    # -----------------------------
    # Confronto diretto (H2H)
    # -----------------------------
    st.subheader("🤝 Confronti Diretti (H2H)")
    h2h = df[
        ((df["Home"] == squadra_casa) & (df["Away"] == squadra_ospite)) |
        ((df["Home"] == squadra_ospite) & (df["Away"] == squadra_casa))
    ].sort_values(by="Data", ascending=False)

    if not h2h.empty:
        st.dataframe(h2h[["Data", "Home", "Away", "Home Goal FT", "Away Goal FT", "Odd home", "Odd Draw", "Odd Away"]], use_container_width=True)
    else:
        st.info("❌ Nessun confronto diretto trovato.")


# -------------------------------------------------------
# 🔹 Statistiche generali squadra
# -------------------------------------------------------
def mostra_statistiche_generali(df_team, squadra):
    total_matches = len(df_team)
    if total_matches == 0:
        st.warning(f"Nessun dato disponibile per {squadra}")
        return

    goals_scored = 0
    goals_conceded = 0
    wins = draws = losses = 0

    for _, row in df_team.iterrows():
        if row["Home"] == squadra:
            goals_scored += row["Home Goal FT"]
            goals_conceded += row["Away Goal FT"]
            if row["Home Goal FT"] > row["Away Goal FT"]:
                wins += 1
            elif row["Home Goal FT"] == row["Away Goal FT"]:
                draws += 1
            else:
                losses += 1
        else:
            goals_scored += row["Away Goal FT"]
            goals_conceded += row["Home Goal FT"]
            if row["Away Goal FT"] > row["Home Goal FT"]:
                wins += 1
            elif row["Away Goal FT"] == row["Home Goal FT"]:
                draws += 1
            else:
                losses += 1

    st.markdown(f"**Partite analizzate:** {total_matches}")
    st.markdown(f"✅ Vittorie: **{wins}** ({wins/total_matches*100:.1f}%)")
    st.markdown(f"🤝 Pareggi: **{draws}** ({draws/total_matches*100:.1f}%)")
    st.markdown(f"❌ Sconfitte: **{losses}** ({losses/total_matches*100:.1f}%)")
    st.markdown(f"⚽ Goal Fatti: **{goals_scored}** - 🥅 Goal Subiti: **{goals_conceded}**")

    over15 = (df_team["goals_total"] > 1.5).mean() * 100
    over25 = (df_team["goals_total"] > 2.5).mean() * 100
    btts_pct = df_team["btts"].mean() * 100

    st.markdown(f"📈 Over 1.5: **{over15:.1f}%**")
    st.markdown(f"📈 Over 2.5: **{over25:.1f}%**")
    st.markdown(f"🤝 BTTS: **{btts_pct:.1f}%**")
