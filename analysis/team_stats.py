"""
analysis/team_stats.py - Statistiche per singola squadra
"""

import streamlit as st
import pandas as pd
from label import label_match, extract_minutes


# -------------------------------------------------------
# 🔹 Funzione principale
# -------------------------------------------------------
def run_team_stats(df: pd.DataFrame, db_selected: str):
    st.title(f"📊 Statistiche Squadre - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Filtra squadre in base al campionato selezionato
    squadre_disponibili = sorted(set(df["Home"].dropna().unique()) | set(df["Away"].dropna().unique()))

    col1, col2 = st.columns(2)
    with col1:
        squadra_casa = st.selectbox(
            "Seleziona Squadra Casa:",
            options=[""] + squadre_disponibili,
            index=0,
            key=f"squadra_casa_{db_selected}"
        )
    with col2:
        squadra_ospite = st.selectbox(
            "Seleziona Squadra Ospite:",
            options=[""] + squadre_disponibili,
            index=0,
            key=f"squadra_ospite_{db_selected}"
        )

    # Mostra statistiche solo se selezionata almeno una squadra
    if not squadra_casa and not squadra_ospite:
        st.info("Seleziona almeno una squadra per visualizzare le statistiche.")
        return

    if squadra_casa:
        mostra_statistiche_squadra(df, squadra_casa, "Home")
    if squadra_ospite:
        mostra_statistiche_squadra(df, squadra_ospite, "Away")


# -------------------------------------------------------
# 🔹 Statistiche dettagliate per una squadra
# -------------------------------------------------------
def mostra_statistiche_squadra(df: pd.DataFrame, squadra: str, ruolo: str):
    st.subheader(f"📌 Statistiche {ruolo} - {squadra}")

    if ruolo == "Home":
        df_team = df[df["Home"] == squadra]
    else:
        df_team = df[df["Away"] == squadra]

    if df_team.empty:
        st.warning(f"Nessun dato trovato per {squadra} ({ruolo}).")
        return

    # Risultati base
    df_team["match_result"] = df_team.apply(
        lambda row: "Win" if (row["Home Goal FT"] > row["Away Goal FT"] and ruolo == "Home") or
                               (row["Away Goal FT"] > row["Home Goal FT"] and ruolo == "Away")
        else "Loss" if (row["Home Goal FT"] < row["Away Goal FT"] and ruolo == "Home") or
                      (row["Away Goal FT"] < row["Home Goal FT"] and ruolo == "Away")
        else "Draw",
        axis=1
    )

    total_matches = len(df_team)
    wins = (df_team["match_result"] == "Win").sum()
    draws = (df_team["match_result"] == "Draw").sum()
    losses = (df_team["match_result"] == "Loss").sum()

    st.markdown(f"**Partite analizzate:** {total_matches}")
    st.markdown(f"✅ Vittorie: **{wins}** ({wins/total_matches*100:.1f}%)")
    st.markdown(f"🤝 Pareggi: **{draws}** ({draws/total_matches*100:.1f}%)")
    st.markdown(f"❌ Sconfitte: **{losses}** ({losses/total_matches*100:.1f}%)")

    # Goal segnati e subiti
    if ruolo == "Home":
        goals_scored = df_team["Home Goal FT"].sum()
        goals_conceded = df_team["Away Goal FT"].sum()
    else:
        goals_scored = df_team["Away Goal FT"].sum()
        goals_conceded = df_team["Home Goal FT"].sum()

    st.markdown(f"⚽ Goal Fatti: **{goals_scored}**")
    st.markdown(f"🥅 Goal Subiti: **{goals_conceded}**")

    # Over / Under
    over15 = (df_team["goals_total"] > 1.5).mean() * 100
    over25 = (df_team["goals_total"] > 2.5).mean() * 100
    btts_pct = df_team["btts"].mean() * 100

    st.markdown(f"📈 Over 1.5: **{over15:.1f}%**")
    st.markdown(f"📈 Over 2.5: **{over25:.1f}%**")
    st.markdown(f"🤝 BTTS: **{btts_pct:.1f}%**")

    # Distribuzione goal per intervallo
    scored_percents, conceded_percents = calcola_goal_timeframes_squadra(df_team, ruolo)

    st.subheader("⚽ Distribuzione Goal per Intervallo (in %)")
    st.write(pd.DataFrame({
        "Intervallo": scored_percents.keys(),
        "Goals Scored %": scored_percents.values(),
        "Goals Conceded %": conceded_percents.values()
    }))


# -------------------------------------------------------
# 🔹 Distribuzione goal per intervallo di minuti
# -------------------------------------------------------
def calcola_goal_timeframes_squadra(df_team, ruolo):
    time_bands = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90"]

    if ruolo == "Home":
        minutes_scored = extract_minutes(df_team["minuti goal segnato home"]) if "minuti goal segnato home" in df_team.columns else []
        minutes_conceded = extract_minutes(df_team["minuti goal segnato away"]) if "minuti goal segnato away" in df_team.columns else []
    else:
        minutes_scored = extract_minutes(df_team["minuti goal segnato away"]) if "minuti goal segnato away" in df_team.columns else []
        minutes_conceded = extract_minutes(df_team["minuti goal segnato home"]) if "minuti goal segnato home" in df_team.columns else []

    scored_counts = {band: 0 for band in time_bands}
    conceded_counts = {band: 0 for band in time_bands}

    for m in minutes_scored:
        for band in time_bands:
            low, high = map(int, band.split("-"))
            if low <= m <= high:
                scored_counts[band] += 1
                break

    for m in minutes_conceded:
        for band in time_bands:
            low, high = map(int, band.split("-"))
            if low <= m <= high:
                conceded_counts[band] += 1
                break

    total_scored = sum(scored_counts.values())
    total_conceded = sum(conceded_counts.values())

    scored_percents = {band: round(scored_counts[band] / total_scored * 100, 2) if total_scored > 0 else 0 for band in time_bands}
    conceded_percents = {band: round(conceded_counts[band] / total_conceded * 100, 2) if total_conceded > 0 else 0 for band in time_bands}

    return scored_percents, conceded_percents
