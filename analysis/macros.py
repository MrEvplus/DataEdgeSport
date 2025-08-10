"""
analysis/macros.py - Macro statistiche campionato + ROI mercati
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from label import label_match, extract_minutes


# -------------------------------------------------------
# 🔹 Calcolo distribuzione goal per intervalli di minuti
# -------------------------------------------------------
def calculate_goal_timeframes(sub_df, label):
    """
    Calcola la distribuzione % dei goal segnati e concessi per intervallo di minuti.
    """
    time_bands = ["0-15", "16-30", "31-45", "46-60", "61-75", "76-90"]

    minutes_home = extract_minutes(sub_df["minuti goal segnato home"]) if "minuti goal segnato home" in sub_df.columns else []
    minutes_away = extract_minutes(sub_df["minuti goal segnato away"]) if "minuti goal segnato away" in sub_df.columns else []

    # Se mancano minuti, usa goal FT come "finto minuto" (per evitare vuoti)
    if not minutes_home and "Home Goal FT" in sub_df.columns:
        for _, row in sub_df.iterrows():
            for _ in range(int(row.get("Home Goal FT", 0))):
                minutes_home.append(90)
    if not minutes_away and "Away Goal FT" in sub_df.columns:
        for _, row in sub_df.iterrows():
            for _ in range(int(row.get("Away Goal FT", 0))):
                minutes_away.append(91)

    # Determina prospettiva (Home/Away/Generale)
    if label.startswith("H_"):
        minutes_scored = minutes_home
        minutes_conceded = minutes_away
    elif label.startswith("A_"):
        minutes_scored = minutes_away
        minutes_conceded = minutes_home
    else:
        minutes_scored = minutes_home + minutes_away
        minutes_conceded = minutes_away + minutes_home

    # Conta goal per intervallo
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


# -------------------------------------------------------
# 🔹 Macro statistiche per campionato
# -------------------------------------------------------
def run_macro_stats(df, db_selected):
    st.title(f"📊 Macro Stats - {db_selected}")

    if df.empty:
        st.warning("⚠️ Nessun dato disponibile.")
        return

    # Calcolo risultato match
    if "match_result" not in df.columns:
        df["match_result"] = df.apply(
            lambda row: "Home Win" if row["Home Goal FT"] > row["Away Goal FT"]
            else "Away Win" if row["Home Goal FT"] < row["Away Goal FT"]
            else "Draw",
            axis=1
        )

    # -----------------------------
    # Riepilogo campionato
    # -----------------------------
    grouped = df.groupby(["country", "Stagione"]).agg(
        Matches=("Home", "count"),
        HomeWin_pct=("match_result", lambda x: (x == "Home Win").mean() * 100),
        Draw_pct=("match_result", lambda x: (x == "Draw").mean() * 100),
        AwayWin_pct=("match_result", lambda x: (x == "Away Win").mean() * 100),
        AvgGoals1T=("goals_1st_half", "mean"),
        AvgGoals2T=("goals_2nd_half", "mean"),
        AvgGoalsTotal=("goals_total", "mean"),
        Over05_FH_pct=("goals_1st_half", lambda x: (x > 0.5).mean() * 100),
        Over15_FH_pct=("goals_1st_half", lambda x: (x > 1.5).mean() * 100),
        Over25_FT_pct=("goals_total", lambda x: (x > 2.5).mean() * 100),
        Over35_FT_pct=("goals_total", lambda x: (x > 3.5).mean() * 100),
        BTTS_pct=("btts", lambda x: x.mean() * 100),
    ).reset_index()

    st.subheader("✅ League Stats Summary")
    st.dataframe(grouped, use_container_width=True)

    # -----------------------------
    # Statistiche per Label
    # -----------------------------
    df["Label"] = df.apply(label_match, axis=1)
    group_label = df.groupby("Label").agg(
        Matches=("Home", "count"),
        HomeWin_pct=("match_result", lambda x: (x == "Home Win").mean() * 100),
        Draw_pct=("match_result", lambda x: (x == "Draw").mean() * 100),
        AwayWin_pct=("match_result", lambda x: (x == "Away Win").mean() * 100),
        AvgGoalsTotal=("goals_total", "mean"),
        Over25_FT_pct=("goals_total", lambda x: (x > 2.5).mean() * 100),
        BTTS_pct=("btts", lambda x: x.mean() * 100),
    ).reset_index()

    st.subheader("✅ League Data by Start Price (Label)")
    st.dataframe(group_label, use_container_width=True)

    # -----------------------------
    # Distribuzione goal per Label
    # -----------------------------
    st.subheader("⚽ Distribuzione Goal Time Frame per Label")
    labels = df["Label"].dropna().unique()

    for i in range(0, len(labels), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j < len(labels):
                label = labels[i + j]
                sub_df = df[df["Label"] == label]
                scored_percents, conceded_percents = calculate_goal_timeframes(sub_df, label)

                fig = go.Figure()
                fig.add_trace(go.Bar(x=list(scored_percents.keys()), y=list(scored_percents.values()), name="Goals Scored", marker_color="green"))
                fig.add_trace(go.Bar(x=list(conceded_percents.keys()), y=list(conceded_percents.values()), name="Goals Conceded", marker_color="red"))

                fig.update_layout(title=f"Goal Time Frame % - {label}", barmode="group", height=400, yaxis=dict(title="Percentage (%)"))

                with cols[j]:
                    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------------
# 🔹 Calcolo ROI mercati
# -------------------------------------------------------
def calcola_roi_mercato_over_under(df, linea, col_ov, col_un, commissione):
    """
    Calcola ROI per Over/Under su una linea gol.
    """
    df = df.dropna(subset=["Home Goal FT", "Away Goal FT", col_ov, col_un])
    df = df[(df[col_ov] >= 1.01) & (df[col_un] >= 1.01)]

    tot = len(df)
    profit_ov = profit_un = 0
    win_ov = win_un = 0

    for _, row in df.iterrows():
        goals = row["Home Goal FT"] + row["Away Goal FT"]
        ov = row[col_ov]
        un = row[col_un]

        if goals > linea:
            win_ov += 1
            profit_ov += (ov - 1) * (1 - commissione)
            profit_un -= 1
        else:
            win_un += 1
            profit_un += (un - 1) * (1 - commissione)
            profit_ov -= 1

    if tot > 0:
        return {
            "Linea": f"Over {linea}",
            "% Over": round((win_ov / tot) * 100, 2),
            "% Under": round((win_un / tot) * 100, 2),
            "ROI Over": round((profit_ov / tot) * 100, 2),
            "ROI Under": round((profit_un / tot) * 100, 2),
            "Profit Over": round(profit_ov, 2),
            "Profit Under": round(profit_un, 2),
            "Match": tot
        }
    return {}
