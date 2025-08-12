from macros import calculate_goal_timeframes
import streamlit as st
import pandas as pd
import plotly.express as px

def run_team_stats(df, db_selected):
    st.header("📊 Statistiche per Squadre")

    df["country"] = df["country"].fillna("").astype(str).str.strip().str.upper()
    db_selected = db_selected.strip().upper()

    if db_selected not in df["country"].unique():
        st.warning(f"⚠️ Il campionato selezionato '{db_selected}' non è presente nel database.")
        st.stop()

    df_filtered = df[df["country"] == db_selected]

    seasons_available = sorted(df_filtered["Stagione"].dropna().unique().tolist(), reverse=True)

    if not seasons_available:
        st.warning(f"⚠️ Nessuna stagione disponibile nel database per il campionato {db_selected}.")
        st.stop()

    st.write(f"Stagioni disponibili nel database: {seasons_available}")

    preset_options = {
        "Ultimi 3 anni": 3,
        "Ultimi 5 anni": 5,
        "Ultimi 10 anni": 10,
        "Tutti gli anni": len(seasons_available),
        "Scelta manuale": None,
    }

    preset_choice = st.selectbox("📅 Seleziona un'opzione per le stagioni:", list(preset_options.keys()))

    if preset_options[preset_choice] is not None:
        num_years = preset_options[preset_choice]
        seasons_selected = seasons_available[:num_years]
        st.success(f"Hai selezionato: {seasons_selected}")
    else:
        seasons_selected = st.multiselect(
            "Seleziona le stagioni su cui vuoi calcolare le statistiche:",
            options=seasons_available,
            default=seasons_available[:1]
        )

    if not seasons_selected:
        st.warning("⚠️ Seleziona almeno una stagione.")
        st.stop()

    df_filtered = df_filtered[df_filtered["Stagione"].isin(seasons_selected)]

    teams_available = sorted(
        set(df_filtered["Home"].dropna().unique()) |
        set(df_filtered["Away"].dropna().unique())
    )

    if "squadra_casa" not in st.session_state:
        st.session_state["squadra_casa"] = teams_available[0] if teams_available else ""

    if "squadra_ospite" not in st.session_state:
        st.session_state["squadra_ospite"] = ""

    col1, col2 = st.columns(2)

    with col1:
        squadra_casa = st.selectbox(
            "Seleziona Squadra 1",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_casa"]) if st.session_state["squadra_casa"] in teams_available else 0,
            key="squadra_casa"
        )

    with col2:
        squadra_ospite = st.selectbox(
            "Seleziona Squadra 2 (facoltativa - per confronto)",
            options=[""] + teams_available,
            index=( [""] + teams_available ).index(st.session_state["squadra_ospite"]) if st.session_state["squadra_ospite"] in teams_available else 0,
            key="squadra_ospite"
        )

    if squadra_casa:
        st.subheader(f"✅ Statistiche Macro per {squadra_casa}")
        show_team_macro_stats(df_filtered, squadra_casa, venue="Home")

    if squadra_ospite and squadra_ospite != squadra_casa:
        st.subheader(f"✅ Statistiche Macro per {squadra_ospite}")
        show_team_macro_stats(df_filtered, squadra_ospite, venue="Away")

        st.subheader(f"⚔️ Goal Patterns - {squadra_casa} vs {squadra_ospite}")
        show_goal_patterns(df_filtered, squadra_casa, squadra_ospite, db_selected, seasons_selected[0])
