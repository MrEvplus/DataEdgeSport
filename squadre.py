import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from datetime import datetime

# --------------------------------------------------------
# ENTRY POINT
# --------------------------------------------------------
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

    seasons_selected = st.multiselect(
        "Seleziona le stagioni su cui vuoi calcolare le statistiche:",
        options=seasons_available,
        default=seasons_available[:1]
    )

    if not seasons_selected:
        st.warning("Seleziona almeno una stagione.")
        st.stop()

    df_filtered = df_filtered[df_filtered["Stagione"].isin(seasons_selected)]

    teams_available = sorted(
        set(df_filtered["Home"].dropna().unique()) |
        set(df_filtered["Away"].dropna().unique())
    )

    # ✅ INIZIALIZZA SESSION STATE
    if "squadra_casa" not in st.session_state:
        st.session_state["squadra_casa"] = teams_available[0] if teams_available else ""

    if "squadra_ospite" not in st.session_state:
        st.session_state["squadra_ospite"] = ""

    # --------------------------
    # SELEZIONE SQUADRE
    # --------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.selectbox(
            "Seleziona Squadra 1",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_casa"]) if st.session_state["squadra_casa"] in teams_available else 0,
            key="squadra_casa"
        )

    with col2:
        st.selectbox(
            "Seleziona Squadra 2 (facoltativa - per confronto)",
            options=[""] + teams_available,
            index=([""] + teams_available).index(st.session_state["squadra_ospite"]) if st.session_state["squadra_ospite"] in teams_available else 0,
            key="squadra_ospite"
        )

    # Debug (opzionale)
    st.sidebar.write("✅ DEBUG selezione squadre:")
    st.sidebar.write("squadra_casa =", st.session_state.get("squadra_casa"))
    st.sidebar.write("squadra_ospite =", st.session_state.get("squadra_ospite"))

    if st.session_state["squadra_casa"]:
        st.subheader(f"✅ Statistiche Macro per {st.session_state['squadra_casa']}")
        show_team_macro_stats(df_filtered, st.session_state["squadra_casa"], venue="Home")

    if st.session_state["squadra_ospite"] and st.session_state["squadra_ospite"] != st.session_state["squadra_casa"]:
        st.subheader(f"✅ Statistiche Macro per {st.session_state['squadra_ospite']}")
        show_team_macro_stats(df_filtered, st.session_state["squadra_ospite"], venue="Away")

        st.subheader(f"⚔️ Goal Patterns - {st.session_state['squadra_casa']} vs {st.session_state['squadra_ospite']}")
        show_goal_patterns(df_filtered, st.session_state["squadra_casa"], st.session_state["squadra_ospite"], db_selected, seasons_selected[0])

# --------------------------------------------------------
# MACRO STATS
# --------------------------------------------------------
def show_team_macro_stats(df, team, venue):
    if venue == "Home":
        data = df[df["Home"] == team]
        goals_for_col = "Home Goal FT"
        goals_against_col = "Away Goal FT"
    else:
        data = df[df["Away"] == team]
        goals_for_col = "Away Goal FT"
        goals_against_col = "Home Goal FT"

    data_debug = data.copy()
    data_debug["played_flag"] = data_debug.apply(is_match_played, axis=1)

    if not data_debug.empty:
        with st.expander(f"🔎 Mostra tutte le partite filtrate di {team}"):
            st.dataframe(
                data_debug[[
                    "Home", "Away", "Data", "Orario",
                    "Home Goal FT", "Away Goal FT",
                    "minuti goal segnato home", "minuti goal segnato away",
                    "played_flag"
                ]],
                use_container_width=True
            )
    else:
        st.info(f"⚠️ Nessuna partita trovata per la squadra {team}.")
        return

    excluded = data_debug[data_debug["played_flag"] == False]
    if len(excluded) > 0:
        st.warning("⚠️ PARTITE ESCLUSE DAL CONTEGGIO:")
        st.dataframe(
            excluded[[
                "Home", "Away", "Data", "Orario",
                "Home Goal FT", "Away Goal FT",
                "minuti goal segnato home", "minuti goal segnato away"
            ]]
        )
    else:
        st.success("✅ Nessuna partita esclusa dal conteggio.")

    mask_played = data.apply(is_match_played, axis=1)
    data = data[mask_played]

    total_matches = len(data)

    if total_matches == 0:
        st.info("⚠️ Nessuna partita disputata trovata per la squadra selezionata.")
        return

    if venue == "Home":
        wins = sum(data["Home Goal FT"] > data["Away Goal FT"])
        draws = sum(data["Home Goal FT"] == data["Away Goal FT"])
        losses = sum(data["Home Goal FT"] < data["Away Goal FT"])
    else:
        wins = sum(data["Away Goal FT"] > data["Home Goal FT"])
        draws = sum(data["Away Goal FT"] == data["Home Goal FT"])
        losses = sum(data["Away Goal FT"] < data["Home Goal FT"])

    goals_for = data[goals_for_col].mean()
    goals_against = data[goals_against_col].mean()

    btts_count = sum(
        (row["Home Goal FT"] > 0) and (row["Away Goal FT"] > 0)
        for _, row in data.iterrows()
    )
    btts = (btts_count / total_matches) * 100 if total_matches > 0 else 0

    stats = {
        "Venue": venue,
        "Matches": total_matches,
        "Win %": round((wins / total_matches) * 100, 2),
        "Draw %": round((draws / total_matches) * 100, 2),
        "Loss %": round((losses / total_matches) * 100, 2),
        "Avg Goals Scored": round(goals_for, 2),
        "Avg Goals Conceded": round(goals_against, 2),
        "BTTS %": round(btts, 2)
    }

    df_stats = pd.DataFrame([stats])
    st.dataframe(df_stats.set_index("Venue"), use_container_width=True)

# --------------------------------------------------------
# LOGICA PER MATCH GIOCATO
# --------------------------------------------------------
def is_match_played(row):
    if pd.notna(row["minuti goal segnato home"]) and row["minuti goal segnato home"].strip() != "":
        return True
    if pd.notna(row["minuti goal segnato away"]) and row["minuti goal segnato away"].strip() != "":
        return True

    goals_home = row.get("Home Goal FT", None)
    goals_away = row.get("Away Goal FT", None)

    if pd.notna(goals_home) and pd.notna(goals_away):
        return True

    return False

# --------------------------------------------------------
# TIMELINE
# --------------------------------------------------------
def build_timeline(row, venue):
    try:
        h_goals = parse_goal_times(row.get("minuti goal segnato home", ""))
        a_goals = parse_goal_times(row.get("minuti goal segnato away", ""))

        timeline = []

        for m in h_goals:
            timeline.append(("H", m))
        for m in a_goals:
            timeline.append(("A", m))

        if timeline:
            timeline.sort(key=lambda x: x[1])
            return timeline

        # timeline vuota → costruisco timeline fake
        h_ft = int(row.get("Home Goal FT", 0))
        a_ft = int(row.get("Away Goal FT", 0))
        fake_timeline = []
        for _ in range(h_ft):
            fake_timeline.append(("H", 90))
        for _ in range(a_ft):
            fake_timeline.append(("A", 91))
        return fake_timeline if fake_timeline else []

    except:
        return []

# --------------------------------------------------------
# PARSE GOAL TIMES
# --------------------------------------------------------
def parse_goal_times(val):
    if pd.isna(val) or val == "":
        return []
    times = []
    for part in str(val).strip().split(";"):
        if part.strip().isdigit():
            times.append(int(part.strip()))
    return times

# --------------------------------------------------------
# TIMEFRAMES
# --------------------------------------------------------
def timeframes():
    return [
        (0, 15),
        (16, 30),
        (31, 45),
        (46, 60),
        (61, 75),
        (76, 120)
    ]
# --------------------------------------------------------
# COMPUTE GOAL PATTERNS
# --------------------------------------------------------
def compute_goal_patterns(df_team, venue, total_matches):
    if total_matches == 0:
        return {key: 0 for key in goal_pattern_keys()}, {}, {}

    def pct(count):
        return round((count / total_matches) * 100, 2) if total_matches > 0 else 0

    def pct_sub(count, base):
        return round((count / base) * 100, 2) if base > 0 else 0

    if venue == "Home":
        wins = sum(df_team["Home Goal FT"] > df_team["Away Goal FT"])
        draws = sum(df_team["Home Goal FT"] == df_team["Away Goal FT"])
        losses = sum(df_team["Home Goal FT"] < df_team["Away Goal FT"])

        zero_zero_count = sum(
            (row["Home Goal FT"] == 0) and (row["Away Goal FT"] == 0)
            for _, row in df_team.iterrows()
        )
    else:
        wins = sum(df_team["Away Goal FT"] > df_team["Home Goal FT"])
        draws = sum(df_team["Away Goal FT"] == df_team["Home Goal FT"])
        losses = sum(df_team["Away Goal FT"] < df_team["Home Goal FT"])

        zero_zero_count = sum(
            (row["Away Goal FT"] == 0) and (row["Home Goal FT"] == 0)
            for _, row in df_team.iterrows()
        )

    zero_zero_pct = round((zero_zero_count / total_matches) * 100, 2) if total_matches > 0 else 0

    tf_scored = {f"{a}-{b}": 0 for a, b in timeframes()}
    tf_conceded = {f"{a}-{b}": 0 for a, b in timeframes()}

    first_goal = 0
    last_goal = 0
    one_zero = one_one_after_one_zero = 0
    two_zero_after_one_zero = zero_one = one_one_after_zero_one = zero_two_after_zero_one = 0

    for _, row in df_team.iterrows():
        timeline = build_timeline(row, venue)
        if not timeline:
            continue

        # FIRST GOAL
        first = timeline[0][0] if len(timeline) > 0 else None

        if venue == "Home":
            if first == "H":
                first_goal += 1
        else:
            if first == "A":
                first_goal += 1

        # LAST GOAL
        last = timeline[-1][0] if len(timeline) > 0 else None

        if venue == "Home":
            if last == "H":
                last_goal += 1
        else:
            if last == "A":
                last_goal += 1

        # Calcolo TF goals
        score_home = 0
        score_away = 0

        for team_char, minute in timeline:
            if team_char == "H":
                score_home += 1
            else:
                score_away += 1

            for start, end in timeframes():
                if start < minute <= end:
                    if venue == "Home":
                        if team_char == "H":
                            tf_scored[f"{start}-{end}"] += 1
                        else:
                            tf_conceded[f"{start}-{end}"] += 1
                    else:
                        if team_char == "A":
                            tf_scored[f"{start}-{end}"] += 1
                        else:
                            tf_conceded[f"{start}-{end}"] += 1

        # -------------------------------
        # PATTERNS ANALYSIS
        # -------------------------------
        if venue == "Home":
            if first == "H":
                one_zero += 1
                score_home = 1
                score_away = 0
                for team_char, _ in timeline[1:]:
                    if team_char == "H":
                        score_home += 1
                    else:
                        score_away += 1

                    if score_home == 2 and score_away == 0:
                        two_zero_after_one_zero += 1
                        break
                    if score_home == 1 and score_away == 1:
                        one_one_after_one_zero += 1
                        break

            elif first == "A":
                zero_one += 1
                score_home = 0
                score_away = 1
                for team_char, _ in timeline[1:]:
                    if team_char == "H":
                        score_home += 1
                    else:
                        score_away += 1

                    if score_home == 1 and score_away == 1:
                        one_one_after_zero_one += 1
                        break
                    if score_home == 0 and score_away == 2:
                        zero_two_after_zero_one += 1
                        break

        elif venue == "Away":
            if first == "H":
                one_zero += 1
                score_home = 1
                score_away = 0
                for team_char, _ in timeline[1:]:
                    if team_char == "H":
                        score_home += 1
                    else:
                        score_away += 1

                    if score_home == 2 and score_away == 0:
                        two_zero_after_one_zero += 1
                        break
                    if score_home == 1 and score_away == 1:
                        one_one_after_one_zero += 1
                        break

            elif first == "A":
                zero_one += 1
                score_home = 0
                score_away = 1
                for team_char, _ in timeline[1:]:
                    if team_char == "H":
                        score_home += 1
                    else:
                        score_away += 1

                    if score_home == 1 and score_away == 1:
                        one_one_after_zero_one += 1
                        break
                    if score_home == 0 and score_away == 2:
                        zero_two_after_zero_one += 1
                        break

    two_up = sum(
        abs(row["Home Goal FT"] - row["Away Goal FT"]) >= 2
        for _, row in df_team.iterrows()
    )

    # -------------------------------
    # CALCOLO H/D/A 1st HALF
    # -------------------------------
    ht_home_win = sum(
        row["Home Goal 1T"] > row["Away Goal 1T"]
        for _, row in df_team.iterrows()
    )
    ht_draw = sum(
        row["Home Goal 1T"] == row["Away Goal 1T"]
        for _, row in df_team.iterrows()
    )
    ht_away_win = sum(
        row["Home Goal 1T"] < row["Away Goal 1T"]
        for _, row in df_team.iterrows()
    )

    # -------------------------------
    # CALCOLO H/D/A 2nd HALF
    # -------------------------------
    sh_home_win = sum(
        (row["Home Goal FT"] - row["Home Goal 1T"]) >
        (row["Away Goal FT"] - row["Away Goal 1T"])
        for _, row in df_team.iterrows()
    )
    sh_draw = sum(
        (row["Home Goal FT"] - row["Home Goal 1T"]) ==
        (row["Away Goal FT"] - row["Away Goal 1T"])
        for _, row in df_team.iterrows()
    )
    sh_away_win = sum(
        (row["Home Goal FT"] - row["Home Goal 1T"]) <
        (row["Away Goal FT"] - row["Away Goal 1T"])
        for _, row in df_team.iterrows()
    )

    tf_scored_pct = {
        k: round((v / sum(tf_scored.values())) * 100, 2) if sum(tf_scored.values()) > 0 else 0
        for k, v in tf_scored.items()
    }
    tf_conceded_pct = {
        k: round((v / sum(tf_conceded.values())) * 100, 2) if sum(tf_conceded.values()) > 0 else 0
        for k, v in tf_conceded.items()
    }

    patterns = {
        "P": total_matches,
        "Win %": pct(wins),
        "Draw %": pct(draws),
        "Loss %": pct(losses),
        "First Goal %": pct(first_goal),
        "Last Goal %": pct(last_goal),
        "1-0 %": pct(one_zero),
        "1-1 after 1-0 %": pct_sub(one_one_after_one_zero, one_zero),
        "2-0 after 1-0 %": pct_sub(two_zero_after_one_zero, one_zero),
        "0-1 %": pct(zero_one),
        "1-1 after 0-1 %": pct_sub(one_one_after_zero_one, zero_one),
        "0-2 after 0-1 %": pct_sub(zero_two_after_zero_one, zero_one),
        "2+ Goals %": pct(two_up),
        "H 1st %": pct(ht_home_win),
        "D 1st %": pct(ht_draw),
        "A 1st %": pct(ht_away_win),
        "H 2nd %": pct(sh_home_win),
        "D 2nd %": pct(sh_draw),
        "A 2nd %": pct(sh_away_win),
        "0-0 %": zero_zero_pct,
    }

    return patterns, tf_scored, tf_conceded
# --------------------------------------------------------
# BUILD HTML TABLE
# --------------------------------------------------------
def build_goal_pattern_html(patterns, team, color):
    def bar_html(value, color, width_max=80):
        width = int(width_max * (value / 100))
        return f"""
        <div style='display: flex; align-items: center;'>
            <div style='height: 10px; width: {width}px; background-color: {color}; margin-right: 5px;'></div>
            <span style='font-size: 12px;'>{value:.1f}%</span>
        </div>
        """

    rows = f"<tr><th>Statistica</th><th>{team}</th></tr>"
    for key, value in patterns.items():
        clean_key = key.replace('%', '').strip()
        cell = str(value) if key == "P" else bar_html(value, color)
        rows += f"<tr><td>{clean_key}</td><td>{cell}</td></tr>"

    html_table = f"""
    <table style='border-collapse: collapse; width: 100%; font-size: 12px;'>
        {rows}
    </table>
    """
    return html_table

# --------------------------------------------------------
# PLOT TIMEFRAME GOALS
# --------------------------------------------------------
def plot_timeframe_goals(tf_scored, tf_conceded, tf_scored_pct, tf_conceded_pct, team):
    # build dataframe
    data = []
    for tf in tf_scored.keys():
        data.append({
            "Time Frame": tf,
            "Type": "Goals Scored",
            "Percentage": tf_scored_pct[tf],
            "Count": tf_scored[tf]
        })
        data.append({
            "Time Frame": tf,
            "Type": "Goals Conceded",
            "Percentage": tf_conceded_pct[tf],
            "Count": tf_conceded[tf]
        })

    df_tf = pd.DataFrame(data)

    chart = alt.Chart(df_tf).mark_bar().encode(
        x=alt.X("Time Frame:N", title="Minute Intervals", sort=list(tf_scored.keys())),
        y=alt.Y("Percentage:Q", title="Percentage (%)"),
        color=alt.Color("Type:N",
                         scale=alt.Scale(
                             domain=["Goals Scored", "Goals Conceded"],
                             range=["green", "red"]
                       )),
       xOffset="Type:N",
       tooltip=["Type", "Time Frame", "Percentage", "Count"]
    ).properties(
       width=500,
       height=300,
       title=f"Goal Time Frame % - {team}"
    )

    # Add text labels
    text = alt.Chart(df_tf).mark_text(
        align='center',
        baseline='middle',
        dy=-5,
        color="black"
    ).encode(
        x=alt.X("Time Frame:N", sort=list(tf_scored.keys())),
        y="Percentage:Q",
        detail="Type:N",
        text=alt.Text("Count:Q", format=".0f")
    )

    return chart + text

# --------------------------------------------------------
# SHOW GOAL PATTERNS
# --------------------------------------------------------
def show_goal_patterns(df, team1, team2, country, stagione):
    # Filtra le partite per le due squadre
    df_team1_home = df[
        (df["Home"] == team1) &
        (df["country"] == country) &
        (df["Stagione"] == stagione)
    ]
    df_team2_away = df[
        (df["Away"] == team2) &
        (df["country"] == country) &
        (df["Stagione"] == stagione)
    ]

    mask_played_home = df_team1_home.apply(is_match_played, axis=1)
    df_team1_home = df_team1_home[mask_played_home]

    mask_played_away = df_team2_away.apply(is_match_played, axis=1)
    df_team2_away = df_team2_away[mask_played_away]

    total_home_matches = len(df_team1_home)
    total_away_matches = len(df_team2_away)

    # Calcola pattern Home
    patterns_home, tf_scored_home, tf_conceded_home = compute_goal_patterns(
        df_team1_home, "Home", total_home_matches
    )
    tf_scored_home_pct = {
        k: round((v / sum(tf_scored_home.values())) * 100, 2) if sum(tf_scored_home.values()) > 0 else 0
        for k, v in tf_scored_home.items()
    }
    tf_conceded_home_pct = {
        k: round((v / sum(tf_conceded_home.values())) * 100, 2) if sum(tf_conceded_home.values()) > 0 else 0
        for k, v in tf_conceded_home.items()
    }

    # Calcola pattern Away
    patterns_away, tf_scored_away, tf_conceded_away = compute_goal_patterns(
        df_team2_away, "Away", total_away_matches
    )
    tf_scored_away_pct = {
        k: round((v / sum(tf_scored_away.values())) * 100, 2) if sum(tf_scored_away.values()) > 0 else 0
        for k, v in tf_scored_away.items()
    }
    tf_conceded_away_pct = {
        k: round((v / sum(tf_conceded_away.values())) * 100, 2) if sum(tf_conceded_away.values()) > 0 else 0
        for k, v in tf_conceded_away.items()
    }

    patterns_total = compute_goal_patterns_total(
        patterns_home, patterns_away,
        total_home_matches, total_away_matches
    )

    html_home = build_goal_pattern_html(patterns_home, team1, "green")
    html_away = build_goal_pattern_html(patterns_away, team2, "red")
    html_total = build_goal_pattern_html(
        {k: patterns_total.get(k, 0) for k in goal_pattern_keys_without_tf()},
        "Totale", "blue"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"### {team1} (Home)")
        st.markdown(html_home, unsafe_allow_html=True)

    with col2:
        st.markdown(f"### {team2} (Away)")
        st.markdown(html_away, unsafe_allow_html=True)

    with col3:
        st.markdown(f"### Totale")
        st.markdown(html_total, unsafe_allow_html=True)

    # Grafico Time Frame Goals HOME
    chart_home = plot_timeframe_goals(
        tf_scored=tf_scored_home,
        tf_conceded=tf_conceded_home,
        tf_scored_pct=tf_scored_home_pct,
        tf_conceded_pct=tf_conceded_home_pct,
        team=team1
    )
    st.markdown(f"### Distribuzione Goal Time Frame - {team1} (Home)")
    st.altair_chart(chart_home, use_container_width=True)

    # Grafico Time Frame Goals AWAY
    chart_away = plot_timeframe_goals(
        tf_scored=tf_scored_away,
        tf_conceded=tf_conceded_away,
        tf_scored_pct=tf_scored_away_pct,
        tf_conceded_pct=tf_conceded_away_pct,
        team=team2
    )
    st.markdown(f"### Distribuzione Goal Time Frame - {team2} (Away)")
    st.altair_chart(chart_away, use_container_width=True)

# --------------------------------------------------------
# COMPUTE GOAL PATTERNS TOTAL
# --------------------------------------------------------
def compute_goal_patterns_total(patterns_home, patterns_away, total_home_matches, total_away_matches):
    total_matches = total_home_matches + total_away_matches
    total_patterns = {}

    for key in goal_pattern_keys():
        if key == "P":
            total_patterns["P"] = total_matches
        elif key in ["Win %", "Draw %", "Loss %"]:
            # Media delle due squadre per Win, Draw, Loss
            if key == "Win %":
                val = (patterns_home["Win %"] + patterns_away["Loss %"]) / 2
            elif key == "Draw %":
                val = (patterns_home["Draw %"] + patterns_away["Draw %"]) / 2
            elif key == "Loss %":
                val = (patterns_home["Loss %"] + patterns_away["Win %"]) / 2
            total_patterns[key] = round(val, 2)
        elif key in ["First Goal %", "Last Goal %"]:
            # non calcoliamo questi valori nel totale
            continue
        else:
            home_val = patterns_home.get(key, 0)
            away_val = patterns_away.get(key, 0)
            val = (
                (home_val * total_home_matches) + (away_val * total_away_matches)
            ) / total_matches if total_matches > 0 else 0
            total_patterns[key] = round(val, 2)

    return total_patterns

# --------------------------------------------------------
# GOAL PATTERN KEYS
# --------------------------------------------------------
def goal_pattern_keys():
    keys = [
        "P", "Win %", "Draw %", "Loss %", "First Goal %", "Last Goal %",
        "1-0 %", "1-1 after 1-0 %", "2-0 after 1-0 %",
        "0-1 %", "1-1 after 0-1 %", "0-2 after 0-1 %",
        "2+ Goals %", "H 1st %", "D 1st %", "A 1st %",
        "H 2nd %", "D 2nd %", "A 2nd %", "0-0 %"
    ]
    for start, end in timeframes():
        keys.append(f"{start}-{end} Goals %")
    return keys

def goal_pattern_keys_without_tf():
    keys = [
        "P", "Win %", "Draw %", "Loss %", "0-0 %",
        "1-0 %", "1-1 after 1-0 %", "2-0 after 1-0 %",
        "0-1 %", "1-1 after 0-1 %", "0-2 after 0-1 %",
        "2+ Goals %", "H 1st %", "D 1st %", "A 1st %",
        "H 2nd %", "D 2nd %", "A 2nd %"
    ]
    return keys

# --------------------------------------------------------
# COMPUTE TEAM MACRO STATS
# --------------------------------------------------------
def compute_team_macro_stats(df, team, venue):
    if venue == "Home":
        data = df[df["Home"] == team]
        goals_for_col = "Home Goal FT"
        goals_against_col = "Away Goal FT"
    else:
        data = df[df["Away"] == team]
        goals_for_col = "Away Goal FT"
        goals_against_col = "Home Goal FT"

    mask_played = data.apply(is_match_played, axis=1)
    data = data[mask_played]

    total_matches = len(data)
    if total_matches == 0:
        return {}

    if venue == "Home":
        wins = sum(data["Home Goal FT"] > data["Away Goal FT"])
        draws = sum(data["Home Goal FT"] == data["Away Goal FT"])
        losses = sum(data["Home Goal FT"] < data["Away Goal FT"])
    else:
        wins = sum(data["Away Goal FT"] > data["Home Goal FT"])
        draws = sum(data["Away Goal FT"] == data["Home Goal FT"])
        losses = sum(data["Away Goal FT"] < data["Home Goal FT"])

    goals_for = data[goals_for_col].mean()
    goals_against = data[goals_against_col].mean()

    btts_count = sum(
        (row["Home Goal FT"] > 0) and (row["Away Goal FT"] > 0)
        for _, row in data.iterrows()
    )
    btts = (btts_count / total_matches) * 100 if total_matches > 0 else 0

    stats = {
        "Matches Played": total_matches,
        "Win %": round((wins / total_matches) * 100, 2),
        "Draw %": round((draws / total_matches) * 100, 2),
        "Loss %": round((losses / total_matches) * 100, 2),
        "Avg Goals Scored": round(goals_for, 2),
        "Avg Goals Conceded": round(goals_against, 2),
        "BTTS %": round(btts, 2)
    }
    return stats
