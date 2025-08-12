import streamlit as st
import pandas as pd
from utils import label_match
from squadre import compute_team_macro_stats, is_match_played, parse_goal_times
from datetime import datetime

def run_reverse_batch(df):
    st.title("🧠 Reverse Batch - EV+, Statistiche & Goal Patterns")

    df["Data"] = pd.to_datetime(df["Data"], errors="coerce").dt.normalize()
    selected_date = st.date_input("📅 Seleziona una data da analizzare:", value=datetime.today().date())

    df_day = df[df["Data"] == pd.to_datetime(selected_date)]
    if df_day.empty:
        st.warning("⚠️ Nessuna partita trovata in questa data.")
        return

    st.info(f"📊 Partite trovate per il {selected_date.strftime('%d/%m/%Y')}: {len(df_day)}")

    risultati = []
    esclusi = []

    for _, row in df_day.iterrows():
        motivo = []
        home = row["Home"]
        away = row["Away"]
        label = label_match(row)
        quota_over = row.get("Odd Over 2.5", None)
        quota_under = row.get("Odd Under 2.5", None)
        gol_home = row.get("Home Goal FT", 0)
        gol_away = row.get("Away Goal FT", 0)
        pos_home = row.get("Posizione Classifica Home", None)
        pos_away = row.get("Posizione classifica away", None)
        data_match = row["Data"]

        if pd.isna(quota_over) or pd.isna(quota_under):
            motivo.append('Quote mancanti')
            esclusi.append({'Match': f'{home} vs {away}', 'Motivo': motivo})
            continue

        df["Label"] = df.apply(label_match, axis=1)
        df_passato = df[(df["Data"] < data_match) & (df["Label"] == label)].copy()

        if df_passato.empty:
            motivo.append('Nessun match storico con stesso label')
            esclusi.append({'Match': f'{home} vs {away}', 'Motivo': motivo})
            continue

        df_validi = df_passato.dropna(subset=["Home Goal FT", "Away Goal FT"])
        if df_validi.empty:
            motivo.append('Dati storici privi di risultati')
            esclusi.append({'Match': f'{home} vs {away}', 'Motivo': motivo})
            continue

        # EV Over 2.5
        over_hits = (df_validi["Home Goal FT"] + df_validi["Away Goal FT"] > 2.5).sum()
        total_matches = len(df_validi)
        over_pct = over_hits / total_matches if total_matches > 0 else 0
        ev_over = (quota_over * over_pct) - 1
        match_result = "✅" if (gol_home + gol_away) > 2.5 else "❌"
        profitto = (quota_over - 1) if match_result == "✅" else -1

        # Goal pattern → chi ha segnato per primo
        timeline_home = parse_goal_times(row.get("minuti goal segnato home", ""))
        timeline_away = parse_goal_times(row.get("minuti goal segnato away", ""))
        first_goal = None
        if timeline_home or timeline_away:
            primo_home = min(timeline_home) if timeline_home else 999
            primo_away = min(timeline_away) if timeline_away else 999
            if primo_home < primo_away:
                first_goal = "HOME"
            elif primo_away < primo_home:
                first_goal = "AWAY"
            else:
                first_goal = "PARI"

        recuperato = "-"
        if first_goal == "HOME" and gol_home == gol_away:
            recuperato = "Pareggio"
        elif first_goal == "HOME" and gol_home < gol_away:
            recuperato = "Ribaltata"
        elif first_goal == "AWAY" and gol_away == gol_home:
            recuperato = "Pareggio"
        elif first_goal == "AWAY" and gol_away < gol_home:
            recuperato = "Ribaltata"

        # Macro stats per casa e trasferta
        stat_home = compute_team_macro_stats(df_passato, home, "Home")
        stat_away = compute_team_macro_stats(df_passato, away, "Away")

        risultati.append({
            "Match": f"{home} vs {away}",
            "Label": label,
            "Pos Home": pos_home,
            "Pos Away": pos_away,
            "Over 2.5": quota_over,
            "Prob Over %": round(over_pct * 100, 1),
            "EV Over %": round(ev_over * 100, 1),
            "Risultato": f"{gol_home}-{gol_away}",
            "Esito": match_result,
            "Profitto": round(profitto, 2),
            "1° Gol": first_goal or "-",
            "Recupero": recuperato,
            "Win % Home": stat_home.get("Win %", 0),
            "Win % Away": stat_away.get("Win %", 0),
            "BTTS % Home": stat_home.get("BTTS %", 0),
            "BTTS % Away": stat_away.get("BTTS %", 0)
        })

    if risultati:
        df_out = pd.DataFrame(risultati)
        st.dataframe(df_out, use_container_width=True)
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Scarica CSV", data=csv, file_name="reverse_batch_output.csv", mime="text/csv")

    # Mostra SEMPRE la tabella di debug dei match esclusi
    if esclusi:
        st.subheader("📛 Match esclusi dal calcolo EV+ (debug)")
        st.dataframe(pd.DataFrame(esclusi), use_container_width=True)
    elif not risultati:
        st.warning("⚠️ Nessun match valido per il calcolo EV+.")
