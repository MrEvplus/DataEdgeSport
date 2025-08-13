import streamlit as st
import pandas as pd
from utils import label_match
from squadre import compute_team_macro_stats
from macros import run_macro_stats

# --------------------------------------------------------
# FUNZIONE PER OTTENERE LEAGUE DATA BY LABEL
# --------------------------------------------------------
def get_league_data_by_label(df, label):
    if "Label" not in df.columns:
        df = df.copy()
        df["Label"] = df.apply(label_match, axis=1)

    df["match_result"] = df.apply(
        lambda row: "Home Win" if row["Home Goal FT"] > row["Away Goal FT"]
        else "Away Win" if row["Home Goal FT"] < row["Away Goal FT"]
        else "Draw",
        axis=1
    )

    group_label = df.groupby("Label").agg(
        Matches=("Home", "count"),
        HomeWin_pct=("match_result", lambda x: (x == "Home Win").mean() * 100),
        Draw_pct=("match_result", lambda x: (x == "Draw").mean() * 100),
        AwayWin_pct=("match_result", lambda x: (x == "Away Win").mean() * 100)
    ).reset_index()

    row = group_label[group_label["Label"] == label]
    return row.iloc[0].to_dict() if not row.empty else None

# --------------------------------------------------------
# LABEL FROM ODDS
# --------------------------------------------------------
def label_from_odds(home_odd, away_odd):
    return label_match({"Odd home": home_odd, "Odd Away": away_odd})

# --------------------------------------------------------
# DETERMINA TIPO DI LABEL
# --------------------------------------------------------
def get_label_type(label):
    if label and label.startswith("H_"):
        return "Home"
    elif label and label.startswith("A_"):
        return "Away"
    return "Both"

# --------------------------------------------------------
# FORMATTING COLORE
# --------------------------------------------------------
def format_value(val, is_roi=False):
    if val is None: val = 0
    suffix = "%" if is_roi else ""
    if val > 0:
        return f"🟢 +{val:.2f}{suffix}"
    if val < 0:
        return f"🔴 {val:.2f}{suffix}"
    return f"0.00{suffix}"

# --------------------------------------------------------
# CALCOLO BACK / LAY STATS
# --------------------------------------------------------
def calculate_back_lay(filtered_df):
    profits_back = {"HOME": 0, "DRAW": 0, "AWAY": 0}
    profits_lay = {"HOME": 0, "DRAW": 0, "AWAY": 0}
    matches = len(filtered_df)

    for _, row in filtered_df.iterrows():
        h_goals = row["Home Goal FT"]
        a_goals = row["Away Goal FT"]
        result = "HOME" if h_goals > a_goals else ("AWAY" if h_goals < a_goals else "DRAW")

        for outcome in ["HOME", "DRAW", "AWAY"]:
            if outcome == "HOME":   price = row.get("Odd home", 2.00)
            elif outcome == "DRAW": price = row.get("Odd Draw", 3.20)
            else:                   price = row.get("Odd Away", 3.00)
            try: price = float(price)
            except: price = 2.00
            if price <= 1: price = 2.00

            # BACK
            profits_back[outcome] += (price - 1) if result == outcome else -1
            # LAY (liability = 1)
            stake = 1 / (price - 1)
            profits_lay[outcome] += stake if result != outcome else -1

    rois_back = {k: round((profits_back[k] / matches) * 100, 2) if matches else 0 for k in profits_back}
    rois_lay  = {k: round((profits_lay[k]  / matches) * 100, 2) if matches else 0 for k in profits_lay}
    return profits_back, rois_back, profits_lay, rois_lay, matches

# --------------------------------------------------------
# ROI OVER / UNDER 2.5 (singolo passaggio, senza duplicati)
# --------------------------------------------------------
def compute_roi_over_under_25(df_label, quota_over_default, quota_under_default, commission=0.045):
    """
    Usa quote reali se presenti e valide; altrimenti fallback alle quote inserite manualmente.
    NESSUN raddoppio del conteggio: ogni match è valutato una sola volta.
    """
    df_eval = df_label.copy()
    df_eval = df_eval.dropna(subset=["Home Goal FT", "Away Goal FT"])
    total = 0
    profit_over = 0.0
    profit_under = 0.0
    over_hits = 0
    under_hits = 0
    quote_over_list = []
    quote_under_list = []

    for _, row in df_eval.iterrows():
        goals = row["Home Goal FT"] + row["Away Goal FT"]
        quote_over = row.get("odd over 2,5", None)
        quote_under = row.get("odd under 2,5", None)

        # fallback se mancano o non valide
        try:
            quote_over = float(quote_over)
        except:
            quote_over = None
        try:
            quote_under = float(quote_under)
        except:
            quote_under = None

        if quote_over is None or quote_over < 1.01:
            quote_over = float(quota_over_default)
        if quote_under is None or quote_under < 1.01:
            quote_under = float(quota_under_default)

        total += 1
        quote_over_list.append(quote_over)
        quote_under_list.append(quote_under)

        if goals > 2.5:
            over_hits += 1
            profit_over += (quote_over - 1) * (1 - commission)
            profit_under -= 1
        else:
            under_hits += 1
            profit_under += (quote_under - 1) * (1 - commission)
            profit_over -= 1

    if total == 0:
        return None

    avg_quote_over  = round(sum(quote_over_list) / len(quote_over_list), 2)
    avg_quote_under = round(sum(quote_under_list) / len(quote_under_list), 2)
    roi_over  = round((profit_over  / total) * 100, 2)
    roi_under = round((profit_under / total) * 100, 2)
    pct_over  = round((over_hits  / total) * 100, 2)
    pct_under = round((under_hits / total) * 100, 2)

    return {
        "Linea": "2.5 Goals",
        "Quote Over": avg_quote_over,
        "Quote Under": avg_quote_under,
        "% Over": f"{pct_over}%",
        "% Under": f"{pct_under}%",
        "ROI Over": f"{roi_over}%",
        "ROI Under": f"{roi_under}%",
        "Profitto Over": round(profit_over, 2),
        "Profitto Under": round(profit_under, 2),
        "Match Analizzati": total
    }

# --------------------------------------------------------
# RUN PRE MATCH PAGE
# --------------------------------------------------------
def run_pre_match(df, db_selected):
    st.title("⚔️ Confronto Pre Match")

    if "Label" not in df.columns:
        df = df.copy()
        df["Label"] = df.apply(label_match, axis=1)

    df["Home"] = df["Home"].str.strip()
    df["Away"] = df["Away"].str.strip()

    teams_available = sorted(
        set(df[df["country"] == db_selected]["Home"].dropna().unique()) |
        set(df[df["country"] == db_selected]["Away"].dropna().unique())
    )

    # Session state
    if "squadra_casa" not in st.session_state:
        st.session_state["squadra_casa"] = teams_available[0] if teams_available else ""
    if "squadra_ospite" not in st.session_state:
        st.session_state["squadra_ospite"] = teams_available[0] if teams_available else ""

    col1, col2 = st.columns(2)
    with col1:
        squadra_casa = st.selectbox(
            "Seleziona Squadra Casa",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_casa"]) if st.session_state["squadra_casa"] in teams_available else 0,
            key=f"squadra_casa_{db_selected}"
        )
    with col2:
        squadra_ospite = st.selectbox(
            "Seleziona Squadra Ospite",
            options=teams_available,
            index=teams_available.index(st.session_state["squadra_ospite"]) if st.session_state["squadra_ospite"] in teams_available else 0,
            key=f"squadra_ospite_{db_selected}"
        )

    c1, c2, c3 = st.columns(3)
    with c1:
        odd_home = st.number_input("Quota Vincente Casa", min_value=1.01, step=0.01, value=st.session_state.get("quota_home", 2.00), key=f"odd_home_{db_selected}")
        st.markdown(f"**Probabilità Casa ({squadra_casa}):** {round(100/odd_home,2)}%")
    with c2:
        odd_draw = st.number_input("Quota Pareggio", min_value=1.01, step=0.01, value=st.session_state.get("quota_draw", 3.20), key=f"odd_draw_{db_selected}")
        st.markdown(f"**Probabilità Pareggio:** {round(100/odd_draw,2)}%")
    with c3:
        odd_away = st.number_input("Quota Vincente Ospite", min_value=1.01, step=0.01, value=st.session_state.get("quota_away", 3.80), key=f"odd_away_{db_selected}")
        st.markdown(f"**Probabilità Ospite ({squadra_ospite}):** {round(100/odd_away,2)}%")

    # Label (mostrata UNA volta)
    label = label_from_odds(odd_home, odd_away)
    label_type = get_label_type(label)
    st.markdown(f"### 🎯 Range di quota identificato (Label): `{label}`")

    if label == "Others" or label not in df["Label"].unique() or df[df["Label"] == label].empty:
        st.info("⚠️ Nessuna partita per il label selezionato. Verranno usati i dati di tutto il campionato per le sezioni dove serve.")
        label = None

    # ---------------------------
    # TABELLONE LEAGUE / SQUADRE
    # ---------------------------
    rows = []

    # League
    if label:
        filtered_league = df[df["Label"] == label]
        profits_back, rois_back, profits_lay, rois_lay, matches_league = calculate_back_lay(filtered_league)
        league_stats = get_league_data_by_label(df, label)
        row_league = {
            "LABEL": "League",
            "MATCHES": matches_league,
            "BACK WIN% HOME": round(league_stats["HomeWin_pct"], 2) if league_stats else 0,
            "BACK WIN% DRAW": round(league_stats["Draw_pct"], 2) if league_stats else 0,
            "BACK WIN% AWAY": round(league_stats["AwayWin_pct"], 2) if league_stats else 0
        }
        for outcome in ["HOME", "DRAW", "AWAY"]:
            row_league[f"BACK PTS {outcome}"] = format_value(profits_back[outcome])
            row_league[f"BACK ROI% {outcome}"] = format_value(rois_back[outcome], is_roi=True)
            row_league[f"Lay pts {outcome}"] = format_value(profits_lay[outcome])
            row_league[f"lay ROI% {outcome}"] = format_value(rois_lay[outcome], is_roi=True)
        rows.append(row_league)

    # Squadra Casa
    row_home = {"LABEL": squadra_casa}
    if label and label_type in ["Home", "Both"]:
        filtered_home = df[(df["Label"] == label) & (df["Home"] == squadra_casa)]
        if filtered_home.empty:
            filtered_home = df[df["Home"] == squadra_casa]
            st.info(f"⚠️ Nessuna partita per questo label. Calcolo eseguito su TUTTO il database per {squadra_casa}.")
        profits_back, rois_back, profits_lay, rois_lay, matches_home = calculate_back_lay(filtered_home)

        if matches_home > 0:
            wins_home  = sum(filtered_home["Home Goal FT"] > filtered_home["Away Goal FT"])
            draws_home = sum(filtered_home["Home Goal FT"] == filtered_home["Away Goal FT"])
            losses_home= sum(filtered_home["Home Goal FT"] < filtered_home["Away Goal FT"])
            pct_win_home = round((wins_home / matches_home) * 100, 2)
            pct_draw     = round((draws_home / matches_home) * 100, 2)
            pct_loss     = round((losses_home / matches_home) * 100, 2)
        else:
            pct_win_home = pct_draw = pct_loss = 0

        row_home["MATCHES"] = matches_home
        row_home["BACK WIN% HOME"] = pct_win_home
        row_home["BACK WIN% DRAW"] = pct_draw
        row_home["BACK WIN% AWAY"] = pct_loss
        for outcome in ["HOME", "DRAW", "AWAY"]:
            row_home[f"BACK PTS {outcome}"] = format_value(profits_back[outcome])
            row_home[f"BACK ROI% {outcome}"] = format_value(rois_back[outcome], is_roi=True)
            row_home[f"Lay pts {outcome}"] = format_value(profits_lay[outcome])
            row_home[f"lay ROI% {outcome}"] = format_value(rois_lay[outcome], is_roi=True)
    else:
        row_home["MATCHES"] = "N/A"
        for outcome in ["HOME", "DRAW", "AWAY"]:
            row_home[f"BACK WIN% {outcome}"] = 0
            row_home[f"BACK PTS {outcome}"] = format_value(0)
            row_home[f"BACK ROI% {outcome}"] = format_value(0, is_roi=True)
            row_home[f"Lay pts {outcome}"] = format_value(0)
            row_home[f"lay ROI% {outcome}"] = format_value(0, is_roi=True)
    rows.append(row_home)

    # Squadra Ospite
    row_away = {"LABEL": squadra_ospite}
    if label and label_type in ["Away", "Both"]:
        filtered_away = df[(df["Label"] == label) & (df["Away"] == squadra_ospite)]
        if filtered_away.empty:
            filtered_away = df[df["Away"] == squadra_ospite]
            st.info(f"⚠️ Nessuna partita per questo label. Calcolo eseguito su TUTTO il database per {squadra_ospite}.")
        profits_back, rois_back, profits_lay, rois_lay, matches_away = calculate_back_lay(filtered_away)

        if matches_away > 0:
            wins_away  = sum(filtered_away["Away Goal FT"] > filtered_away["Home Goal FT"])
            draws_away = sum(filtered_away["Away Goal FT"] == filtered_away["Home Goal FT"])
            losses_away= sum(filtered_away["Away Goal FT"] < filtered_away["Home Goal FT"])
            pct_win_away = round((wins_away / matches_away) * 100, 2)
            pct_draw     = round((draws_away / matches_away) * 100, 2)
            pct_loss     = round((losses_away / matches_away) * 100, 2)
        else:
            pct_win_away = pct_draw = pct_loss = 0

        row_away["MATCHES"] = matches_away
        row_away["BACK WIN% HOME"] = pct_loss
        row_away["BACK WIN% DRAW"] = pct_draw
        row_away["BACK WIN% AWAY"] = pct_win_away
        for outcome in ["HOME", "DRAW", "AWAY"]:
            row_away[f"BACK PTS {outcome}"] = format_value(profits_back[outcome])
            row_away[f"BACK ROI% {outcome}"] = format_value(rois_back[outcome], is_roi=True)
            row_away[f"Lay pts {outcome}"] = format_value(profits_lay[outcome])
            row_away[f"lay ROI% {outcome}"] = format_value(rois_lay[outcome], is_roi=True)
    else:
        row_away["MATCHES"] = "N/A"
        for outcome in ["HOME", "DRAW", "AWAY"]:
            row_away[f"BACK WIN% {outcome}"] = 0
            row_away[f"BACK PTS {outcome}"] = format_value(0)
            row_away[f"BACK ROI% {outcome}"] = format_value(0, is_roi=True)
            row_away[f"Lay pts {outcome}"] = format_value(0)
            row_away[f"lay ROI% {outcome}"] = format_value(0, is_roi=True)
    rows.append(row_away)

    # Tabella finale (no duplicazioni label/intestazioni)
    rows_long = []
    for row in rows:
        for outcome in ["HOME", "DRAW", "AWAY"]:
            rows_long.append({
                "LABEL": row["LABEL"],
                "SEGNO": outcome,
                "Matches": row.get("MATCHES", 0),
                "Win %": row.get(f"BACK WIN% {outcome}", 0),
                "Back Pts": row.get(f"BACK PTS {outcome}", format_value(0)),
                "Back ROI %": row.get(f"BACK ROI% {outcome}", format_value(0, is_roi=True)),
                "Lay Pts": row.get(f"Lay pts {outcome}", format_value(0)),
                "Lay ROI %": row.get(f"lay ROI% {outcome}", format_value(0, is_roi=True))
            })
    df_long = pd.DataFrame(rows_long)
    df_long.loc[df_long.duplicated(subset=["LABEL"]), "LABEL"] = ""
    st.dataframe(df_long, use_container_width=True)

    # -------------------------------------------------------
    # 📊 Confronto Statistiche Pre-Match
    # -------------------------------------------------------
    st.markdown("---")
    st.markdown("## 📊 Confronto Statistiche Pre-Match")
    stats_home = compute_team_macro_stats(df, squadra_casa, "Home")
    stats_away = compute_team_macro_stats(df, squadra_ospite, "Away")
    if not stats_home or not stats_away:
        st.info("⚠️ Una delle due squadre non ha partite disponibili per il confronto.")
        return
    st.dataframe(pd.DataFrame({squadra_casa: stats_home, squadra_ospite: stats_away}), use_container_width=True)

    # -------------------------------------------------------
    # 📈 ROI Back & Lay + EV Live
    # -------------------------------------------------------
    st.success("✅ Confronto Pre Match generato con successo!")
    st.header("📈 ROI Back & Lay + EV Live (Over e BTTS)")

    commission = 0.045
    df_label_ev = df.copy()
    df_label_ev["Label"] = df_label_ev.apply(label_match, axis=1)
    if label:
        df_label_ev = df_label_ev[df_label_ev["Label"] == label]
    df_label_ev = df_label_ev.dropna(subset=["Home Goal FT", "Away Goal FT"])

    st.markdown("### 🎯 Calcolo ROI Back & Lay su Over 1.5, 2.5, 3.5 e BTTS")

    lines = {
        "Over 1.5": ("cotao1", 1.5),
        "Over 2.5": ("cotao", 2.5),
        "Over 3.5": ("cotao3", 3.5),
        "BTTS": ("gg", None)
    }

    table_data = []
    for label_text, (col_name, goal_line) in lines.items():
        total = 0; back_profit = 0; lay_profit = 0; quote_list = []; hits = 0

        for _, row in df_label_ev.iterrows():
            goals = row["Home Goal FT"] + row["Away Goal FT"]
            gg = row.get("gg", None)
            odd = row.get(col_name, None)
            try: odd = float(odd)
            except: odd = None
            if (odd is None) or odd < 1.01:
                continue

            quote_list.append(odd); total += 1
            if label_text == "BTTS":
                if gg == 1:
                    hits += 1; back_profit += (odd - 1) * (1 - commission); lay_profit -= 1
                else:
                    lay_profit += 1 / (odd - 1); back_profit -= 1
            else:
                if goals > goal_line:
                    hits += 1; back_profit += (odd - 1) * (1 - commission); lay_profit -= 1
                else:
                    lay_profit += 1 / (odd - 1); back_profit -= 1

        if total > 0:
            avg_quote = round(sum(quote_list) / len(quote_list), 2)
            pct = round((hits / total) * 100, 2)
            roi_back = round((back_profit / total) * 100, 2)
            roi_lay = round((lay_profit / total) * 100, 2)
            table_data.append({
                "Mercato": label_text,
                "Quota Media": avg_quote,
                "Esiti %": f"{pct}%",
                "ROI Back %": f"{roi_back}%",
                "ROI Lay %": f"{roi_lay}%",
                "Match Analizzati": total
            })

    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

    # -------------------------------------------------------
    # ⚖️ ROI OVER / UNDER 2.5 con quote reali + fallback
    # -------------------------------------------------------
    st.markdown("## ⚖️ ROI Over / Under 2.5 Goals")
    apply_team_filter = st.checkbox(
        "🔍 Calcola ROI solo sui match delle squadre selezionate che rientrano nel range (label)",
        value=True
    )

    df_label = df.copy()
    df_label["LabelTemp"] = df_label.apply(label_match, axis=1)
    if label:
        df_label = df_label[df_label["LabelTemp"] == label]
    if apply_team_filter:
        df_label = df_label[
            (df_label["Home"] == st.session_state["squadra_casa"]) |
            (df_label["Away"] == st.session_state["squadra_ospite"])
        ]

    df_label = df_label.dropna(subset=["Home Goal FT", "Away Goal FT"])
    st.write("✅ Partite incluse nel calcolo ROI:", len(df_label))

    quota_inserita_over = st.number_input("📥 Quota Over 2.5 (fallback)", min_value=1.01, step=0.01, value=st.session_state.get("quota_over", 2.00), key=f"quota_over_roi_{db_selected}")
    quota_inserita_under = st.number_input("📥 Quota Under 2.5 (fallback)", min_value=1.01, step=0.01, value=st.session_state.get("quota_under", 1.80), key=f"quota_under_roi_{db_selected}")

    roi_row = compute_roi_over_under_25(df_label, quota_inserita_over, quota_inserita_under, commission)
    if roi_row:
        st.dataframe(pd.DataFrame([roi_row]), use_container_width=True)
    else:
        st.info("⚠️ Dati insufficienti per il calcolo ROI Over/Under 2.5.")

    # -------------------------------------------------------
    # 🧠 EV Manuale (mostrato una sola volta)
    # -------------------------------------------------------
    st.markdown("## 🧠 Expected Value (EV) Manuale")
    c_ev1, c_ev2, c_ev3, c_ev4 = st.columns(4)
    with c_ev1: quota_ov15 = st.number_input("Quota Live Over 1.5", min_value=1.01, step=0.01, value=2.00, key=f"quota_ev_ov15_{db_selected}")
    with c_ev2: quota_ov25 = st.number_input("Quota Live Over 2.5", min_value=1.01, step=0.01, value=2.00, key=f"quota_ev_ov25_{db_selected}")
    with c_ev3: quota_ov35 = st.number_input("Quota Live Over 3.5", min_value=1.01, step=0.01, value=2.00, key=f"quota_ev_ov35_{db_selected}")
    with c_ev4: quota_btts = st.number_input("Quota Live BTTS",      min_value=1.01, step=0.01, value=2.00, key=f"quota_ev_btts_{db_selected}")

    ev_table = pd.DataFrame([
        {"Mercato": "Over 1.5", "Quota Inserita": quota_ov15, "Probabilità Storica": f"{0.0:.1f}%", "EV": 0.0, "Note": "—"},
        {"Mercato": "Over 2.5", "Quota Inserita": quota_ov25, "Probabilità Storica": f"{0.0:.1f}%", "EV": 0.0, "Note": "—"},
        {"Mercato": "Over 3.5", "Quota Inserita": quota_ov35, "Probabilità Storica": f"{0.0:.1f}%", "EV": 0.0, "Note": "—"},
        {"Mercato": "BTTS",     "Quota Inserita": quota_btts, "Probabilità Storica": f"{0.0:.1f}%", "EV": 0.0, "Note": "—"},
    ])
    st.dataframe(ev_table, use_container_width=True)
