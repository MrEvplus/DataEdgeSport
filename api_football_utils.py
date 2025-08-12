# api_football_utils.py

import streamlit as st
import requests
import pandas as pd
from datetime import date

def get_fixtures_today_for_countries(country_list):
    API_KEY = st.secrets["API_FOOTBALL_KEY"]

    today = date.today().strftime("%Y-%m-%d")
    url = f"https://v3.football.api-sports.io/fixtures?date={today}"

    headers = {
        "x-rapidapi-host": "v3.football.api-sports.io",
        "x-rapidapi-key": API_KEY
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    matches = []

    for f in data.get("response", []):
        league_country = f["league"]["country"]
        league_name = f["league"]["name"]

        if league_country not in country_list:
            continue

        matches.append({
            "Country": league_country,
            "League": league_name,
            "Home": f["teams"]["home"]["name"],
            "Away": f["teams"]["away"]["name"],
            "DateTime": f["fixture"]["date"],
            "Status": f["fixture"]["status"]["short"]
        })

    if matches:
        df = pd.DataFrame(matches)
    else:
        df = pd.DataFrame(columns=["Country", "League", "Home", "Away", "DateTime", "Status"])

    return df
