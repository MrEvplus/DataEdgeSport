import requests
import json
import pandas as pd
import streamlit as st

def fetch_all_leagues():
    API_KEY = st.secrets["API_FOOTBALL_KEY"]

    url = "https://v3.football.api-sports.io/leagues"

    headers = {
        'x-rapidapi-host': "v3.football.api-sports.io",
        'x-rapidapi-key': API_KEY
    }

    response = requests.get(url, headers=headers)
    data = response.json()

    leagues = []

    for l in data.get("response", []):
        country_name = l["country"]["name"]
        league_name = l["league"]["name"]
        league_id = l["league"]["id"]

        leagues.append({
            "Country": country_name,
            "League": league_name,
            "LeagueID": league_id
        })

    df = pd.DataFrame(leagues)
    return df


if __name__ == "__main__":
    df = fetch_all_leagues()
    df.to_csv("leagues_mapping.csv", index=False)
    print(df.head())
