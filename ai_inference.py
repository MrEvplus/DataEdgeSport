import streamlit as st
import pandas as pd
import requests

def ask_huggingface(question, context, model="tiiuae/falcon-rw-1b"):
    """
    Invia una domanda + contesto al modello Hugging Face.
    Usa un modello gratuito, leggero e pubblico via Inference API.
    """
    HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    HEADERS = {"Authorization": f"Bearer {HF_TOKEN}"}

    st.caption(f"🔐 Token Hugging Face attivo: {HF_TOKEN[:5]}...")

    context = context[:3000]
    prompt = f"Domanda: {question}\nContesto: {context}"
    payload = {"inputs": prompt}

    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=30)
        response.raise_for_status()
        json_resp = response.json()

        if isinstance(json_resp, list) and len(json_resp) > 0:
            return json_resp[0].get("generated_text", "⚠️ Nessuna risposta generata dal modello.")
        elif isinstance(json_resp, dict) and "error" in json_resp:
            return f"⚠️ Errore dal modello AI: {json_resp['error']}"
        else:
            return "⚠️ Nessuna risposta valida dal modello AI."

    except Exception as e:
        return f"⚠️ Errore di connessione o timeout: {e}"

def run_ai_inference(df, db_selected):
    st.title("🧠 Domande AI sul tuo Database")

    st.write(f"📊 Campionato selezionato: **{db_selected}**")
    st.write(f"📁 Righe disponibili nel database: {df.shape[0]}")
    st.dataframe(df.head(10))

    question = st.text_input("✍️ Inserisci la tua domanda sul campionato:")

    if question:
        with st.spinner("🤖 Elaborazione della risposta..."):
            context = df.head(100).to_string(index=False)
            answer = ask_huggingface(question, context)
            st.success("✅ Risposta AI:")
            st.markdown(answer)
