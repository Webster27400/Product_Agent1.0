import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.llms.groq import Groq
import os
import pandas as pd

# --- Konfiguracja klucza API ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Analityczny v1.0")

# --- Panel Boczny ---
with st.sidebar:
    st.header("Ustawienia")
    language = st.radio("Wybierz język odpowiedzi:", ('Polski', 'Angielski'))
    
    st.header("Dodaj Nowego Klienta")
    with st.form("new_client_form", clear_on_submit=True):
        client_name = st.text_input("Nazwa Klienta")
        client_country = st.text_input("Kraj")
        client_product = st.text_input("Produkt")
        client_status = st.selectbox("Status Projektu", ["Planowany", "W Trakcie", "Zakończony", "Pytanie"])
        client_feedback = st.text_area("Feedback")
        submitted = st.form_submit_button("Dodaj Klienta do data.csv")

# --- Zarządzanie Danymi (Nowa, Niezawodna Logika) ---
@st.cache_data
def load_data():
    try:
        return pd.read_csv("data.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=['Klient', 'Kraj', 'Produkt', 'StatusProjektu', 'Feedback'])

data_df = load_data()

if submitted:
    new_data = pd.DataFrame([{"Klient": client_name, "Kraj": client_country, "Produkt": client_product, "StatusProjektu": client_status, "Feedback": client_feedback}])
    data_df = pd.concat([data_df, new_data], ignore_index=True)
    data_df.to_csv("data.csv", index=False, encoding='utf-8')
    st.cache_data.clear()
    st.rerun()

# --- Konfiguracja i Budowa Agenta ---
prompt_pl = "Jesteś precyzyjnym asystentem analitycznym. Twoim zadaniem jest odpowiadanie na pytania wyłącznie na podstawie dostarczonego kontekstu. Jeśli odpowiedź nie znajduje się w kontekście, odpowiedz: 'Nie znalazłem tej informacji w danych'. Nie zgaduj. Zawsze odpowiadaj TYLKO po polsku."
prompt_en = "You are a precise analytical assistant. Your task is to answer questions solely based on the provided context. If the answer is not in the context, respond with: 'I did not find this information in the data'. Do not guess. Always respond ONLY in English."

system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-70b-8192", system_prompt=system_prompt)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

docs = [Document(text=row.to_json()) for index, row in data_df.iterrows()]
if not docs:
    st.warning("Baza wiedzy jest pusta. Dodaj klienta lub upewnij się, że plik data.csv istnieje.")
    st.stop()

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(llm=Settings.llm)

# --- Logika Czatu ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Cześć! Jestem gotów do analizy Twoich danych."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Twoje pytanie:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent myśli... 🤔"):
            response = query_engine.query(prompt)
            st.write(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})