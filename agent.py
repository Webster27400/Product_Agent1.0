import streamlit as st
import os # Upewnij się, że ten import jest na górze

# === BLOK DIAGNOSTYCZNY ===
st.write("--- DIAGNOSTYKA ---")
st.write(f"Aktualny folder roboczy skryptu: {os.getcwd()}")
try:
    st.write(f"Pliki w folderze roboczym: {os.listdir('.')}")
except Exception as e:
    st.write(f"Błąd przy listowaniu plików: {e}")
st.write("--- KONIEC DIAGNOSTYKI ---")
# ==========================


# Tutaj zaczyna się reszta Twojego kodu...
# from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
# ...itd.

import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
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
st.title("💡 Agent Interaktywny v11.0")
st.markdown("Agent, który uczy się na bieżąco i pomaga w nawigacji.")

# --- Panel Boczny ---
with st.sidebar:
    st.header("Ustawienia")
    # Usunęliśmy przełącznik języka, aby uprościć tę wersję i skupić się na zarządzaniu danymi
    
    st.header("Dodaj Nowego Klienta")
    with st.form("new_client_form", clear_on_submit=True): # clear_on_submit=True automatycznie czyści formularz
        client_name = st.text_input("Nazwa Klienta")
        client_country = st.text_input("Kraj")
        client_product = st.text_input("Produkt")
        client_status = st.selectbox("Status Projektu", ["Planowany", "W Trakcie", "Zakończony"])
        client_feedback = st.text_area("Feedback")
        submitted = st.form_submit_button("Dodaj Klienta")

# --- Konfiguracja Modeli ---
Settings.llm = Groq(model="llama3-70b-8192", system_prompt="Jesteś precyzyjnym asystentem analitycznym. Odpowiadaj tylko na podstawie dostarczonych dokumentów. Zawsze odpowiadaj po polsku.")
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- Zarządzanie Danymi w Pamięci Aplikacji ---
# Inicjalizujemy bazę danych w pamięci sesji, jeśli jeszcze nie istnieje
if 'data_df' not in st.session_state:
    try:
        # Próbujemy wczytać domyślny plik
        st.session_state.data_df = pd.read_csv("data.csv")
    except FileNotFoundError:
        # Jeśli go nie ma, tworzymy pustą ramkę danych
        st.session_state.data_df = pd.DataFrame(columns=['Klient', 'Kraj', 'Produkt', 'StatusProjektu', 'Feedback'])

# Jeśli formularz został wysłany, dodajemy nowe dane do naszej bazy w pamięci
if submitted:
    new_data = pd.DataFrame([{
        'Klient': client_name, 'Kraj': client_country, 'Produkt': client_product,
        'StatusProjektu': client_status, 'Feedback': client_feedback
    }])
    st.session_state.data_df = pd.concat([st.session_state.data_df, new_data], ignore_index=True)
    st.sidebar.success(f"Klient '{client_name}' został dodany do sesji!")
    # Zapisujemy zmiany z powrotem do pliku na dysku
    st.session_state.data_df.to_csv("data.csv", index=False, encoding='utf-8')


# --- Tworzenie Bazy Wiedzy Agenta (zawsze z aktualnych danych) ---
# Konwertujemy naszą tabelę z danymi na listę dokumentów, które agent zrozumie
docs = [Document(text=row.to_json()) for index, row in st.session_state.data_df.iterrows()]
if docs:
    index = VectorStoreIndex.from_documents(docs)
    query_engine = index.as_query_engine(llm=Settings.llm)
else:
    query_engine = None

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
        if query_engine is not None:
            with st.spinner("Analityk myśli... 🤔"):
                response = query_engine.query(prompt)
                st.write(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})
                
                # NOWA FUNKCJONALNOŚĆ: Sugestie klientów
                if "brak informacji" in str(response).lower() or "nie znalazłem" in str(response).lower():
                    st.write("Nie jestem pewien, o którego klienta chodzi. Czy miałeś na myśli któregoś z poniższych?")
                    
                    # Tworzymy kolumny na przyciski
                    client_list = st.session_state.data_df['Klient'].unique()
                    cols = st.columns(len(client_list))
                    for i, client_name in enumerate(client_list):
                        with cols[i]:
                            # Po kliknięciu przycisku, jego nazwa staje się nowym promptem
                            if st.button(client_name):
                                st.session_state.new_prompt = f"Opisz zgłoszenia dla klienta {client_name}"
                                st.rerun() # Odświeżamy aplikację, aby przetworzyć nowy prompt

else:
    st.error("Brak danych do analizy. Dodaj nowego klienta lub upewnij się, że plik data.csv istnieje.")

# Sprawdzamy, czy został wygenerowany nowy prompt z przycisku
if "new_prompt" in st.session_state and st.session_state.new_prompt:
    prompt_from_button = st.session_state.new_prompt
    st.session_state.new_prompt = None  # Czyścimy, aby nie odpalać w pętli
    
    # Wyświetlamy go i przetwarzamy tak jak zwykły prompt
    st.session_state.messages.append({"role": "user", "content": prompt_from_button})
    with st.chat_message("user"):
        st.markdown(prompt_from_button)
    
    with st.chat_message("assistant"):
        if query_engine is not None:
            with st.spinner("Analityk myśli... 🤔"):
                response = query_engine.query(prompt_from_button)
                st.write(str(response))
                st.session_state.messages.append({"role": "assistant", "content": str(response)})