import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from datetime import datetime
import pandas as pd # Dodajemy bibliotekę pandas do obsługi pliku CSV

# --- Konfiguracja klucza API ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Sekretarz v1.0")
st.markdown("Zarządzaj bazą wiedzy i rozmawiaj z agentem.")

# --- Panel Boczny z Ustawieniami i NOWYM FORMULARZEM ---
with st.sidebar:
    st.header("Ustawienia")
    language = st.radio("Wybierz język odpowiedzi:", ('Polski', 'Angielski'))
    
    st.header("Dodaj Nowego Klienta")
    # Używamy st.form, aby zgrupować pola i przycisk. clear_on_submit=True czyści formularz po wysłaniu.
    with st.form("new_client_form", clear_on_submit=True):
        client_name = st.text_input("Nazwa Klienta")
        client_country = st.text_input("Kraj")
        client_product = st.text_input("Produkt")
        client_status = st.selectbox("Status Projektu", ["Planowany", "W Trakcie", "Zakończony", "Pytanie"])
        client_feedback = st.text_area("Feedback")
        
        # Przycisk do zatwierdzenia formularza
        submitted = st.form_submit_button("Dodaj Klienta do Bazy Wiedzy")
        if submitted:
            # Wczytujemy istniejący plik CSV
            df = pd.read_csv("data.csv")
            # Tworzymy nowy wiersz z danymi
            new_data = pd.DataFrame([{
                'Klient': client_name, 'Kraj': client_country, 'Produkt': client_product,
                'StatusProjektu': client_status, 'Feedback': client_feedback
            }])
            # Łączymy stare i nowe dane
            df = pd.concat([df, new_data], ignore_index=True)
            # Zapisujemy zaktualizowaną tabelę z powrotem do pliku
            df.to_csv("data.csv", index=False, encoding='utf-8')
            st.success(f"Klient '{client_name}' został dodany!")
            # WAŻNE: Czyścimy cache, aby agent "nauczył się" nowych danych przy następnym pytaniu
            st.cache_data.clear()
            st.cache_resource.clear()

# --- Dynamiczna Konfiguracja Agenta (bez zmian) ---
prompt_pl = "Jesteś ekspertem, asystentem Product Managera. Analizujesz dane i tworzysz konkretne, wykonalne zadania. Zawsze odpowiadaj TYLKO w języku polskim. Używaj formatowania Markdown."
prompt_en = "You are an expert Product Manager assistant. You analyze data and create specific, actionable tasks. Always respond ONLY in English. Use Markdown formatting."
system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-70b-8192", system_prompt=system_prompt)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- Ładowanie danych (cachowane) ---
@st.cache_resource
def load_index():
    with st.spinner("Wczytuję i indeksuję dane..."):
        # Reader teraz wczytuje wszystkie pliki .txt i .csv w folderze
        reader = SimpleDirectoryReader(input_files=["data.csv", "notatki.txt"])
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_index()
query_engine = index.as_query_engine(llm=Settings.llm)

# --- Tworzenie narzędzi i Agenta (bez zmian) ---
def get_todays_date(fake_arg: str = "") -> str:
    """Zwraca dzisiejszą datę."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty")
document_tool = QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name="analizator_danych", description="Użyj tego narzędzia do wszystkich pytań i poleceń dotyczących opinii klientów, produktów i notatek."))
    
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = ReActAgent.from_tools(tools=[date_tool, document_tool], llm=Settings.llm, memory=st.session_state.agent_memory, verbose=True)

# --- Logika Czatu (bez zmian) ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Cześć! Jestem gotów do analizy. Możesz też dodać nowego klienta za pomocą formularza."})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Twoje pytanie:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent myśli... 🤔"):
            response = agent.chat(prompt)
            st.write(str(response))
            st.session_state.messages.append({"role": "assistant", "content": str(response)})