import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from datetime import datetime
import pandas as pd

# --- Konfiguracja klucza API ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Geneza (Wersja Kompletna)")
st.markdown("Porozmawiaj z agentem o danych z domyślnych plików lub wgraj własne pliki do analizy.")

# --- Panel Boczny z Ustawieniami ---
with st.sidebar:
    st.header("Ustawienia")
    language = st.radio("Wybierz język odpowiedzi:", ('Polski', 'Angielski'))
    uploaded_files = st.file_uploader(
        "Wgraj nowe pliki do analizy (.csv, .txt)",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

# --- Dynamiczna Konfiguracja Agenta ---
prompt_pl = "Jesteś ekspertem, asystentem Product Managera. Analizujesz dane i tworzysz konkretne, wykonalne zadania. Zawsze odpowiadaj TYLKO w języku polskim. Używaj formatowania Markdown."
prompt_en = "You are an expert Product Manager assistant. You analyze data and create specific, actionable tasks. Always respond ONLY in English. Use Markdown formatting."
system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-70b-8192", system_prompt=system_prompt)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- Zarządzanie Danymi i Bazą Wiedzy ---
@st.cache_data(show_spinner="Przetwarzam i indeksuję pliki...")
def load_and_index_data(uploaded_file_list, default_file_paths):
    all_documents = []
    # Używamy kombinacji nazw plików jako klucza do cache'owania
    file_identifiers = [f.file_id for f in uploaded_file_list] + default_file_paths
    
    # 1. Wczytaj dane domyślne
    for file_path in default_file_paths:
        if os.path.exists(file_path):
            reader = SimpleDirectoryReader(input_files=[file_path])
            all_documents.extend(reader.load_data())

    # 2. Wczytaj dane z wgranych plików
    for uploaded_file in uploaded_file_list:
        if not os.path.exists("temp_files"):
            os.makedirs("temp_files")
        temp_file_path = f"temp_files/{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        reader = SimpleDirectoryReader(input_files=[temp_file_path])
        all_documents.extend(reader.load_data())
        st.sidebar.success(f"Plik '{uploaded_file.name}' dodany do bazy wiedzy!")

    if not all_documents:
        return None

    # 3. Zbuduj jeden, wspólny indeks
    index = VectorStoreIndex.from_documents(all_documents)
    return index

# Lista domyślnych plików
default_files = ["data.csv", "notatki.txt"]
index = load_and_index_data(uploaded_files, default_files)


# --- Tworzenie Agenta i Narzędzi (jeśli dane są dostępne) ---
if index is not None:
    query_engine = index.as_query_engine(llm=Settings.llm)

    def get_todays_date(fake_arg: str = "") -> str:
        """Zwraca dzisiejszą datę."""
        return datetime.now().strftime("%Y-%m-%d")

    date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty", description="To narzędzie służy do sprawdzania dzisiejszej daty.")
    document_tool = QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            name="analizator_dokumentow",
            description="Użyj tego narzędzia do wszystkich pytań i poleceń dotyczących opinii klientów, produktów, zgłoszeń, bugów, sentymentu i notatek.",
        ),
    )
    
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

    agent = ReActAgent.from_tools(
        tools=[date_tool, document_tool],
        llm=Settings.llm,
        memory=st.session_state.agent_memory,
        verbose=True,
        max_iterations=10
    )

    # --- Logika Czatu ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.messages.append({"role": "assistant", "content": "Cześć! Jestem gotów do analizy. Możesz rozmawiać o domyślnych danych lub wgrać własne pliki."})

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
else:
    st.error("Nie można uruchomić agenta, ponieważ nie znaleziono żadnych danych do analizy.")