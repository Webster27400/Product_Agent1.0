import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
from llama_index.core.memory import ChatMemoryBuffer
import os
from datetime import datetime

# --- Konfiguracja klucza API ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("🧠 Agent z Pamięcią v4.1")
st.markdown("Ten agent pamięta kontekst naszej rozmowy.")

language = st.sidebar.radio(
    "Wybierz język odpowiedzi:",
    ('Polski', 'Angielski')
)

# --- Dynamiczna Konfiguracja Agenta ---
prompt_pl = "Jesteś polskim, proaktywnym asystentem. Zawsze, bezwzględnie odpowiadaj TYLKO w języku polskim, nawet jeśli pytanie lub dane są w innym języku."
prompt_en = "You are a proactive assistant. Always, without exception, respond ONLY in English, even if the user's question or the source data is in another language."

system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-70b-8192", system_prompt=system_prompt)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- Ładowanie danych (cachowane) ---
@st.cache_resource
def load_index():
    with st.spinner("Wczytuję i indeksuję dane (tylko raz)..."):
        reader = SimpleDirectoryReader(input_dir=".")
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_index()
query_engine = index.as_query_engine(llm=Settings.llm)

# --- Tworzenie narzędzi ---
def get_todays_date(fake_arg: str = "") -> str:
    """Zwraca dzisiejszą datę."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_do_sprawdzania_daty", description="To narzędzie służy do sprawdzania dzisiejszej daty.")
document_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="analizator_dokumentow_klientow",
        description="Użyj tego narzędzia do wszystkich pytań i poleceń dotyczących opinii klientów, produktów, zgłoszeń, bugów, sentymentu i notatek.",
    ),
)

# --- Tworzenie Agenta z PAMIĘCIĄ ---
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = ReActAgent.from_tools(
    tools=[date_tool, document_tool],
    llm=Settings.llm,
    memory=st.session_state.agent_memory,
    verbose=True,
    max_iterations=10
)

# --- Logika Czatu z wiadomością powitalną ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Przywracamy wiadomość powitalną
    st.session_state.messages.append({"role": "assistant", "content": "Cześć! Jestem Twoim proaktywnym asystentem. Jak mogę pomóc?"})

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