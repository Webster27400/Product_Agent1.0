import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
import os
from datetime import datetime

# --- Konfiguracja klucza API ---
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Automatyk (Wersja Premium)")
st.markdown("Agent, który analizuje dane i zapisuje dla Ciebie raporty.")

language = st.sidebar.radio(
    "Wybierz język odpowiedzi:",
    ('Polski', 'Angielski')
)

# --- Dynamiczna Konfiguracja Agenta ---
prompt_pl = "Jesteś polskim, proaktywnym asystentem menedżera produktu. Twoim celem jest analiza danych i identyfikacja ryzyk oraz szans. Zawsze, bezwzględnie odpowiadaj TYLKO w języku polskim."
prompt_en = "You are a proactive assistant to a product manager. Your goal is to analyze data and identify risks and opportunities. Always, without exception, respond ONLY in English."

system_prompt = prompt_pl if language == 'Polski' else prompt_en

# POPRAWKA 1: Używamy znacznie potężniejszego modelu Llama 3 70B
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

# --- Tworzenie narzędzi z BARDZO PRECYZYJNYMI OPISAMI ---

def get_todays_date(fake_arg: str = "") -> str:
    """Zwraca dzisiejszą datę."""
    return datetime.now().strftime("%Y-%m-%d")

# POPRAWKA 2: Bardziej precyzyjny opis narzędzia do zapisu
def save_report(filename: str, content: str) -> str:
    """Użyj tego narzędzia do zapisania tekstu (content) w pliku o podanej nazwie (filename).
    Tego narzędzia należy użyć DOPIERO WTEDY, gdy masz już treść do zapisania, uzyskaną za pomocą innego narzędzia."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Raport został pomyślnie zapisany w pliku {filename}."

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty", description="Zwraca dzisiejszą datę.")
# POPRAWKA 2: Bardziej precyzyjny opis narzędzia do analizy
document_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="analizator_dokumentow_klientow",
        description="Użyj tego narzędzia, aby uzyskać lub podsumować informacje z dokumentów o feedbacku od klientów, zanim użyjesz innych narzędzi.",
    ),
)
file_writer_tool = FunctionTool.from_defaults(fn=save_report, name="narzedzie_do_zapisu_raportu", description="Służy do zapisywania raportów w plikach tekstowych.")

# --- Tworzenie Agenta ---
agent = ReActAgent.from_tools(tools=[date_tool, document_tool, file_writer_tool], llm=Settings.llm, verbose=True)

# --- Logika Czatu ---
if "messages" not in st.session_state:
    st.session_state.messages = []

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