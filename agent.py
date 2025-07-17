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
st.title("💡 Agent Geneza v3.0")
st.markdown("Zadaj pytanie dotyczące danych, zapytaj o dzisiejszą datę lub poproś o znalezienie czegoś w internecie.")

# Przywracamy przełącznik języka
language = st.sidebar.radio(
    "Wybierz język odpowiedzi:",
    ('Polski', 'Angielski')
)

# --- Dynamiczna Konfiguracja Agenta ---
prompt_pl = "Jesteś proaktywnym asystentem menedżera produktu. Twoim celem jest nie tylko odpowiadać na pytania, ale także identyfikować potencjalne ryzyka i szanse. Odpowiadaj zawsze po polsku, w przyjaznym, ale profesjonalnym tonie."
prompt_en = "You are a proactive assistant to a product manager. Your goal is not only to answer questions but also to identify potential risks and opportunities. Always respond in English, in a friendly yet professional tone."

system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-8b-8192", system_prompt=system_prompt)
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
    """Zwraca dzisiejszą datę w formacie ROK-MIESIĄC-DZIEŃ."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty", description="To narzędzie zwraca dzisiejszą datę.")

document_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="analizator_dokumentow_klientow",
        description=(
            "To narzędzie służy do analizy i podsumowywania dokumentów zawierających opinie i zgłoszenia od klientów. "
            "Użyj go, jeśli pytanie dotyczy feedbacku, bugów, próśb o nowe funkcje, notatek ze spotkań, plików CSV lub podsumowania informacji."
        ),
    ),
)

# --- Tworzenie Agenta ---
agent = ReActAgent.from_tools(tools=[date_tool, document_tool], llm=Settings.llm, verbose=True)

# --- Pytanie od użytkownika i odpowiedź ---
user_question = st.text_input("Twoje pytanie:")

if user_question:
    with st.spinner("Agent myśli... 🤔"):
        response = agent.chat(user_question)
        st.markdown(f"### Odpowiedź Agenta ({language}):")
        st.write(str(response))