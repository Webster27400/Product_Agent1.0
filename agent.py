import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata, FunctionTool
import os
from datetime import datetime

# --- Konfiguracja klucza API ---
# Wczytujemy klucz z menedżera sekretów Streamlit
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
else:
    st.error("Nie znaleziono klucza GROQ_API_KEY w sekretach. Dodaj go w ustawieniach wdrożenia.")
    st.stop()

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Geneza (Wersja Finalna)")
st.markdown("Rozpocznij rozmowę z agentem.")

# ✅ Funkcjonalność: Przełącznik języka w panelu bocznym
language = st.sidebar.radio(
    "Wybierz język odpowiedzi:",
    ('Polski', 'Angielski')
)

# ✅ Funkcjonalność: Dynamiczne instrukcje dla AI
prompt_pl = "Jesteś proaktywnym asystentem menedżera produktu. Twoim celem jest nie tylko odpowiadać na pytania, ale także identyfikować potencjalne ryzyka i szanse. Odpowiadaj zawsze po polsku, w przyjaznym, ale profesjonalnym tonie."
prompt_en = "You are a proactive assistant to a product manager. Your goal is not only to answer questions but also to identify potential risks and opportunities. Always respond in English, in a friendly yet professional tone."

system_prompt = prompt_pl if language == 'Polski' else prompt_en

# ✅ Funkcjonalność: Połączenie z potężnym modelem AI (Groq) z dynamiczną instrukcją
Settings.llm = Groq(model="llama3-8b-8192", system_prompt=system_prompt)
# ✅ Funkcjonalność: Użycie lokalnego modelu do "rozumienia" tekstu
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# ✅ Funkcjonalność: Optymalizacja - Wczytywanie danych tylko raz
@st.cache_resource
def load_index():
    with st.spinner("Wczytuję i indeksuję dane (tylko raz)..."):
        reader = SimpleDirectoryReader(input_dir=".")
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_index()

# Tworzymy silnik zapytań z aktualnym modelem LLM
query_engine = index.as_query_engine(llm=Settings.llm)

# ✅ Funkcjonalność: Dwa różne narzędzia dla agenta
# Narzędzie 1: Data
def get_todays_date(fake_arg: str = "") -> str:
    """Zwraca dzisiejszą datę w formacie ROK-MIESIĄC-DZIEŃ."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty", description="To narzędzie zwraca dzisiejszą datę.")

# Narzędzie 2: Dokumenty
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

# ✅ Funkcjonalność: Zaawansowany agent typu ReAct, który decyduje, którego narzędzia użyć
agent = ReActAgent.from_tools(tools=[date_tool, document_tool], llm=Settings.llm, verbose=True)

# ✅ Funkcjonalność: Interfejs w stylu czatu z historią rozmowy
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