import streamlit as st
from llama_index.core import VectorStoreIndex, Settings, Document
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
st.title("💡 Agent Sekretarz (Wersja Stabilna)")
st.markdown("Zarządzaj bazą wiedzy i rozmawiaj z agentem.")

# --- Panel Boczny z Ustawieniami i Formularzem ---
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
        submitted = st.form_submit_button("Dodaj Klienta")

# --- Zarządzanie Danymi w Pamięci Aplikacji (Nowa, Stabilna Logika) ---
if 'data_df' not in st.session_state:
    try:
        # Próbujemy wczytać domyślny plik tylko raz, przy pierwszym uruchomieniu
        st.session_state.data_df = pd.read_csv("data.csv")
    except FileNotFoundError:
        st.session_state.data_df = pd.DataFrame(columns=['Klient', 'Kraj', 'Produkt', 'StatusProjektu', 'Feedback'])

# Jeśli formularz został wysłany, dodajemy nowe dane do naszej bazy w pamięci
if submitted:
    new_data = pd.DataFrame([{"Klient": client_name, "Kraj": client_country, "Produkt": client_product, "StatusProjektu": client_status, "Feedback": client_feedback}])
    st.session_state.data_df = pd.concat([st.session_state.data_df, new_data], ignore_index=True)
    st.sidebar.success(f"Klient '{client_name}' dodany do bieżącej sesji!")
    # Opcjonalnie: zapisujemy zmiany z powrotem do pliku na dysku
    st.session_state.data_df.to_csv("data.csv", index=False, encoding='utf-8')

# --- Konfiguracja i Budowa Agenta (zawsze na aktualnych danych) ---
prompt_pl = "Jesteś ekspertem, asystentem Product Managera. Analizujesz dane i tworzysz konkretne, wykonalne zadania. Zawsze odpowiadaj TYLKO w języku polskim. Używaj formatowania Markdown."
prompt_en = "You are an expert Product Manager assistant. You analyze data and create specific, actionable tasks. Always respond ONLY in English. Use Markdown formatting."
system_prompt = prompt_pl if language == 'Polski' else prompt_en

Settings.llm = Groq(model="llama3-70b-8192", system_prompt=system_prompt)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# Tworzymy bazę wiedzy z danych w pamięci sesji
docs = [Document(text=row.to_json()) for index, row in st.session_state.data_df.iterrows()]
if not docs:
    st.warning("Baza wiedzy jest pusta. Dodaj klienta za pomocą formularza lub upewnij się, że plik data.csv istnieje.")
    st.stop()

index = VectorStoreIndex.from_documents(docs)
query_engine = index.as_query_engine(llm=Settings.llm)

# Tworzenie narzędzi
def get_todays_date(fake_arg: str = "") -> str:
    """Zwraca dzisiejszą datę."""
    return datetime.now().strftime("%Y-%m-%d")

date_tool = FunctionTool.from_defaults(fn=get_todays_date, name="narzedzie_daty")
document_tool = QueryEngineTool(query_engine=query_engine, metadata=ToolMetadata(name="analizator_danych", description="Użyj tego narzędzia do wszystkich pytań i poleceń dotyczących opinii klientów, produktów i notatek."))
    
if "agent_memory" not in st.session_state:
    st.session_state.agent_memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

agent = ReActAgent.from_tools(tools=[date_tool, document_tool], llm=Settings.llm, memory=st.session_state.agent_memory, verbose=True)

# --- Logika Czatu ---
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({"role": "assistant", "content": "Cześć! Jestem gotów do analizy."})

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