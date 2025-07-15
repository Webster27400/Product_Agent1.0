import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
import os

# --- Konfiguracja klucza API ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Geneza")
st.markdown("Zadaj pytanie dotyczące danych z plików `data.csv` i `notatki.txt`.")

# Przełącznik języka w panelu bocznym
language = st.sidebar.radio(
    "Wybierz język odpowiedzi:",
    ('Polski', 'Angielski')
)

# --- Dynamiczna Konfiguracja Agenta ---
# Definiujemy instrukcje systemowe dla obu języków
prompt_pl = "Jesteś proaktywnym asystentem menedżera produktu. Twoim celem jest nie tylko odpowiadać na pytania, ale także identyfikować potencjalne ryzyka i szanse. Odpowiadaj zawsze po polsku, w przyjaznym, ale profesjonalnym tonie."
prompt_en = "You are a proactive assistant to a product manager. Your goal is not only to answer questions but also to identify potential risks and opportunities. Always respond in English, in a friendly yet professional tone."

# Wybieramy instrukcję na podstawie przełącznika
system_prompt = prompt_pl if language == 'Polski' else prompt_en

# Ustawiamy modele z wybraną instrukcją
Settings.llm = Groq(
    model="llama3-8b-8192",
    system_prompt=system_prompt
)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# Ładowanie i indeksowanie danych (ta część jest cachowana)
@st.cache_resource
def load_index():
    with st.spinner("Wczytuję i indeksuję dane (tylko raz)..."):
        reader = SimpleDirectoryReader(input_dir=".")
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index

index = load_index()
query_engine = index.as_query_engine(llm=Settings.llm) # Przekazujemy zaktualizowany LLM

# Pytanie od użytkownika i odpowiedź
user_question = st.text_input("Twoje pytanie:")

if user_question:
    with st.spinner("Agent myśli... 🤔"):
        response = query_engine.query(user_question)
        st.markdown(f"### Odpowiedź Agenta ({language}):")
        st.write(str(response))