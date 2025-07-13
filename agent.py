import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
import os

# --- Konfiguracja Agenta ---
# BEZPIECZNY SPOSÓB: Wczytujemy klucz z menedżera sekretów Streamlit
# Ten kod będzie działał zarówno lokalnie, jak i po wdrożeniu w chmurze
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# Ustawiamy modele
Settings.llm = Groq(
    model="llama3-8b-8192",
    system_prompt="Jesteś proaktywnym asystentem menedżera produktu. Odpowiadaj zawsze po polsku."
)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5"

# --- Interfejs Aplikacji Streamlit ---
st.title("💡 Agent Geneza") # Zmieniamy nazwę, tak jak chciałeś :)
st.markdown("Zadaj pytanie dotyczące danych z plików `data.csv` i `notatki.txt`.")

@st.cache_resource
def load_index():
    with st.spinner("Wczytuję i indeksuję dane (tylko raz)..."):
        reader = SimpleDirectoryReader(input_dir=".")
        documents = reader.load_data()
        index = VectorStoreIndex.from_documents(documents)
        return index.as_query_engine()

query_engine = load_index()

user_question = st.text_input("Twoje pytanie:")

if user_question:
    with st.spinner("Agent myśli... 🤔"):
        response = query_engine.query(user_question)
        st.markdown("### Odpowiedź Agenta:")
        st.write(str(response))