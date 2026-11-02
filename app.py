import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Streamlit UI Configuration
st.set_page_config(page_title="Spotify Architecture Hybrid RAG", layout="wide")
st.title("🎵 Spotify Architecture Assistant (Hybrid RAG)")

# 1. Verify API Key first
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or "sk-" not in api_key:
    st.error("⚠️ OPENAI_API_KEY is missing or invalid in your .env file! Please add your valid OpenAI API key to the .env file and restart the app.")
    st.stop()

# Import LangChain components safely
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.vectorstores import InMemoryVectorStore
except Exception as e:
    st.error(f"Import Error: {e}")
    st.stop()

# Initialize Embeddings & LLM
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Cached Vector Store Initialization (Kept cleanly in memory - NO PICKLE)
@st.cache_resource
def get_vector_store():
    pdf_path = "spotify_web_app_architecture.pdf"
    if not os.path.exists(pdf_path):
        return None
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(chunks)
    return vector_store

try:
    with st.spinner("Loading and vectorizing architecture document..."):
        vector_store = get_vector_store()
except Exception as e:
    st.error(f"Failed to process vector store: {e}")
    vector_store = None

vector_retriever = vector_store.as_retriever(search_kwargs={"k": 3}) if vector_store else None

# User Input Box
query = st.text_input("Ask a question about the architecture:", placeholder="e.g., What databases are used in the backend?")

if query:
    if not vector_retriever:
        st.error("Vector store is missing! Ensure `spotify_web_app_architecture.pdf` is in your folder.")
    else:
        with st.spinner("Searching architecture document..."):
            try:
                vector_docs = vector_retriever.invoke(query)
                context_text = "\n\n".join([doc.page_content for doc in vector_docs])

                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a technical assistant. Answer the user question accurately using only the provided context[cite: 1]:\n\n{context}"),
                    ("human", "{input}")
                ])
                
                chain = prompt | llm | StrOutputParser()
                response = chain.invoke({"context": context_text, "input": query})

                st.subheader("Answer:")
                st.write(response)

                with st.expander("View Retrieved Source Chunks"):
                    for i, doc in enumerate(vector_docs):
                        st.markdown(f"**Chunk {i+1}:**")
                        st.text(doc.page_content)
            except Exception as e:
                st.error(f"An error occurred during query generation: {e}")