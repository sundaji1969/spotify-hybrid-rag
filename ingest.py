import os
import pickle
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

# 1. Load the Spotify Architecture PDF
pdf_path = "spotify_web_app_architecture.pdf"
loader = PyPDFLoader(pdf_path)
docs = loader.load()

# 2. Split text into manageable chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# 3. Create embeddings and store in LangChain's InMemoryVectorStore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = InMemoryVectorStore(embeddings)
vector_store.add_documents(chunks)

# 4. Save to disk so you don't have to re-ingest every time
with open("vector_store.pkl", "wb") as f:
    pickle.dump(vector_store, f)

print(f"Successfully vectorized and saved {len(chunks)} chunks!")