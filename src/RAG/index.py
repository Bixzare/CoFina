import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def create_vector_store(api_key: str, persist_directory: str = "./chroma_db"):
    """Create or load ChromaDB vector store."""
    embeddings = OpenAIEmbeddings(
        model="azure/text-embedding-3-small",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/'
    )
    
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    return vector_store

def add_documents_to_index(vector_store, documents):
    """Add chunked documents to the vector store."""
    vector_store.add_documents(documents)
    return vector_store
