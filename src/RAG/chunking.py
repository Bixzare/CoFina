from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

def create_semantic_chunker(api_key: str):
    """Create a semantic chunker using OpenAI embeddings via CMU gateway."""
    embeddings = OpenAIEmbeddings(
        model="azure/text-embedding-3-small",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/'
    )
    return SemanticChunker(embeddings)

def chunk_documents(documents, api_key: str):
    """Chunk documents using semantic chunking."""
    chunker = create_semantic_chunker(api_key)
    return chunker.split_documents(documents)
