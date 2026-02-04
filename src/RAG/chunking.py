from pathlib import Path
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
    """Chunk documents using semantic chunking and assign stable IDs."""
    import hashlib
    
    chunker = create_semantic_chunker(api_key)
    chunks = chunker.split_documents(documents)
    
    # Enrich chunks with stable IDs and clean metadata
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        # Generate stable ID based on content and source
        # We include index to differentiate identical chunks in same file if any
        content_hash = hashlib.md5(f"{source}:{chunk.page_content}:{i}".encode()).hexdigest()
        
        chunk.metadata.update({
            "chunk_id": content_hash,
            "filename": Path(source).name if source != "unknown" else "unknown",
            "chunk_index": i
        })
        
    return chunks
