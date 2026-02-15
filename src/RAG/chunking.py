from pathlib import Path
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import hashlib
import os

def create_semantic_chunker(api_key: str):
    """Create a semantic chunker using OpenAI embeddings via CMU gateway."""
    embeddings = OpenAIEmbeddings(
        model="azure/text-embedding-3-small",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/'
    )
    return SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

def chunk_documents(documents, api_key: str):
    """Chunk documents using semantic chunking and assign stable IDs."""
    import hashlib
    
    # Create chunks with semantic chunker
    chunker = create_semantic_chunker(api_key)
    chunks = chunker.split_documents(documents)
    
    print(f"📄 Created {len(chunks)} semantic chunks from {len(documents)} documents")
    
    # Enrich chunks with stable IDs and clean metadata
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        
        # Generate stable ID based on content and source
        content_hash = hashlib.md5(f"{source}:{chunk.page_content}:{i}".encode()).hexdigest()
        
        # Clean up metadata - keep only essential info
        chunk.metadata.update({
            "chunk_id": content_hash,
            "filename": Path(source).name if source != "unknown" else "unknown",
            "chunk_index": i,
            "char_length": len(chunk.page_content)
        })
        
        # Remove any None values
        chunk.metadata = {k: v for k, v in chunk.metadata.items() if v is not None}
        
    return chunks

def chunk_documents_optimized(documents, api_key: str, chunk_size: int = 1000):
    """
    Optimized chunking with size control for better retrieval performance
    """
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    # Use recursive splitter for more control over chunk sizes
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    
    # Add metadata
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        content_hash = hashlib.md5(f"{source}:{chunk.page_content[:100]}:{i}".encode()).hexdigest()
        
        chunk.metadata.update({
            "chunk_id": content_hash,
            "filename": Path(source).name if source != "unknown" else "unknown",
            "chunk_index": i,
            "char_length": len(chunk.page_content)
        })
    
    print(f"📄 Created {len(chunks)} optimized chunks from {len(documents)} documents")
    return chunks