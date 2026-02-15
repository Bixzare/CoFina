"""
RAG Indexer - Creates/loads vector store and indexes documents on startup
Automatically caches and only re-indexes when documents change
"""

import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
import os
import hashlib
import json
from datetime import datetime
from pathlib import Path

# Cache directory for embeddings
EMBEDDING_CACHE_DIR = "./cache/embeddings"
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)

# Hash file to track document changes
DOCS_HASH_FILE = "./chroma_db/.docs_hash"

def get_docs_hash(docs_dir: str = "./src/docs") -> str:
    """Generate a hash of all PDF files to detect changes."""
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        return None
    
    hash_input = ""
    for pdf_file in pdf_files:
        hash_input += f"{pdf_file.name}:{pdf_file.stat().st_mtime}:"
    
    return hashlib.md5(hash_input.encode()).hexdigest()

def load_pdfs(docs_dir: str = "./src/docs") -> list:
    """Load all PDF documents from the docs directory."""
    docs_path = Path(docs_dir)
    
    if not docs_path.exists():
        print(f"📁 Creating docs directory: {docs_dir}")
        docs_path.mkdir(parents=True, exist_ok=True)
        return []
    
    pdf_files = list(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        print(f"📄 No PDF files found in {docs_dir}")
        return []
    
    print(f"📚 Found {len(pdf_files)} PDF files:")
    for pdf in pdf_files:
        print(f"   • {pdf.name}")
    
    all_documents = []
    for pdf_file in pdf_files:
        print(f"📖 Loading: {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"   ✅ Loaded {len(documents)} pages")
    
    return all_documents

def needs_reindexing(persist_directory: str = "./chroma_db") -> bool:
    """Check if documents need reindexing."""
    current_hash = get_docs_hash()
    
    # If no PDFs, no need to index
    if not current_hash:
        return False
    
    # If vector store doesn't exist, need indexing
    if not os.path.exists(persist_directory):
        print("🆕 No existing vector store found")
        return True
    
    # If hash file doesn't exist, need indexing
    if not os.path.exists(DOCS_HASH_FILE):
        print("📝 No document hash found")
        return True
    
    # Compare hashes
    with open(DOCS_HASH_FILE, 'r') as f:
        stored_hash = f.read().strip()
    
    if stored_hash != current_hash:
        print("📄 Documents have changed, reindexing...")
        return True
    
    print("✅ Documents unchanged, using existing index")
    return False

def save_current_hash():
    """Save current documents hash."""
    current_hash = get_docs_hash()
    if current_hash:
        os.makedirs(os.path.dirname(DOCS_HASH_FILE), exist_ok=True)
        with open(DOCS_HASH_FILE, 'w') as f:
            f.write(current_hash)

def create_vector_store(api_key: str, persist_directory: str = "./chroma_db", force_reindex: bool = False):
    """
    Create or load ChromaDB vector store with optimized settings.
    Automatically indexes documents if needed.
    """
    from RAG.chunking import chunk_documents
    
    # Check if we need to reindex
    should_reindex = force_reindex or needs_reindexing(persist_directory)
    
    # Create embeddings
    embeddings = OpenAIEmbeddings(
        model="azure/text-embedding-3-small",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        chunk_size=1000  # Process embeddings in batches
    )
    
    # Create or load vector store
    print(f"🔧 Initializing vector store at: {persist_directory}")
    
    # Handle reindexing by deleting existing collection if needed
    if should_reindex and os.path.exists(persist_directory):
        try:
            import shutil
            shutil.rmtree(persist_directory)
            print("   ✅ Removed old index")
        except:
            pass
    
    # Create new vector store
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"}
    )
    
    # Check if collection is empty
    try:
        collection = vector_store._collection
        existing_count = collection.count()
        print(f"📊 Existing chunks in vector store: {existing_count}")
    except Exception as e:
        print(f"   Note: {e}")
        existing_count = 0
    
    # Index documents if needed
    if should_reindex or existing_count == 0:
        print("\n" + "="*60)
        print("📚 RAG Indexing - Building Knowledge Base")
        print("="*60)
        
        # Load PDFs
        print("\n📖 Step 1: Loading PDF documents...")
        documents = load_pdfs()
        
        if documents:
            # Chunk documents
            print("\n✂️ Step 2: Chunking documents...")
            chunks = chunk_documents(documents, api_key)
            print(f"✅ Created {len(chunks)} semantic chunks")
            
            # Add documents in batches
            print("\n📥 Step 3: Adding chunks to vector store...")
            batch_size = 100
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                vector_store.add_documents(batch)
                print(f"   ✅ Added batch {i//batch_size + 1}/{total_batches}")
            
            # No need to call persist() - Chroma auto-persists
            
            # Save hash for future comparison
            save_current_hash()
            
            print("\n" + "="*60)
            print("✅ RAG Indexing Complete!")
            print("="*60)
            
            # Show stats
            try:
                final_count = vector_store._collection.count()
                print(f"📊 Final statistics:")
                print(f"   • Total chunks indexed: {final_count}")
                print(f"   • Source documents: {len(documents)} pages")
                print(f"   • Location: {persist_directory}")
            except Exception as e:
                print(f"   Note: {e}")
        else:
            print("\n⚠️ No PDF documents found to index.")
            print(f"   Please add PDF files to: ./src/docs/")
    else:
        print("✅ Using existing vector store (no reindexing needed)")
    
    return vector_store

def add_documents_to_index(vector_store, documents, batch_size: int = 100):
    """
    Add chunked documents to the vector store in batches for better performance.
    """
    if not documents:
        return vector_store
    
    print(f"\n📥 Adding {len(documents)} chunks to vector store...")
    
    # Add in batches
    total_batches = (len(documents) + batch_size - 1) // batch_size
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        vector_store.add_documents(batch)
        print(f"   ✅ Added batch {i//batch_size + 1}/{total_batches}")
    
    # Update hash after adding documents
    save_current_hash()
    
    return vector_store

def get_collection_stats(vector_store):
    """Get statistics about the vector store collection."""
    try:
        collection = vector_store._collection
        count = collection.count()
        
        # Get sample metadata to show sources
        try:
            samples = collection.peek(5)
            sources = set()
            if samples and 'metadatas' in samples and samples['metadatas']:
                for meta in samples['metadatas']:
                    if meta and 'filename' in meta:
                        sources.add(meta['filename'])
        except:
            sources = []
        
        return {
            "chunk_count": count,
            "collection_name": collection.name,
            "source_files": list(sources)[:5],  # First 5 unique sources
            "metadata": collection.metadata,
            "has_documents": count > 0
        }
    except Exception as e:
        return {
            "error": str(e),
            "chunk_count": 0,
            "has_documents": False
        }

def rebuild_index(api_key: str, documents=None, persist_directory: str = "./chroma_db"):
    """
    Rebuild the entire index from scratch.
    """
    import shutil
    
    print("🔄 Rebuilding entire index...")
    
    # Remove existing index
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print("   ✅ Removed old index")
    
    # Remove hash file
    if os.path.exists(DOCS_HASH_FILE):
        os.remove(DOCS_HASH_FILE)
    
    # Create new vector store
    vector_store = create_vector_store(api_key, persist_directory, force_reindex=True)
    
    return vector_store

def ensure_index(api_key: str, force_reindex: bool = False):
    """
    Ensure the vector store is properly indexed.
    Call this from app.py on startup.
    """
    print("\n🔍 Checking RAG knowledge base...")
    vector_store = create_vector_store(api_key, force_reindex=force_reindex)
    
    # Show stats
    stats = get_collection_stats(vector_store)
    if stats['chunk_count'] > 0:
        print(f"📚 RAG ready: {stats['chunk_count']} chunks available")
        if stats.get('source_files'):
            print(f"   Sources: {', '.join(stats['source_files'])}")
    else:
        print("⚠️ RAG knowledge base is empty. Add PDFs to src/docs/")
    
    return vector_store