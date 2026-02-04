import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from RAG.chunking import chunk_documents
from RAG.index import create_vector_store, add_documents_to_index
from RAG.retriever import create_retriever, create_rag_chain, query

# Load environment variables from parent directory
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

def load_pdfs(docs_dir: str = "./docs"):
    """Load all PDF documents from the docs directory."""
    docs_path = Path(docs_dir)
    all_documents = []
    
    for pdf_file in docs_path.glob("*.pdf"):
        loader = PyPDFLoader(str(pdf_file))
        documents = loader.load()
        all_documents.extend(documents)
        print(f"Loaded: {pdf_file.name}")
    
    return all_documents

def get_docs_hash(docs_dir: str = "./docs"):
    """Generate a hash of all PDF files to detect changes."""
    import hashlib
    docs_path = Path(docs_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    
    if not pdf_files:
        return None
    
    # Create hash from filenames and modification times
    hash_input = ""
    for pdf_file in pdf_files:
        hash_input += f"{pdf_file.name}:{pdf_file.stat().st_mtime}:"
    
    return hashlib.md5(hash_input.encode()).hexdigest()

def setup_rag(force_reindex: bool = False):
    """Initialize the RAG system."""
    persist_dir = Path("./chroma_db")
    hash_file = persist_dir / ".docs_hash"
    
    # Check if we need to reindex
    current_hash = get_docs_hash()
    if not current_hash:
        print("No PDFs found in ./docs directory")
        return None
    
    needs_reindex = force_reindex
    
    if not needs_reindex and persist_dir.exists() and hash_file.exists():
        # Check if documents have changed
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash == current_hash:
            print("Loading existing vector store (documents unchanged)...")
            vector_store = create_vector_store(API_KEY)
            
            # Verify the store has documents
            try:
                collection = vector_store._collection
                if collection.count() > 0:
                    print(f"Loaded {collection.count()} existing chunks from vector store")
                    print("Setting up RAG chain...")
                    retriever = create_retriever(vector_store)
                    chain = create_rag_chain(retriever, API_KEY)
                    return chain
                else:
                    print("Vector store is empty, reindexing...")
                    needs_reindex = True
            except Exception as e:
                print(f"Error loading vector store: {e}")
                needs_reindex = True
        else:
            print("Documents have changed, reindexing...")
            needs_reindex = True
    else:
        if not persist_dir.exists():
            print("No existing vector store found, creating new one...")
        needs_reindex = True
    
    # Reindex if needed
    if needs_reindex:
        print("Loading PDFs...")
        documents = load_pdfs()
        
        if not documents:
            print("No PDFs found in ./docs directory")
            return None
        
        print(f"Chunking {len(documents)} documents...")
        chunks = chunk_documents(documents, API_KEY)
        print(f"Created {len(chunks)} chunks")
        
        print("Creating vector store...")
        vector_store = create_vector_store(API_KEY)
        
        # Clear existing data if any
        try:
            vector_store.delete_collection()
            vector_store = create_vector_store(API_KEY)
        except:
            pass
        
        add_documents_to_index(vector_store, chunks)
        
        # Save the hash
        persist_dir.mkdir(exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(current_hash)
        
        print("Setting up RAG chain...")
        retriever = create_retriever(vector_store)
        chain = create_rag_chain(retriever, API_KEY)
        
        return chain

def main():
    """Main CLI loop."""
    import argparse
    from agent.core import CoFinaAgent
    from utils.logger import AgentLogger
    
    parser = argparse.ArgumentParser(description="CoFina RAG Agent")
    parser.add_argument('--reindex', action='store_true', 
                       help='Force reindexing of all documents')
    args = parser.parse_args()
    
    if not API_KEY:
        print("Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Initialize logger
    logger = AgentLogger()
    print(f"Logging traces to {logger.log_dir}")
    
    # Setup RAG (Index check)
    # We still use setup_rag to handle the indexing part, but we don't need the chain it returns
    # The Agent will create its own components
    print("Checking document index...")
    _ = setup_rag(force_reindex=args.reindex)
    
    # Initialize Agent
    print("\nInitializing CoFina Agent...")
    agent = CoFinaAgent(API_KEY, logger)
    
    print("\n=== RAG Agent Ready ===")
    print("Type 'quit' to exit\n")
    
    # Simple user ID for demo
    user_id = "default_user"
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if not question:
            continue
        
        try:
            # Run Agent
            answer = agent.run(question, user_id=user_id)
            print(f"\nAnswer: {answer}\n")
            
        except Exception as e:
            print(f"\nError: {e}\n")
            logger.log_step("error", str(e))
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
