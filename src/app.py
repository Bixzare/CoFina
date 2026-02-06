import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader

from RAG.chunking import chunk_documents
from RAG.index import create_vector_store, add_documents_to_index
from RAG.retriever import create_retriever, create_rag_chain, query

# Load environment variables
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
    
    hash_input = ""
    for pdf_file in pdf_files:
        hash_input += f"{pdf_file.name}:{pdf_file.stat().st_mtime}:"
    
    return hashlib.md5(hash_input.encode()).hexdigest()

def setup_rag(force_reindex: bool = False):
    """Initialize the RAG system."""
    persist_dir = Path("./chroma_db")
    hash_file = persist_dir / ".docs_hash"
    
    current_hash = get_docs_hash()
    if not current_hash:
        print("No PDFs found in ./docs directory")
        return None
    
    needs_reindex = force_reindex
    
    if not needs_reindex and persist_dir.exists() and hash_file.exists():
        with open(hash_file, 'r') as f:
            stored_hash = f.read().strip()
        
        if stored_hash == current_hash:
            print("Loading existing vector store (documents unchanged)...")
            vector_store = create_vector_store(API_KEY)
            
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
        
        try:
            vector_store.delete_collection()
            vector_store = create_vector_store(API_KEY)
        except:
            pass
        
        add_documents_to_index(vector_store, chunks)
        
        persist_dir.mkdir(exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(current_hash)
        
        print("Setting up RAG chain...")
        retriever = create_retriever(vector_store)
        chain = create_rag_chain(retriever, API_KEY)
        
        return chain

def detect_user_id_in_message(message: str) -> str:
    """
    Intelligently detect user ID from natural conversation.
    
    """
    import re
    
    # Pattern 1: "my id is X" or "my user id is X"
    pattern1 = r"(?:my (?:user )?id is |i'?m |this is |it'?s )\s*([a-zA-Z0-9_]+)"
    match = re.search(pattern1, message.lower())
    if match:
        return match.group(1)
    
    # Pattern 2: Standalone alphanumeric (e.g., "iliya0003")
    # Only if message is short (to avoid false positives)
    if len(message.split()) <= 3:
        pattern2 = r"\b([a-zA-Z][a-zA-Z0-9_]{3,15})\b"
        match = re.search(pattern2, message)
        if match:
            return match.group(1)
    
    return None

def main():
    """Main CLI loop with intelligent session management."""
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
    
    logger = AgentLogger()
    agent = CoFinaAgent(API_KEY, logger)
    
    current_user_id = "guest"
    authenticated = False
    
    print("\n" + "-" * 110)
    print("🤖 CoFina Agent: Your Intelligent Financial Assistant")
    print("💡 Ask me anything about personal finance, budgeting, or products!")
    print("🔐 For personalized advice, I'll guide you through authentication.")
    print("-" * 110)
    print("Commands: 'exit', 'quit', 'end' to close | 'logout' to switch users\n")

    while True:
        question = input("You 👤: ").strip()
        
        if question.lower() in ['quit', 'exit', 'end']:
            print("\n👋 Thank you for using CoFina! Have a great day!\n")
            break
        
        if question.lower() == 'logout':
            current_user_id = "guest"
            authenticated = False
            print("\n Logged out. You're now in guest mode.\n")
            continue
            
        if not question:
            continue

        # Smart user ID detection
        detected_id = detect_user_id_in_message(question)
        if detected_id and current_user_id == "guest":
            # Don't auto-switch, let agent handle it through tools
            pass

        try:
            answer = agent.run(question, user_id=current_user_id)
            print(f"\n🤖 CoFina: {answer}\n")
            
            # Check if agent authenticated user successfully
            # Look for verification success patterns
            if "successfully verified" in answer.lower() or "authenticated" in answer.lower():
                if detected_id:
                    current_user_id = detected_id
                    authenticated = True
                    print(f"---  Session authenticated as: {current_user_id} ---\n")
            
            # Check for successful registration
            if "successfully registered" in answer.lower() or "registration complete" in answer.lower():
                if detected_id:
                    current_user_id = detected_id
                    authenticated = True
                    print(f"---  Registered and logged in as: {current_user_id} ---\n")

        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            logger.log_step("error", str(e))

if __name__ == "__main__":
    main()