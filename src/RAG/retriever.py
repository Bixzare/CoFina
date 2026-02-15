from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import hashlib
import json
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Cache directory for RAG results
CACHE_DIR = "cache/rag_results"
os.makedirs(CACHE_DIR, exist_ok=True)

class RAGCache:
    """Simple cache for RAG results"""
    
    def __init__(self, cache_dir: str = CACHE_DIR, ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, question: str) -> str:
        """Generate cache key from question"""
        return hashlib.md5(question.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        """Get cache file path"""
        return os.path.join(self.cache_dir, f"{key}.json")
    
    def get(self, question: str) -> Optional[Dict[str, Any]]:
        """Get cached result if exists and not expired"""
        key = self._get_cache_key(question)
        path = self._get_cache_path(key)
        
        if not os.path.exists(path):
            return None
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.ttl:
                os.remove(path)  # Remove expired cache
                return None
            
            return data['result']
        except Exception:
            return None
    
    def set(self, question: str, result: Dict[str, Any]):
        """Cache a result"""
        key = self._get_cache_key(question)
        path = self._get_cache_path(key)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'result': result
        }
        
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def clear(self):
        """Clear all cache entries"""
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(self.cache_dir, filename))

# Global cache instance
_rag_cache = RAGCache()

def create_retriever(vector_store, k: int = 3):
    """
    Create a retriever from the vector store.
    Reduced k from 4 to 3 for faster retrieval.
    """
    return vector_store.as_retriever(
        search_kwargs={"k": k},
        search_type="similarity"  # Use similarity search for speed
    )

def format_docs(docs):
    """Format retrieved documents into a single string."""
    if not docs:
        return ""
    
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("filename", "unknown")
        # Truncate very long docs
        content = doc.page_content
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        formatted.append(f"[Source {i}: {source}]\n{content}")
    
    return "\n\n".join(formatted)

def create_rag_chain(retriever, api_key: str):
    """
    Create RAG chain with Gemini LLM via CMU gateway.
    Optimized for faster responses.
    """
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        temperature=0.1,  # Lower temperature for faster, more consistent responses
        max_tokens=500,    # Limit response length
        timeout=30         # Timeout after 30 seconds
    )
    
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful financial assistant. Answer the question based on the context provided.
        Keep your answer concise and to the point.

Context: {context}

Question: {question}

Answer:"""
    )
    
    # Simpler chain without RunnableParallel for speed
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain

def query(chain, question: str, use_cache: bool = True) -> Dict[str, Any]:
    """
    Query the RAG chain with caching.
    
    Args:
        chain: The RAG chain
        question: The question to ask
        use_cache: Whether to use cache
    
    Returns:
        Dict with 'result' and 'source_documents'
    """
    # Check cache first
    if use_cache:
        cached = _rag_cache.get(question)
        if cached:
            cached['cached'] = True
            return cached
    
    # If not cached, run the query
    try:
        # Get source documents first
        retriever = chain.steps[0]["context"].steps[0]  # Extract retriever from chain
        source_docs = retriever.invoke(question)
        
        # Get result
        result = chain.invoke(question)
        
        output = {
            "result": result,
            "source_documents": [
                {
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "source": doc.metadata.get("filename", "unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", "unknown")
                }
                for doc in source_docs
            ],
            "cached": False
        }
        
        # Cache the result
        if use_cache:
            _rag_cache.set(question, output)
        
        return output
        
    except Exception as e:
        return {
            "result": f"I encountered an error while searching: {str(e)}",
            "source_documents": [],
            "error": str(e),
            "cached": False
        }

def query_fast(chain, question: str) -> str:
    """
    Fast query that only returns the answer text, no source docs.
    Useful for quick responses.
    """
    try:
        return chain.invoke(question)
    except Exception as e:
        return f"Error: {str(e)}"

def clear_cache():
    """Clear the RAG cache"""
    _rag_cache.clear()
    print("🧹 RAG cache cleared")

def get_cache_stats() -> Dict[str, Any]:
    """Get statistics about the cache"""
    cache_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('.json')]
    
    total_size = 0
    for f in cache_files:
        path = os.path.join(CACHE_DIR, f)
        total_size += os.path.getsize(path)
    
    return {
        "entry_count": len(cache_files),
        "total_size_bytes": total_size,
        "total_size_mb": round(total_size / (1024 * 1024), 2),
        "cache_dir": CACHE_DIR
    }