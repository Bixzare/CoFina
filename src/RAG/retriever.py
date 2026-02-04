from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def create_retriever(vector_store, k: int = 4):
    """Create a retriever from the vector store."""
    return vector_store.as_retriever(search_kwargs={"k": k})

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

def create_rag_chain(retriever, api_key: str):
    """Create RAG chain with Gemini LLM via CMU gateway."""
    llm = ChatOpenAI(
        model="gemini-2.5-pro",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        temperature=0.3
    )
    
    prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question. If you don't know the answer, say so.

Context: {context}

Question: {question}

Answer:"""
    )
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # We need to return source docs too for citation
    from langchain_core.runnables import RunnableParallel
    chain_with_source = RunnableParallel(
        {
            "result": chain, 
            "source_documents": retriever
        }
    )
    
    return chain_with_source

def query(chain, question: str):
    """Query the RAG chain."""
    # invoke returns a dict with 'result' and 'source_documents'
    return chain.invoke(question)
