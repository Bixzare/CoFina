from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class VerificationResult(BaseModel):
    score: float = Field(description="Groundedness score between 0.0 and 1.0")
    reason: str = Field(description="Explanation for the score")
    action: str = Field(description="Recommended action: 'accept', 'retry', 'refuse'")

def verify_response(
    question: str, 
    answer: str, 
    context: str, 
    api_key: str
) -> Dict[str, Any]:
    """
    Verify the groundedness of the answer against the retrieved context.
    """
    llm = ChatOpenAI(
        model="gemini-2.5-pro",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        temperature=0.0
    )
    
    parser = JsonOutputParser(pydantic_object=VerificationResult)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a strict quality control judge for a financial RAG agent.
        
        Evaluate the following Answer based ONLY on the provided Context.
        
        Question: {question}
        
        Context:
        {context}
        
        Answer:
        {answer}
        
        Criteria:
        1. Groundedness: Does the answer only contain facts present in the context?
        2. Relevance: Does the answer directly address the question?
        3. Accuracy: Are numbers and citations correct?
        
        Score Guide:
        - 1.0: Perfect, fully supported by context.
        - 0.5-0.9: Mostly supported, minor hallucinations or missing citations.
        - < 0.5: Major hallucinations, irrelevant, or dangerous advice not in context.
        
        Action Guide:
        - accept: Score >= 0.8
        - retry: Score between 0.5 and 0.8 (e.g. minor fix needed)
        - refuse: Score < 0.5 (e.g. "I don't have enough info")
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "question": question,
            "context": context,
            "answer": answer,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        return {
            "score": 0.0, 
            "reason": f"Verification failed: {e}", 
            "action": "refuse"
        }
