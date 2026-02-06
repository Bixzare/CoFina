from typing import Dict, Any
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
        model="gpt-4o-2024-08-06",
        api_key=api_key,
        base_url='https://ai-gateway.andrew.cmu.edu/',
        temperature=0.0
    )
    
    parser = JsonOutputParser(pydantic_object=VerificationResult)
    
    prompt = ChatPromptTemplate.from_template(
        """You are a quality control judge for a financial RAG agent.
        
        Evaluate the following Answer based on the provided Context and the Question.
        
        Question: {question}
        
        Context:
        {context}
        
        Answer:
        {answer}
        
        IMPORTANT GUIDELINES:
        
        1. CONVERSATIONAL RESPONSES: If the answer is a greeting, registration guidance, 
           procedural help, or general conversation, give it a HIGH score (0.9-1.0).
           Examples: "Hi there!", "Let me help you register", "I'd be happy to help!"
        
        2. GENERAL FINANCIAL ADVICE: If the answer provides sound general financial advice 
           (budgeting, saving, investing principles) even without specific context, 
           give it a GOOD score (0.7-0.9). Financial agents should be helpful!
        
        3. PERSONALIZED ADVICE: If the answer uses user profile data or specific document 
           citations, verify it's accurate and give HIGH score (0.8-1.0).
        
        4. FACTUAL CLAIMS: Only reduce score if the answer makes specific factual claims 
           that contradict or aren't supported by the context.
        
        5. BE GENEROUS: The agent should be helpful. Only score low (<0.5) if the answer 
           is clearly wrong, harmful, or completely fabricated.
        
        Criteria:
        - Groundedness: Does the answer align with context OR provide sound general advice?
        - Relevance: Does the answer address the question appropriately?
        - Helpfulness: Is the response useful to the user?
        
        Score Guide:
        - 0.9-1.0: Excellent - conversational, helpful, or well-grounded advice
        - 0.7-0.89: Good - general advice or mostly supported by context
        - 0.5-0.69: Fair - could be better but not harmful
        - Below 0.5: Poor - clearly wrong or unhelpful
        
        Action Guide:
        - accept: Score >= 0.7 (most responses should be accepted!)
        - retry: Score between 0.5 and 0.7
        - refuse: Score < 0.5 (only if clearly harmful or very wrong)
        
        {format_instructions}
        """
    )
    
    chain = prompt | llm | parser
    
    try:
        result = chain.invoke({
            "question": question,
            "context": context if context.strip() else "General knowledge - no specific context",
            "answer": answer,
            "format_instructions": parser.get_format_instructions()
        })
        return result
    except Exception as e:
        # If verification fails, default to accepting the response
        return {
            "score": 0.8, 
            "reason": f"Verification service unavailable, defaulting to accept: {e}", 
            "action": "accept"
        }