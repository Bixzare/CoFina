"""
Response Verifier for CoFina - Grounds responses against RAG context.

Scores responses on a 0-1 scale:
  1.0     = perfectly grounded or appropriate conversational response
  0.7-0.9 = good general advice or mostly supported
  0.5-0.7 = acceptable but could be better
  < 0.5   = clearly wrong or unsupported (rare)

The verifier is GENEROUS — it understands that financial agents should be
helpful, and only penalizes responses that are demonstrably wrong or harmful.
"""

from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured verification output."""
    score: float = Field(
        description="Groundedness score 0.0-1.0 (1.0 = excellent)",
        ge=0.0,
        le=1.0,
    )
    reason: str = Field(description="Brief explanation (1-2 sentences max)")
    action: str = Field(
        description="accept (≥0.7) | retry (0.5-0.7) | refuse (<0.5)"
    )


def verify_response(
    question: str,
    answer: str,
    context: str,
    api_key: str,
) -> Dict[str, Any]:
    """
    Verify answer groundedness against retrieved context.

    Args:
        question: User's query
        answer: Agent's response
        context: RAG-retrieved context (or empty if none)
        api_key: LLM API key

    Returns:
        {"score": float, "reason": str, "action": str}
        Falls back to {"score": 0.85, "action": "accept"} on LLM errors.
    """
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        base_url="https://ai-gateway.andrew.cmu.edu/",
        temperature=0.0,
        timeout=10,
    )

    parser = JsonOutputParser(pydantic_object=VerificationResult)

    prompt = ChatPromptTemplate.from_template(
        """You are a quality judge for a financial assistant.

Score the ANSWER on how well it addresses the QUESTION, using the CONTEXT if provided.

QUESTION: {question}

CONTEXT: {context}

ANSWER: {answer}

SCORING RULES (be GENEROUS):

1. CONVERSATIONAL (greetings, acknowledgments, procedural help)
   → Score 0.95-1.0
   Examples: "Hello!", "I can help with that", "Let me guide you through registration"

2. GENERAL FINANCIAL ADVICE (budgeting, saving, debt strategies)
   → Score 0.8-0.95 if sound advice, even without specific citations
   Financial agents should help — don't penalize for being helpful!

3. PROFILE-BASED or CITED RESPONSES
   → Score 0.85-1.0 if accurately uses user data or cites documents

4. MINOR ISSUES (slightly vague, missing detail)
   → Score 0.6-0.8

5. CLEARLY WRONG or HARMFUL
   → Score < 0.5 (this should be RARE)

DECISION:
- accept  : score ≥ 0.7  (default for most responses)
- retry   : score 0.5-0.7
- refuse  : score < 0.5  (only if demonstrably wrong)

{format_instructions}

Output JSON only, no extra text."""
    )

    try:
        result = (prompt | llm | parser).invoke({
            "question": question,
            "context": context.strip() or "No specific context — general knowledge",
            "answer": answer,
            "format_instructions": parser.get_format_instructions(),
        })
        # Clamp score to valid range
        result["score"] = max(0.0, min(1.0, float(result["score"])))
        return result
    except Exception as exc:
        # Graceful degradation: assume answer is fine
        return {
            "score": 0.85,
            "reason": f"Verification unavailable ({type(exc).__name__}), accepting by default",
            "action": "accept",
        }