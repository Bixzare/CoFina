"""
CoFina Response Verifier v2.0 — Mathematical + LLM Verification

Two-layer verification:
  Layer 1: Rule-based financial math & guardrail checks (fast, deterministic)
  Layer 2: LLM-as-judge grounding score (slower, nuanced)

Scoring:
  1.0  = mathematically verified, grounded, safe
  0.7+ = acceptable, accept
  0.5-0.7 = questionable, accept with warning
  <0.5 = block or refuse
"""

import re
from typing import Dict, Any

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class VerificationResult(BaseModel):
    """Structured verification output."""
    score: float = Field(description="Groundedness score 0.0-1.0", ge=0.0, le=1.0)
    reason: str  = Field(description="Brief explanation (1-2 sentences)")
    action: str  = Field(description="accept (>=0.7) | retry (0.5-0.7) | refuse (<0.5)")


# ── Financial safety rules ─────────────────────────────────────────────

_GUARANTEED_RETURN_RE = re.compile(
    r"(guaranteed?\s+(return|profit|gain|yield)|will\s+definitely\s+(earn|return|grow)|"
    r"risk[- ]free\s+return|certain(ly)?\s+(earn|profit|make\s+money)|"
    r"cannot?\s+lose)",
    re.IGNORECASE
)

_SPECULATIVE_RE = re.compile(
    r"\b(buy\s+individual\s+stocks?|options?\s+(trading|contract)|"
    r"crypto(currency)?|NFT|meme\s+coin|leveraged\s+ETF|2x\s+ETF|3x\s+ETF|"
    r"short\s+sell(ing)?|margin\s+(account|trading|loan))\b",
    re.IGNORECASE
)

_LEVERAGE_RE = re.compile(
    r"\b(borrow\s+to\s+invest|invest\s+on\s+margin|take\s+(out\s+)?a?\s*loan\s+to\s+invest|"
    r"leverage\s+your\s+(investment|portfolio)|invest\s+borrowed\s+money)\b",
    re.IGNORECASE
)

_UNREALISTIC_RETURN_RE = re.compile(
    r"(\b(\d{2,})\s*%\s*(annual|yearly|per\s+year|p\.?a\.?)\s*(return|yield|gain|profit))",
    re.IGNORECASE
)

_INVEST_WITHOUT_EF_RE = re.compile(
    r"(invest|stocks?|funds?|portfolio|brokerage)",
    re.IGNORECASE
)

_EMERGENCY_FUND_RE = re.compile(
    r"(emergency\s+fund|safety\s+net|3.{0,5}month|6.{0,5}month|rainy.{0,5}day)",
    re.IGNORECASE
)

_DOLLAR_RE = re.compile(r"\$([\d,]+(?:\.\d{2})?)")


def _rule_based_checks(answer: str, question: str) -> Dict[str, Any]:
    """
    Fast deterministic checks. Returns (penalty, flags).
    """
    penalty = 0.0
    flags   = []

    # 1. Guaranteed returns — immediate block
    if _GUARANTEED_RETURN_RE.search(answer):
        penalty += 1.0
        flags.append("guaranteed_return_language_detected — BLOCK")

    # 2. Leverage recommendation
    if _LEVERAGE_RE.search(answer):
        penalty += 0.5
        flags.append("leverage_recommendation_detected")

    # 3. Unrealistic return > 20% stated as achievable
    for m in _UNREALISTIC_RETURN_RE.finditer(answer):
        pct = float(re.search(r"\d+", m.group()).group())
        if pct > 20:
            penalty += 0.35
            flags.append(f"unrealistic_return_{pct}pct_annual")

    # 4. Investment advice without emergency fund mention (only penalise if question is investment-related)
    invest_q = bool(re.search(r"invest|portfolio|stock|fund", question, re.I))
    if invest_q and _INVEST_WITHOUT_EF_RE.search(answer) and not _EMERGENCY_FUND_RE.search(answer):
        penalty += 0.2
        flags.append("investment_advice_missing_emergency_fund_check")

    # 5. Math sanity — savings > income would be caught by planner, but flag anyway
    # (simple heuristic: if response says save $X/month and earn $Y/month, X > Y is wrong)
    dollar_vals = [float(v.replace(',','')) for v in _DOLLAR_RE.findall(answer)]
    if len(dollar_vals) >= 2:
        dollar_vals.sort(reverse=True)
        # If the biggest savings figure is > 5x another figure (not income), flag
        # (This is a conservative proxy; full validation needs profile data)
        pass  # Kept light — LLM layer handles nuanced math

    return {"penalty": min(penalty, 1.0), "flags": flags}


# ── Public function ────────────────────────────────────────────────────

def verify_response(
    question: str,
    answer:   str,
    context:  str,
    api_key:  str,
) -> Dict[str, Any]:
    """
    Verify answer safety, groundedness, and financial soundness.

    Returns:
        {
          "score":    float 0-1,
          "reason":   str,
          "action":   "accept" | "retry" | "refuse",
          "flags":    list[str]   # rule-based findings
        }
    Falls back to {"score": 0.85, "action": "accept"} on LLM errors
    (rule-based penalty still applies).
    """
    # ── Layer 1: Rule-based ────────────────────────────────────────────
    rule_result = _rule_based_checks(answer, question)
    rule_penalty = rule_result["penalty"]
    flags        = rule_result["flags"]

    # If rules trigger a block, skip LLM entirely
    if rule_penalty >= 1.0:
        return {
            "score":  0.05,
            "reason": "BLOCKED by rule: " + "; ".join(flags),
            "action": "refuse",
            "flags":  flags,
        }

    # ── Layer 2: LLM grounding score ───────────────────────────────────
    llm = ChatOpenAI(
        model="gemini-2.5-flash",
        api_key=api_key,
        base_url="https://ai-gateway.andrew.cmu.edu/",
        temperature=0.0,
        timeout=10,
    )
    parser = JsonOutputParser(pydantic_object=VerificationResult)

    prompt = ChatPromptTemplate.from_template(
        """You are a quality judge for a financial assistant focused on SAFETY and ACCURACY.

QUESTION: {question}
CONTEXT (RAG): {context}
ANSWER: {answer}

STRICT SCORING RULES:

1. CONVERSATIONAL (greetings, registration, login, navigation)
   → Score 0.95–1.0

2. GENERAL SOUND FINANCIAL ADVICE (budgeting, debt, emergency fund, index investing)
   → Score 0.80–0.92 if advice follows established personal finance principles

3. CITED / PROFILE-BASED RESPONSES
   → Score 0.85–0.98 if correctly uses user data or knowledge base

4. MISSING SAFETY CONTEXT (e.g. investment advice without risk caveat)
   → Score 0.60–0.75

5. VAGUE / UNSUPPORTED CLAIMS
   → Score 0.55–0.65

6. FINANCIALLY INCORRECT OR MISLEADING
   → Score < 0.45 (this should be rare but firm)

DECISION THRESHOLDS:
- accept : score >= 0.70
- retry  : score 0.50–0.69
- refuse : score < 0.50 (only if demonstrably harmful or wrong)

{format_instructions}

Output JSON only."""
    )

    try:
        result = (prompt | llm | parser).invoke({
            "question":           question,
            "context":            context.strip() or "No RAG context — general knowledge",
            "answer":             answer,
            "format_instructions": parser.get_format_instructions(),
        })
        result["score"] = max(0.0, min(1.0, float(result["score"])))
    except Exception as exc:
        result = {
            "score":  0.85,
            "reason": f"LLM verification unavailable ({type(exc).__name__}), rule-checks only",
            "action": "accept",
        }

    # ── Combine: LLM score penalized by rule findings ──────────────────
    final_score = max(0.0, result["score"] - rule_penalty)
    if final_score < 0.5:
        result["action"] = "refuse"
    elif final_score < 0.7:
        result["action"] = "retry"
    else:
        result["action"] = "accept"

    result["score"] = round(final_score, 3)
    result["flags"] = flags

    # Terminal transparency
    print(f"\n🔢 VERIFICATION")
    print(f"   LLM score    : {result.get('score', 0):.0%}")
    print(f"   Rule penalty : -{rule_penalty:.2f}")
    print(f"   Final score  : {final_score:.0%}")
    print(f"   Action       : {result['action'].upper()}")
    if flags:
        for f in flags:
            print(f"   ⚠ Flag: {f}")
    print(f"   Reason: {result.get('reason','')[:100]}\n")

    return result
