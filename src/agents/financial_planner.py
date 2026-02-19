"""
Financial Planner Agent - Personalised financial planning with LLM-driven intent.

Intent classification
─────────────────────
Uses a lightweight LangChain LLM call with structured output to classify the
user's intent into one of a fixed set of categories. No keyword matching.

Two-step topic plan flow
────────────────────────
1. User asks about a topic → agent composes plan, shows it, asks "Save as PDF?"
2. User confirms → PDF generated from cached plan text.
3. Explicit full plan → PDF always generated immediately.
4. Guest users → general best-practice plan PDF.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from tools.user_profile import get_user_profile
from tools.generatePlan import create_financial_plan_pdf, create_topic_plan_pdf
from tools.financial_calculator import calculate_simple_interest
from utils.cache import get_cache


# ── Structured output schema ──────────────────────────────────────────────────

class FinancialIntent(BaseModel):
    """Classified intent for a financial planning query."""
    intent: Literal[
        "full_plan",       # User wants a complete personal financial plan / PDF
        "car",             # Car / vehicle buying plan
        "house",           # Home / property buying plan
        "debt",            # Debt repayment plan
        "retirement",      # Retirement planning
        "investment",      # Investment strategy
        "emergency",       # Emergency fund plan
        "budget",          # Budgeting plan
        "savings",         # Savings goals plan
        "pdf_confirm",     # User is confirming they want a PDF of the last plan
        "pdf_decline",     # User is declining the PDF offer
        "calculate",       # Interest / financial calculation with numbers
        "advice",          # General question / advice — no plan generation needed
    ] = Field(description="The user's financial planning intent")
    wants_pdf_now: bool = Field(
        default=False,
        description=(
            "True only when the user is EXPLICITLY asking to generate, save, "
            "download, or create a PDF or document right now — not just asking "
            "a question about a topic."
        ),
    )


_INTENT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an intent classifier for a financial planning assistant.
Classify the user's message into exactly one intent category.

Intent definitions:
- full_plan    : User wants a complete personalised financial plan or PDF of their overall finances
- car          : Questions about buying a car / vehicle / auto
- house        : Questions about buying a home, property, or mortgage
- debt         : Questions about repaying debt, loans, credit cards
- retirement   : Questions about retirement saving or planning
- investment   : Questions about investing, stocks, index funds, portfolio
- emergency    : Questions about emergency fund or financial safety net
- budget       : Questions about budgeting or the 50/30/20 rule
- savings      : Questions about saving money or savings goals
- pdf_confirm  : User is saying yes / confirming they want a PDF of a previously discussed plan
- pdf_decline  : User is saying no / declining the PDF offer
- calculate    : User wants a numerical interest or financial calculation (must have numbers)
- advice       : General educational question, definition, or concept explanation — NOT a request to create a plan

wants_pdf_now must be TRUE only if the user is explicitly requesting to generate/save/create/download a PDF or plan document RIGHT NOW. A question like "What is financial planning?" must have wants_pdf_now=False and intent=advice.

Examples:
"What is financial planning?" → intent=advice, wants_pdf_now=false
"Generate a financial plan for me" → intent=full_plan, wants_pdf_now=true
"How should I buy a house in 2027?" → intent=house, wants_pdf_now=false
"Generate a PDF for that" → intent=pdf_confirm, wants_pdf_now=true
"What is debt?" → intent=advice, wants_pdf_now=false
"Help me pay off my debt" → intent=debt, wants_pdf_now=false
"Create a car buying plan for me as a PDF" → intent=car, wants_pdf_now=true
"I owe $2000, how do I pay it?" → intent=debt, wants_pdf_now=false
"yes please generate it" → intent=pdf_confirm, wants_pdf_now=true
""",
    ),
    ("human", "{query}"),
])


class FinancialPlannerAgent:
    """Specialised agent for financial planning with LLM-driven intent classification."""

    def __init__(self, rag_retriever=None, api_key: str = ""):
        self.rag_retriever = rag_retriever
        self.api_key = api_key
        self.cache = get_cache()
        # State for two-step topic plan → PDF flow
        self._pending_topic:     Optional[str]  = None
        self._pending_plan_text: Optional[str]  = None
        self._pending_profile:   Optional[Dict] = None
        self._pending_user_id:   Optional[str]  = None
        # Lazy-initialised classifier
        self._classifier = None

    def _get_classifier(self):
        """Lazily build the structured-output LLM classifier."""
        if self._classifier is None:
            if not self.api_key:
                return None
            llm = ChatOpenAI(
                model="gemini-2.5-flash",
                temperature=0,
                api_key=self.api_key,
                base_url="https://ai-gateway.andrew.cmu.edu/",
                max_tokens=100,   # classification only — keep fast
                timeout=10,
            )
            self._classifier = _INTENT_PROMPT | llm.with_structured_output(FinancialIntent)
        return self._classifier

    def _classify(self, query: str) -> FinancialIntent:
        """
        Classify the user query via LLM structured output.
        Falls back to a safe 'advice' intent if the LLM call fails.
        """
        classifier = self._get_classifier()
        if classifier is None:
            return FinancialIntent(intent="advice", wants_pdf_now=False)
        try:
            return classifier.invoke({"query": query})
        except Exception:
            # If classification fails, never trigger plan generation silently
            return FinancialIntent(intent="advice", wants_pdf_now=False)

    # ── Public entry point ────────────────────────────────────────────────────

    def process(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        profile = self._load_profile(user_id)
        classified = self._classify(query)
        intent = classified.intent
        wants_pdf = classified.wants_pdf_now

        # ── Pending PDF confirmation / decline ────────────────────────────
        if self._pending_plan_text:
            if intent == "pdf_confirm" or (intent not in (
                "advice", "calculate"
            ) and wants_pdf):
                return self._generate_pending_pdf()
            if intent == "pdf_decline":
                self._clear_pending()
                return {
                    "action": "advice",
                    "message": "No problem! Let me know if you'd like the PDF later.",
                    "data": {},
                }

        # ── Explicit full plan ────────────────────────────────────────────
        if intent == "full_plan":
            return self._create_full_plan(query, user_id, profile)

        # ── Topic plan ────────────────────────────────────────────────────
        topic_intents = {"car", "house", "debt", "retirement",
                         "investment", "emergency", "budget", "savings"}
        if intent in topic_intents:
            return self._present_topic_plan(
                query, user_id, profile,
                topic=intent,
                generate_pdf=wants_pdf,
            )

        # ── Interest calculation ──────────────────────────────────────────
        if intent == "calculate":
            return self._calculate_interest(query)

        # ── General advice (no plan, no PDF) ─────────────────────────────
        return self._quick_advice(query, user_id, profile)

    # ── Topic plan: show reasoning, optionally save PDF ───────────────────────

    def _present_topic_plan(self, query: str, user_id: str,
                             profile: Dict, topic: str,
                             generate_pdf: bool) -> Dict[str, Any]:
        income  = profile.get("income", 0)
        monthly = float(income) / 12 if income else 0
        name    = (f"{profile.get('first_name', '')} "
                   f"{profile.get('other_names', '')}").strip() or "You"
        prefs   = profile.get("preferences", {}) or {}
        year_match  = re.search(r"\b(20\d{2})\b", query)
        target_year = year_match.group(1) if year_match else None

        plan_text = self._compose_topic_plan(
            topic, name, income, monthly, prefs, profile, query, target_year
        )

        # Cache for later PDF step
        self._pending_topic     = topic
        self._pending_plan_text = plan_text
        self._pending_profile   = profile
        self._pending_user_id   = user_id

        if generate_pdf:
            return self._generate_pending_pdf()

        msg = (
            f"{plan_text}\n\n"
            f"{'─' * 44}\n"
            f"💾 Would you like me to save this as a **PDF plan**?"
        )
        return {"action": "plan_presented", "message": msg,
                "data": {"topic": topic, "profile": profile}}

    # ── PDF generation ────────────────────────────────────────────────────────

    def _generate_pending_pdf(self) -> Dict[str, Any]:
        result = create_topic_plan_pdf(
            user_id=self._pending_user_id or "guest",
            topic=self._pending_topic or "financial",
            plan_content=self._pending_plan_text or "",
            profile_data=self._pending_profile,
            plan_name=f"{(self._pending_topic or 'Financial').title()} Plan",
        )
        topic_label = (self._pending_topic or "financial").title()
        self._clear_pending()

        if result["success"]:
            msg = (
                f"✅ Your **{topic_label} Plan** PDF has been generated!\n"
                f"📄 Saved to: {result['filepath']}"
            )
        else:
            msg = (
                f"⚠️ PDF generation failed: {result.get('error', 'unknown error')}.\n"
                "Your plan is shown above."
            )
        return {"action": "plan_created", "message": msg, "data": {"pdf_result": result}}

    def _clear_pending(self):
        self._pending_topic = self._pending_plan_text = None
        self._pending_profile = self._pending_user_id = None

    # ── Full profile plan ─────────────────────────────────────────────────────

    def _create_full_plan(self, query: str, user_id: str, profile: Dict) -> Dict[str, Any]:
        if not profile:
            return self._guest_plan(user_id)

        goals      = profile.get("goals", {}) or {}
        short_term = goals.get("short_term") or "Build emergency fund; pay down debt"
        long_term  = goals.get("long_term")  or "Retirement; house down payment"

        result = create_financial_plan_pdf(
            user_id=user_id,
            profile_data=profile,
            short_term_goals=short_term,
            long_term_goals=long_term,
            plan_name="Financial Plan",
            include_projections=True,
        )

        income  = profile.get("income", 0)
        monthly = float(income) / 12 if income else 0
        name    = (f"{profile.get('first_name', '')} "
                   f"{profile.get('other_names', '')}").strip() or user_id
        prefs   = profile.get("preferences", {}) or {}

        if result["success"]:
            msg = (
                f"✅ Your personalised financial plan has been generated, **{name}**!\n"
                f"📄 Saved to: {result['filepath']}\n\n"
                f"**Plan Summary**\n"
                f"• Monthly income: ${monthly:,.0f}  |  Annual: ${float(income):,.0f}\n"
                f"• Savings target (20%): ${monthly * 0.2:,.0f}/month\n"
                f"• Debt strategy: {prefs.get('debt_strategy', 'N/A').title()}\n"
                f"• Short-term goals: {short_term}\n"
                f"• Long-term goals: {long_term}"
            )
        else:
            msg = (
                f"⚠️ PDF generation failed: {result.get('error', 'unknown error')}.\n"
                "Here is your plan summary:\n\n"
                + self._profile_summary(profile)
            )
        return {"action": "plan_created", "message": msg,
                "data": {"profile": profile, "pdf_result": result}}

    # ── Guest plan ────────────────────────────────────────────────────────────

    def _guest_plan(self, user_id: str = "guest") -> Dict[str, Any]:
        result = create_financial_plan_pdf(
            user_id=user_id,
            profile_data={},
            short_term_goals="Build a 3-month emergency fund; pay off high-interest debt",
            long_term_goals="Achieve financial independence; build retirement savings",
            plan_name="General Financial Plan",
            include_projections=False,
        )
        if result["success"]:
            msg = (
                "✅ Here is a general financial plan based on CoFina best practices!\n"
                f"📄 Saved to: {result['filepath']}\n\n"
                "💡 Register and log in to get a plan personalised to your income, "
                "debts, and goals."
            )
        else:
            msg = (
                "Here is a general financial plan:\n\n"
                "**50/30/20 Budget Rule**\n"
                "• 50% Needs · 30% Wants · 20% Savings & Debt\n\n"
                "**Priority Order**\n"
                "1. Emergency fund (3–6 months of expenses)\n"
                "2. Pay off high-interest debt\n"
                "3. Invest in low-cost index funds\n"
                "4. Maximise retirement contributions\n\n"
                "Register to get a personalised plan with your exact numbers."
            )
        return {"action": "plan_created", "message": msg, "data": {"pdf_result": result}}

    # ── Topic plan text composer ──────────────────────────────────────────────

    def _compose_topic_plan(self, topic: str, name: str, income,
                             monthly: float, prefs: Dict, profile: Dict,
                             query: str, target_year: Optional[str] = None) -> str:
        has_income = bool(income and float(income) > 0)
        max_car_pmt = f"${monthly * 0.15:,.0f}" if has_income else "15% of monthly income"
        down_car    = f"${float(income) * 0.02:,.0f}" if has_income else "20% of car price"
        house_down  = f"${float(income) * 0.20:,.0f}" if has_income else "20% of home price"
        sav_20      = f"${monthly * 0.20:,.0f}/month" if has_income else "20% of monthly income"
        year_note   = f" by {target_year}" if target_year else ""

        if topic == "car":
            return (
                f"🚗 **Car Buying Plan — {name}**\n\n"
                f"**1. Budget**\n"
                f"• Max monthly payment (principal + interest): {max_car_pmt}\n"
                f"• Target down payment (≥20%): {down_car}\n"
                f"• Total car costs (payment + insurance + fuel + maintenance) ≤ 15–20% of monthly income.\n\n"
                f"**2. Vehicle Research**\n"
                f"• Prioritise reliability and low running costs (Toyota, Honda, Mazda).\n"
                f"• Consider Certified Pre-Owned (1–3 years old) — avoids steepest depreciation.\n"
                f"• Check Consumer Reports, IIHS safety ratings, and resale value.\n\n"
                f"**3. Financing**\n"
                f"• Get pre-approved at your bank or credit union before visiting a dealership.\n"
                f"• Compare APR across at least 3 lenders.\n"
                f"• Loan term: 36–48 months max to minimise total interest.\n\n"
                f"**4. Monthly Cost Estimate**\n"
                f"• Insurance: $100–$200/month\n"
                f"• Fuel: $80–$150/month\n"
                f"• Maintenance reserve: $50–$100/month\n\n"
                f"**5. Negotiation**\n"
                f"• Focus on out-the-door price, not monthly payment.\n"
                f"• Decline dealer add-ons (paint protection, fabric guard, etc.).\n"
                f"• Being willing to walk away is your strongest tool."
            )

        if topic == "house":
            triple_income = f"${float(income) * 3:,.0f}" if has_income else "3× annual income"
            mortgage_max  = f"${monthly * 0.28:,.0f}" if has_income else "28% of monthly income"
            return (
                f"🏠 **Home Buying Plan — {name}{year_note}**\n\n"
                f"**1. Savings Target**\n"
                f"• Down payment (20% avoids PMI): {house_down}\n"
                f"• Closing costs (~3%): {f'${float(income)*0.03:,.0f}' if has_income else '~3% of price'}\n"
                f"• Post-purchase emergency reserve: 3–6 months of expenses\n\n"
                f"**2. Affordability Rules**\n"
                f"• Home price target: ≤ {triple_income}\n"
                f"• Monthly mortgage: ≤ {mortgage_max}\n"
                f"• DTI ratio: < 36%\n\n"
                f"**3. Mortgage Readiness**\n"
                f"• Credit score target: 740+ for best rates\n"
                f"• Get pre-approved before house-hunting\n"
                f"• Stable employment history: 2+ years\n\n"
                f"**4. Steps To Take Now**\n"
                f"1. Automate {sav_20} into a dedicated HYSA.\n"
                f"2. Pay down debt to lower DTI.\n"
                f"3. Avoid new credit applications.\n"
                f"4. Research first-time buyer grants in your area.\n\n"
                f"**5. Hidden Costs**\n"
                f"• Property tax: 1–2% of value/year\n"
                f"• Home insurance: $1,000–$2,000/year\n"
                f"• Maintenance reserve: 1% of value/year"
            )

        if topic == "debt":
            debts    = profile.get("debts", []) or []
            strategy = prefs.get("debt_strategy", "avalanche").lower()
            debt_lines  = ""
            total_remaining = 0.0
            for d in debts:
                rem = float(d.get("remaining_amount", 0))
                total_remaining += rem
                debt_lines += (
                    f"  • {d.get('name', 'Debt')}: ${rem:,.0f} remaining "
                    f"@ {d.get('interest_rate', 0)}% — "
                    f"min. ${d.get('minimum_payment', 0):,.0f}/mo\n"
                )
            strat_desc = (
                "**Snowball** — clear smallest balance first for momentum."
                if strategy == "snowball"
                else "**Avalanche** — attack highest interest rate first to minimise total interest."
            )
            no_debts_msg = "  • No debts on file — add them in your profile for a tailored plan.\n"
            debt_reasoning = f"Your total remaining debt is ${total_remaining:,.0f}. " if total_remaining else ""
            months_to_free = (
                round(total_remaining / (monthly * 0.20))
                if has_income and monthly > 0 and total_remaining > 0 else None
            )
            timeline_line = (
                f"**Timeline Estimate**\n"
                f"At {sav_20} toward debt: ~{months_to_free} months to be debt-free.\n\n"
                if months_to_free else ""
            )
            return (
                f"💳 **Debt Repayment Plan — {name}**\n\n"
                f"{debt_reasoning}"
                f"A structured strategy frees up cash flow and reduces financial stress.\n\n"
                f"**Your Debts**\n"
                f"{debt_lines or no_debts_msg}\n"
                f"**Strategy: {strat_desc}**\n\n"
                f"**Step-by-Step**\n"
                f"1. Pay minimums on ALL debts every month.\n"
                f"2. Build a $500–$1,000 starter emergency fund first.\n"
                f"3. Direct every extra dollar toward your target debt.\n"
                f"4. When cleared, roll that payment onto the next debt.\n"
                f"5. Once debt-free, redirect payments into savings and investing.\n\n"
                f"{timeline_line}"
                f"**Tips to Accelerate**\n"
                f"• Use windfalls (tax refunds, bonuses) for lump-sum paydowns.\n"
                f"• Consider 0% APR balance transfers for high-interest credit card debt.\n"
                f"• Avoid new debt until existing balances are cleared."
            )

        if topic == "retirement":
            ret_age = profile.get("profile", {}).get("retirement_age", 65)
            age     = profile.get("profile", {}).get("age", 30)
            years   = max(int(ret_age) - int(age), 1) if ret_age and age else 35
            target  = f"${float(income) * 25:,.0f}" if has_income else "25× annual expenses"
            return (
                f"🏦 **Retirement Plan — {name}**\n\n"
                f"**Your Numbers**\n"
                f"• Retire at {ret_age} — {years} years away\n"
                f"• Nest egg target (25× rule): {target}\n"
                f"• Monthly savings needed (15% rule): {sav_20}\n\n"
                f"**Priority Order**\n"
                f"1. 401(k) up to full employer match — free money first.\n"
                f"2. Max Roth IRA ($7,000/year for 2025).\n"
                f"3. Return to 401(k) up to IRS limit ($23,500 for 2025).\n"
                f"4. Taxable brokerage for additional savings.\n\n"
                f"**Investment Mix (Moderate Risk)**\n"
                f"• 80% global equity index funds (VT / VTSAX)\n"
                f"• 20% bond index funds — rebalance annually.\n"
                f"• Shift more conservative as you approach retirement.\n\n"
                f"**Key Principles**\n"
                f"• Start now — time in market beats timing the market.\n"
                f"• Keep costs low: expense ratio < 0.20%.\n"
                f"• Never cash out early — penalties destroy decades of growth."
            )

        if topic == "investment":
            return (
                f"📈 **Investment Plan — {name}**\n\n"
                f"**Prerequisites**\n"
                f"• Emergency fund complete ✅\n"
                f"• High-interest debt paid ✅ (guaranteed return beats market)\n\n"
                f"**Account Priority**\n"
                f"1. 401(k) to employer match\n"
                f"2. Roth IRA (max $7,000/year)\n"
                f"3. 401(k) to IRS limit\n"
                f"4. Taxable brokerage\n\n"
                f"**Portfolio (Moderate Risk)**\n"
                f"• 70–80% global equity index (VT / VTSAX / FZROX)\n"
                f"• 20–30% bond index (BND / FXNAX)\n"
                f"• Rebalance annually — don't chase performance.\n\n"
                f"**Golden Rules**\n"
                f"• Time in market > timing the market.\n"
                f"• Keep expense ratios below 0.20%.\n"
                f"• Invest consistently; don't sell during downturns."
            )

        if topic == "budget":
            needs = f"${monthly * .5:,.0f}" if has_income else "50% of income"
            wants = f"${monthly * .3:,.0f}" if has_income else "30% of income"
            saves = f"${monthly * .2:,.0f}" if has_income else "20% of income"
            return (
                f"📊 **Monthly Budget Plan — {name}**\n\n"
                f"**50/30/20 Rule**\n"
                f"• Needs (rent, food, transport, utilities): {needs}/month\n"
                f"• Wants (dining, entertainment, subscriptions): {wants}/month\n"
                f"• Savings & Debt repayment: {saves}/month\n\n"
                f"**Tips**\n"
                f"1. Track every expense for 30 days — awareness drives change.\n"
                f"2. Automate your savings transfer on payday.\n"
                f"3. Review subscriptions quarterly and cancel unused ones.\n"
                f"4. Build a monthly 'no-spend' day to reset habits.\n\n"
                f"**Tools:** YNAB, Copilot, or a simple spreadsheet."
            )

        if topic == "savings":
            return (
                f"💰 **Savings Plan — {name}**\n\n"
                f"**Step 1: Emergency Fund** ({sav_20} until complete)\n"
                f"• Target: 3–6 months of essential expenses\n"
                f"• Account: High-yield savings (4%+ APY)\n\n"
                f"**Step 2: Goal-Based Savings**\n"
                f"• Open separate sub-accounts for each goal.\n"
                f"• Automate a fixed amount to each goal on payday.\n\n"
                f"**Step 3: Invest the Rest**\n"
                f"• After emergency fund is complete, redirect surplus to investments.\n\n"
                f"**Rate Context**\n"
                f"• HYSA: ~4–5% (short-term, < 3 years)\n"
                f"• Stock market: ~7–10% long-term (3+ years)"
            )

        # emergency / fallback
        return (
            f"🛡 **Emergency Fund Plan — {name}**\n\n"
            f"**Target:** 3–6 months of essential expenses\n"
            f"**Monthly savings:** {sav_20}\n\n"
            f"**Steps**\n"
            f"1. Open a high-yield savings account (4%+ APY).\n"
            f"2. Automate a fixed transfer every payday.\n"
            f"3. Treat it as a non-negotiable bill.\n"
            f"4. Replenish immediately after any withdrawal.\n\n"
            f"**Why It Matters**\n"
            f"• Prevents going into debt for unexpected costs.\n"
            f"• Gives you confidence to invest and take career risks."
        )

    # ── Interest calculation ──────────────────────────────────────────────────

    def _calculate_interest(self, query: str) -> Dict[str, Any]:
        nums = re.findall(r"(\d+\.?\d*)", query)
        if len(nums) >= 3:
            p, r, t = float(nums[0]), float(nums[1]), float(nums[2])
            result  = calculate_simple_interest(p, r, t)
            return {
                "action": "calculation",
                "message": (
                    f"Simple interest on ${p:,.2f} at {r}% for {t} years:\n"
                    f"• Interest earned: ${result['interest_earned']:,.2f}\n"
                    f"• Total amount:    ${result['total_amount']:,.2f}"
                ),
                "data": result,
            }
        return {
            "action": "calculation",
            "message": "Please provide principal, rate%, and years. "
                       "Example: '1000 principal, 5% rate, 2 years'",
            "data": {},
        }

    # ── Quick advice ──────────────────────────────────────────────────────────

    def _quick_advice(self, query: str, user_id: str, profile: Dict) -> Dict[str, Any]:
        q       = query.lower()
        income  = profile.get("income", 0)
        monthly = float(income) / 12 if income else 0

        if "budget" in q:
            msg = "**50/30/20 Rule:** 50% Needs · 30% Wants · 20% Savings & Debt"
        elif "save" in q or "saving" in q:
            sug = f" (${monthly*0.2:,.0f}/month for you)" if monthly else ""
            msg = f"**Savings:** Automate 20% of income on payday{sug}. Use a high-yield savings account."
        elif "invest" in q:
            msg = "**Invest:** Build emergency fund first → low-cost index funds (VT/VTSAX). Stay consistent."
        elif "debt" in q or "owe" in q:
            msg = "**Debt:** Pay minimums on all, then attack highest-interest (avalanche) or smallest balance (snowball)."
        elif "emergency" in q:
            target = f" (~${monthly*3:,.0f}–${monthly*6:,.0f} for you)" if monthly else ""
            msg = f"**Emergency Fund:** 3–6 months of expenses{target} in a high-yield savings account."
        elif "retirement" in q:
            msg = "**Retirement:** Save 15% of income. Capture employer 401(k) match first, then max Roth IRA."
        elif "financial plan" in q or "what is" in q or "explain" in q or "define" in q:
            msg = (
                "**Financial planning** is the process of setting money goals and creating a "
                "strategy to achieve them — covering budgeting, saving, debt management, "
                "investing, and retirement. It helps you make intentional decisions about "
                "your money so you can build long-term financial independence."
            )
        else:
            msg = "**Priority order:** Emergency fund → High-interest debt → Invest → Build wealth"

        return {"action": "advice", "message": msg, "data": {}}

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_profile(self, user_id: str) -> Dict:
        if not user_id or user_id == "guest":
            return {}
        try:
            return get_user_profile(user_id) or {}
        except Exception:
            return {}

    def _profile_summary(self, profile: Dict) -> str:
        income  = profile.get("income", 0)
        monthly = float(income) / 12 if income else 0
        name    = (f"{profile.get('first_name', '')} "
                   f"{profile.get('other_names', '')}").strip()
        lines = [f"**Name:** {name}"] if name else []
        if income:
            lines.append(f"**Annual income:** ${float(income):,.0f}  (${monthly:,.0f}/month)")
        prefs = profile.get("preferences", {}) or {}
        if prefs.get("risk_profile"):
            lines.append(f"**Risk profile:** {prefs['risk_profile'].title()}")
        if prefs.get("debt_strategy"):
            lines.append(f"**Debt strategy:** {prefs['debt_strategy'].title()}")
        return "\n".join(f"• {l}" for l in lines) if lines else "Profile data not available."