"""
Orchestrator Agent - Main LLM-driven coordinator for CoFina.

Key design decisions
────────────────────
1. When a registration session is active, user input routes DIRECTLY to
   registration_agent — the LLM is NOT involved mid-registration.

2. Intent classification for financial planning is handled by the
   FinancialPlannerAgent's own LLM classifier — NOT by keyword matching
   in the orchestrator.  The orchestrator's LLM calls financial_planning_flow
   when it judges the query is financial; the planner then classifies the
   fine-grained intent internally.

3. After successful registration the orchestrator updates current_user_id and
   authenticates the guardrail session automatically.
"""

from __future__ import annotations

import json
import time
import traceback
import uuid
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from agents.financial_planner import FinancialPlannerAgent
from agents.guardrail_agent import GuardrailAgent
from agents.market_agent import MarketAgent
from agents.monitor_agent import MonitorAgent
from agents.registration_agent import RegistrationAgent
from agents.summarizer_agent import SummarizerAgent
from agents.verifier import verify_response
from core.adaptive_control import AdaptiveController
from core.checkpoint import CheckpointManager
from core.evaluation import EvaluationMetrics
from core.memory_manager import MemoryManager
from core.state_manager import SessionPhase, StateManager
from tools.dateTime import TIME_TOOLS
from utils.logger import AgentLogger


class CoFinaOrchestrator:
    """Main orchestrator — exposes specialised sub-agents as LLM tools."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.logger  = AgentLogger()

        # Verification & timing
        self.turn_number            = 0
        self.last_rag_context       = ""
        self.last_retrieval_time_ms = 0

        # Core infrastructure
        self.state_manager       = StateManager()
        self.memory_manager      = MemoryManager()
        self.checkpoint_manager  = CheckpointManager()
        self.evaluator           = EvaluationMetrics()
        self.adaptive_controller = AdaptiveController()

        # Sub-agents — pass api_key to FinancialPlannerAgent for its LLM classifier
        self.registration_agent = RegistrationAgent()
        self.financial_planner  = FinancialPlannerAgent(api_key=api_key)
        self.market_agent       = MarketAgent()
        self.monitor_agent      = MonitorAgent()
        self.summarizer_agent   = SummarizerAgent(api_key)
        self.guardrail_agent    = GuardrailAgent()

        # RAG
        self.retriever = None
        try:
            from RAG.index import ensure_index
            from RAG.retriever import create_retriever
            vector_store   = ensure_index(api_key)
            self.retriever = create_retriever(vector_store)
            self.financial_planner.rag_retriever = self.retriever
            print("✅ RAG system ready")
        except Exception as exc:
            self.logger.log_step("warning", f"RAG init skipped: {exc}")
            print(f"⚠️  RAG unavailable: {exc}")

        # LLM (orchestrator level)
        self.llm = ChatOpenAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=api_key,
            base_url="https://ai-gateway.andrew.cmu.edu/",
        )
        self.tools          = self._build_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Session state
        self.current_session_id: str  = str(uuid.uuid4())[:8]
        self.current_user_id:    str  = "guest"
        self.conversation_history: List = []

    # ────────────────────────────────────────────────────────────────────
    # Tool definitions
    # ────────────────────────────────────────────────────────────────────

    def _build_tools(self) -> list:

        @tool
        def registration_flow(user_input: str) -> str:
            """Start or continue user registration / sign-up."""
            result = self.registration_agent.process(
                user_input,
                {"user_id": self.current_user_id, "session_id": self.current_session_id},
            )
            self._handle_registration_result(result)
            return json.dumps(result)

        @tool
        def financial_planning_flow(user_input: str) -> str:
            """
            Handle ALL financial planning requests including:
            - Creating or generating a financial plan or PDF
            - Topic-specific plans: car, house, debt, retirement, investment,
              budget, savings, emergency fund
            - Budget advice, savings goals, interest calculations
            - Confirming or declining a previously offered PDF

            Use this tool whenever the user mentions plans, budgets, goals,
            savings, debt, investing, retirement, car buying, or home buying.
            The tool internally classifies the fine-grained intent using an LLM
            and decides whether to generate a PDF.
            """
            profile = self._load_profile()
            result  = self.financial_planner.process(
                query=user_input,
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def market_research_flow(user_input: str) -> str:
            """Product searches, price comparisons, and affordability checks."""
            profile = self._load_profile()
            result  = self.market_agent.process(
                query=user_input,
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def monitoring_flow(user_input: str) -> str:
            """Financial alerts, spending tracking, and goal progress monitoring."""
            profile = self._load_profile()
            result  = self.monitor_agent.process(
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def login_flow(user_id: str, password: str) -> str:
            """
            Log an existing user in with their user_id and password.
            Ask for both together; call this tool as soon as you have both.
            Relay the returned 'message' word-for-word to the user.
            """
            from db.queries import verify_login, user_exists, get_user_profile
            if not user_exists(user_id):
                return json.dumps({
                    "success": False,
                    "message": (
                        f"⚠️  No account found for '{user_id}'. "
                        "Please check your User ID or say 'register' to create a new account."
                    ),
                })
            if verify_login(user_id, password):
                self.current_user_id = user_id
                self.guardrail_agent.authenticate_session(self.current_session_id, user_id)
                self.logger.log_step("login", {"user_id": user_id})
                profile    = get_user_profile(user_id) or {}
                first_name = profile.get("first_name", user_id)
                return json.dumps({
                    "success": True,
                    "message": (
                        f"✅ Welcome back, {first_name}! You're now logged in.\n"
                        "What would you like to do today? I can help with your "
                        "financial plan, debt tracking, savings goals, and more."
                    ),
                })
            return json.dumps({
                "success": False,
                "message": (
                    "❌ Incorrect password. Please try again, or say "
                    "'forgot password' if you'd like to reset it."
                ),
            })

        @tool
        def get_user_status() -> str:
            """Return the current user's authentication status."""
            return json.dumps({
                "user_id":       self.current_user_id,
                "authenticated": self.current_user_id != "guest",
                "session_id":    self.current_session_id,
            })

        @tool
        def get_my_profile() -> str:
            """
            Fetch the full profile of the currently logged-in user.
            Use when the user asks what CoFina knows about them.
            """
            if self.current_user_id == "guest":
                return json.dumps({"error": "Not logged in. Please log in first."})
            profile = self._load_profile()
            if not profile:
                return json.dumps({"error": "Profile not found."})
            return json.dumps(profile)

        @tool
        def search_financial_documents(query: str) -> str:
            """
            Search the financial knowledge base for concepts, definitions,
            and general guidance. Use for educational questions like
            'What is compound interest?' or 'How does a Roth IRA work?'
            """
            if not self.retriever:
                return json.dumps({"error": "Knowledge base not available"})

            t0   = time.perf_counter()
            docs = self.retriever.invoke(query)
            self.last_retrieval_time_ms = round((time.perf_counter() - t0) * 1000)

            self.last_rag_context = "\n\n".join([
                f"[{d.metadata.get('source', 'unknown')}]\n{d.page_content}"
                for d in docs[:3]
            ])
            results = [
                {"content": d.page_content, "source": d.metadata.get("source", "unknown")}
                for d in docs[:3]
            ]
            return json.dumps(results)

        return [
            registration_flow,
            login_flow,
            financial_planning_flow,
            market_research_flow,
            monitoring_flow,
            get_user_status,
            get_my_profile,
            search_financial_documents,
            *TIME_TOOLS,
        ]

    # ────────────────────────────────────────────────────────────────────
    # Main entry point
    # ────────────────────────────────────────────────────────────────────

    def process(self, user_query: str) -> str:
        self.logger.start_turn(user_query)
        self.last_retrieval_time_ms = 0

        # ── Guardrail ────────────────────────────────────────────────────
        guardrail = self.guardrail_agent.process(
            query=user_query,
            session_id=self.current_session_id,
            user_id=self.current_user_id,
        )
        if not guardrail["passed"]:
            response = self._guardrail_response(guardrail)
            self.logger.end_turn(response)
            return response

        _lc = user_query.lower().strip()

        # ── Intercept logout ─────────────────────────────────────────────
        if any(kw in _lc for kw in ("logout", "log out", "log off", "sign out")):
            self.logout()
            response = "✅ You've been logged out. See you next time!"
            self._append_history(user_query, response)
            self.logger.end_turn(response)
            return response

        # ── Direct registration routing ──────────────────────────────────
        if self.registration_agent.is_active():
            result = self.registration_agent.process(
                user_query,
                {"user_id": self.current_user_id, "session_id": self.current_session_id},
            )
            self._handle_registration_result(result)
            response = result["message"]
            self._append_history(user_query, response)
            self.logger.end_turn(response)
            return response

        # ── LLM tool-calling loop ────────────────────────────────────────
        messages = [SystemMessage(content=self._system_prompt())]
        messages.extend(self.conversation_history[-6:])
        messages.append(HumanMessage(content=user_query))

        response_text = self._tool_loop(messages, user_query)

        # ── Verification + timing badge ──────────────────────────────────
        self.turn_number += 1
        if not self._is_conversational(user_query) and self.last_rag_context:
            try:
                verification = verify_response(
                    question=user_query,
                    answer=response_text,
                    context=self.last_rag_context,
                    api_key=self.api_key,
                )
                self.evaluator.log_verification(
                    session_id=self.current_session_id,
                    user_id=self.current_user_id,
                    turn_number=self.turn_number,
                    query=user_query,
                    response=response_text,
                    rag_context=self.last_rag_context,
                    verification=verification,
                )
                self.logger.log_step("verification", verification)

                badges = []
                if self.last_retrieval_time_ms > 0:
                    badges.append(f"⏱ Retrieved in {self.last_retrieval_time_ms}ms")
                score = verification.get("score", 0.0)
                if score >= 0.85:
                    badges.append(f"✓ Verified (confidence: {score:.0%})")
                elif score >= 0.7:
                    badges.append(f"⚠️ Moderate confidence ({score:.0%})")
                if badges:
                    response_text += "\n\n" + "  │  ".join(badges)

                self.last_rag_context       = ""
                self.last_retrieval_time_ms = 0
            except Exception as exc:
                self.logger.log_step("verification_error", str(exc))
                if self.last_retrieval_time_ms > 0:
                    response_text += f"\n\n⏱ Retrieved in {self.last_retrieval_time_ms}ms"
        elif self.last_retrieval_time_ms > 0:
            response_text += f"\n\n⏱ Retrieved in {self.last_retrieval_time_ms}ms"

        self.logger.end_turn(response_text)
        return response_text

    # ────────────────────────────────────────────────────────────────────
    # LLM tool loop
    # ────────────────────────────────────────────────────────────────────

    def _tool_loop(self, messages: list, user_query: str) -> str:
        tool_map  = {t.name: t for t in self.tools}
        max_turns = 4

        for _ in range(max_turns):
            try:
                llm_response = self.llm_with_tools.invoke(messages)
                messages.append(llm_response)
                tool_calls = getattr(llm_response, "tool_calls", None) or []

                if not tool_calls:
                    final = llm_response.content
                    self._append_history(user_query, final)
                    return final

                for call in tool_calls:
                    name    = call.get("name")
                    args    = call.get("args", {})
                    call_id = call.get("id")

                    self.logger.log_step("tool_call", {"tool": name, "args": args})
                    print(f"🛠️  {name}({args})")

                    fn = tool_map.get(name)
                    if fn:
                        try:
                            result = fn.invoke(args)
                            try:
                                parsed  = json.loads(result) if isinstance(result, str) else result
                                success = not ("error" in parsed or parsed.get("success") is False)
                                self.evaluator.log_tool_call(self.current_session_id, name, success)
                            except Exception:
                                pass
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)})
                            self.evaluator.log_tool_call(self.current_session_id, name, False)
                    else:
                        result = json.dumps({"error": f"Unknown tool: {name}"})
                        self.evaluator.log_tool_call(self.current_session_id, name, False)

                    print(f"📦  → {str(result)[:200]}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

                    try:
                        parsed = json.loads(result)
                        self._handle_registration_result(parsed)
                    except Exception:
                        pass

            except Exception as exc:
                error_msg = f"Processing error: {exc}"
                print(traceback.format_exc())
                self.logger.log_step("error", error_msg)
                return f"I encountered an error: {error_msg}"

        last_content = getattr(messages[-1], "content", "I need more information to help you.")
        self._append_history(user_query, last_content)
        return last_content

    # ────────────────────────────────────────────────────────────────────
    # Registration result handler
    # ────────────────────────────────────────────────────────────────────

    def _handle_registration_result(self, result: Dict[str, Any]) -> None:
        action = result.get("action")
        data   = result.get("data", {})
        if action == "complete" and data.get("user_id"):
            self.current_user_id = data["user_id"]
            self.guardrail_agent.authenticate_session(
                self.current_session_id, self.current_user_id
            )
            self.logger.log_step("registration_complete", {"user_id": self.current_user_id})

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _load_profile(self) -> Dict:
        if self.current_user_id == "guest":
            return {}
        try:
            from tools.user_profile import get_user_profile
            return get_user_profile(self.current_user_id) or {}
        except Exception:
            return {}

    def _append_history(self, user_query: str, response: str) -> None:
        self.conversation_history.append(HumanMessage(content=user_query))
        self.conversation_history.append(AIMessage(content=response))
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _system_prompt(self) -> str:
        reg_status = (
            "🔴 REGISTRATION IN PROGRESS — route all input to registration_flow"
            if self.registration_agent.is_active()
            else "No active registration"
        )
        auth_status = (
            f"Logged in as {self.current_user_id}"
            if self.current_user_id != "guest"
            else "Guest (not logged in)"
        )
        return f"""You are CoFina, an intelligent financial assistant for young professionals.

SESSION
───────
User         : {auth_status}
Session ID   : {self.current_session_id}
Registration : {reg_status}

TOOLS
─────
registration_flow          — new account creation
login_flow                 — log in an existing user (needs user_id + password)
financial_planning_flow    — ALL financial planning: plans, budgets, goals, PDFs,
                             debt, retirement, investing, car/house buying, savings.
                             This tool classifies intent internally — always call it
                             for financial topics rather than answering directly.
market_research_flow       — product search, price comparison, affordability
monitoring_flow            — alerts, spending tracker, goal progress
get_user_status            — current auth status
get_my_profile             — fetch the logged-in user's full profile
search_financial_documents — educational questions: definitions, concepts, explanations
TIME_TOOLS                 — date/time calculations

ROUTING RULES
─────────────
• DEFINITION / CONCEPT questions ("What is X?", "Explain Y", "How does Z work?")
  → search_financial_documents for grounded answers.
• PLANNING / ACTION questions ("How should I buy a house?", "Help me pay off debt",
  "Generate a plan", "Create a budget") → financial_planning_flow.
  The tool decides internally whether to generate a PDF.
• Never refuse to create a plan or PDF — always call financial_planning_flow.
• LOGIN: ask for User ID and password together; call login_flow when you have both.
• REGISTER: call registration_flow immediately.
• Keep responses focused on financial well-being and actionable next steps.
"""

    def _is_conversational(self, query: str) -> bool:
        patterns = ["hi", "hello", "hey", "thanks", "thank you", "bye",
                    "goodbye", "register", "login", "logout", "status", "help"]
        q = query.lower()
        return any(p in q for p in patterns) or len(query.split()) < 4

    def _guardrail_response(self, result: Dict) -> str:
        actions = result.get("actions", [])
        if "authenticate" in actions:
            return "For personalised advice please log in first. Say 'login' to start."
        if "block" in actions:
            return "I'm unable to process that request. Please ask a financial question."
        return "Let's keep our conversation focused on your financial goals. How can I help?"

    # ── Session management ───────────────────────────────────────────────

    def login(self, user_id: str) -> None:
        self.current_user_id = user_id
        self.guardrail_agent.authenticate_session(self.current_session_id, user_id)

    def logout(self) -> None:
        self.guardrail_agent.end_session(self.current_session_id)
        self.current_user_id      = "guest"
        self.current_session_id   = str(uuid.uuid4())[:8]
        self.conversation_history = []
        self.registration_agent.reset()