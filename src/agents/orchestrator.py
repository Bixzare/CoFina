"""
Orchestrator Agent - Main LLM-driven coordinator for CoFina.

Key design decisions
────────────────────
1. When a registration session is active (registration_agent.is_active()),
   user input is routed DIRECTLY to the registration agent — the LLM is NOT
   involved in mid-registration turns.  This prevents the LLM from accidentally
   routing to another tool and breaking the step sequence.

2. A single `registration_flow` tool is exposed to the LLM for starting
   registration.  Once started, the direct-routing logic above takes over.

3. After successful registration the orchestrator updates current_user_id and
   authenticates the guardrail session automatically.
"""

from __future__ import annotations

import json
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
from core.checkpoint import CheckpointManager
from core.memory_manager import MemoryManager
from core.state_manager import SessionPhase, StateManager
from tools.dateTime import TIME_TOOLS
from utils.logger import AgentLogger


class CoFinaOrchestrator:
    """
    Main orchestrator.  Exposes specialised sub-agents as LLM tools.
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.logger = AgentLogger()

        # ── Core infrastructure ──────────────────────────────────────────
        self.state_manager = StateManager()
        self.memory_manager = MemoryManager()
        self.checkpoint_manager = CheckpointManager()

        # ── Sub-agents (long-lived, stateful) ────────────────────────────
        self.registration_agent = RegistrationAgent()
        self.financial_planner = FinancialPlannerAgent()
        self.market_agent = MarketAgent()
        self.monitor_agent = MonitorAgent()
        self.summarizer_agent = SummarizerAgent(api_key)
        self.guardrail_agent = GuardrailAgent()

        # ── RAG ──────────────────────────────────────────────────────────
        self.retriever = None
        try:
            from RAG.index import ensure_index
            from RAG.retriever import create_retriever
            vector_store = ensure_index(api_key)
            self.retriever = create_retriever(vector_store)
            self.financial_planner.rag_retriever = self.retriever
            print("✅ RAG system ready")
        except Exception as exc:
            self.logger.log_step("warning", f"RAG init skipped: {exc}")
            print(f"⚠️  RAG unavailable: {exc}")

        # ── LLM ──────────────────────────────────────────────────────────
        self.llm = ChatOpenAI(
            model="gemini-2.5-flash",
            temperature=0,
            api_key=api_key,
            base_url="https://ai-gateway.andrew.cmu.edu/",
        )
        self.tools = self._build_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # ── Session state ────────────────────────────────────────────────
        self.current_session_id: str = str(uuid.uuid4())[:8]
        self.current_user_id: str = "guest"
        self.conversation_history: List = []

    # ────────────────────────────────────────────────────────────────────
    # Tool definitions
    # ────────────────────────────────────────────────────────────────────

    def _build_tools(self) -> list:

        @tool
        def registration_flow(user_input: str) -> str:
            """
            Start or continue user registration.  Use whenever the user wants
            to create a new account, register, or sign up.
            """
            result = self.registration_agent.process(
                user_input,
                {"user_id": self.current_user_id, "session_id": self.current_session_id},
            )
            self._handle_registration_result(result)
            return json.dumps(result)

        @tool
        def financial_planning_flow(user_input: str) -> str:
            """
            Financial planning, budgeting, savings goals, investment advice,
            and plan creation / adjustment.
            """
            profile = self._load_profile()
            result = self.financial_planner.process(
                query=user_input,
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def market_research_flow(user_input: str) -> str:
            """
            Product searches, price comparisons, and affordability checks.
            """
            profile = self._load_profile()
            result = self.market_agent.process(
                query=user_input,
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def monitoring_flow(user_input: str) -> str:
            """
            Financial alerts, spending tracking, and goal progress monitoring.
            """
            profile = self._load_profile()
            result = self.monitor_agent.process(
                user_id=self.current_user_id,
                context={"user_profile": profile, "session_id": self.current_session_id},
            )
            return json.dumps(result)

        @tool
        def login_flow(user_id: str, password: str) -> str:
            """
            Log an existing user in with their user_id and password.
            Use whenever the user says they have an account and want to log in.
            If you don't have the password yet, ask for it before calling this tool.
            After calling this tool, relay the message field directly to the user.
            """
            from db.queries import verify_login, user_exists, get_user_profile
            if not user_exists(user_id):
                return json.dumps({
                    "success": False,
                    "message": (
                        f"⚠️  No account found for '{user_id}'. "
                        "Please check your User ID or say 'register' to create a new account."
                    )
                })
            if verify_login(user_id, password):
                self.current_user_id = user_id
                self.guardrail_agent.authenticate_session(self.current_session_id, user_id)
                self.logger.log_step("login", {"user_id": user_id})
                # Pull first name for a personal greeting
                profile = get_user_profile(user_id) or {}
                first_name = profile.get("first_name", user_id)
                return json.dumps({
                    "success": True,
                    "message": (
                        f"✅ Welcome back, {first_name}! You're now logged in.\n"
                        "What would you like to do today? I can help with your "
                        "financial plan, debt tracking, savings goals, and more."
                    )
                })
            return json.dumps({
                "success": False,
                "message": (
                    "❌ Incorrect password. Please try again, or say "
                    "'forgot password' if you'd like to reset it."
                )
            })

        @tool
        def get_user_status() -> str:
            """Return the current user's authentication status."""
            return json.dumps({
                "user_id": self.current_user_id,
                "authenticated": self.current_user_id != "guest",
                "session_id": self.current_session_id,
            })

        @tool
        def get_my_profile() -> str:
            """
            Fetch and return the full profile of the currently logged-in user.
            Use when the user asks what information CoFina has about them,
            or wants to see their profile, plan, debts, or preferences.
            """
            if self.current_user_id == "guest":
                return json.dumps({"error": "Not logged in. Please log in first."})
            profile = self._load_profile()
            if not profile:
                return json.dumps({"error": "Profile not found."})
            return json.dumps(profile)

        @tool
        def search_financial_documents(query: str) -> str:
            """Search the financial knowledge base for concepts and guidance."""
            if not self.retriever:
                return json.dumps({"error": "Knowledge base not available"})
            docs = self.retriever.invoke(query)
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

        # ── Intercept logout before hitting the LLM ──────────────────────
        _lc = user_query.lower().strip()
        if any(kw in _lc for kw in ("logout", "log out", "log off", "sign out")):
            self.logout()
            response = "✅ You've been logged out. See you next time!"
            self._append_history(user_query, response)
            self.logger.end_turn(response)
            return response

        # ── Direct registration routing ──────────────────────────────────
        # When registration is in progress, bypass the LLM completely.
        # This guarantees the step sequence is never broken by tool mis-routing.
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
        self.logger.end_turn(response_text)
        return response_text

    # ────────────────────────────────────────────────────────────────────
    # LLM tool loop
    # ────────────────────────────────────────────────────────────────────

    def _tool_loop(self, messages: list, user_query: str) -> str:
        tool_map = {t.name: t for t in self.tools}
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
                    name = call.get("name")
                    args = call.get("args", {})
                    call_id = call.get("id")

                    self.logger.log_step("tool_call", {"tool": name, "args": args})
                    print(f"🛠️  {name}({args})")

                    fn = tool_map.get(name)
                    if fn:
                        try:
                            result = fn.invoke(args)
                        except Exception as exc:
                            result = json.dumps({"error": str(exc)})
                    else:
                        result = json.dumps({"error": f"Unknown tool: {name}"})

                    print(f"📦  → {str(result)[:200]}")
                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

                    # If registration just completed, update state
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

        # Max turns exhausted
        last_content = getattr(messages[-1], "content", "I need more information to help you.")
        self._append_history(user_query, last_content)
        return last_content

    # ────────────────────────────────────────────────────────────────────
    # Registration result handler
    # ────────────────────────────────────────────────────────────────────

    def _handle_registration_result(self, result: Dict[str, Any]) -> None:
        """
        After any registration_agent.process() call, inspect the result and
        update orchestrator state (user_id, guardrail session) if appropriate.
        """
        action = result.get("action")
        data = result.get("data", {})

        if action == "complete" and data.get("user_id"):
            self.current_user_id = data["user_id"]
            self.guardrail_agent.authenticate_session(
                self.current_session_id, self.current_user_id
            )
            self.logger.log_step("registration_complete", {
                "user_id": self.current_user_id,
            })

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
        auth_status = f"Logged in as {self.current_user_id}" if self.current_user_id != "guest" else "Guest (not logged in)"
        return f"""You are CoFina, an intelligent financial assistant for young professionals.

SESSION
───────
User         : {auth_status}
Session ID   : {self.current_session_id}
Registration : {reg_status}

TOOLS
─────
registration_flow          — new account creation (register / sign up)
login_flow                 — log in an EXISTING user (needs user_id + password)
financial_planning_flow    — budgets, plans, savings goals, investments
market_research_flow       — product search, price comparison, affordability
monitoring_flow            — alerts, spending tracker, goal progress
get_user_status            — current auth status
get_my_profile             — fetch full profile of the logged-in user
search_financial_documents — search the financial knowledge base
TIME_TOOLS                 — date/time calculations

RULES
─────
• Be concise, direct, and actionable.
• Use the appropriate tool; do not guess or make up data.
• After tool results, explain them clearly in plain English.
• LOGIN: if user says they have an account / want to log in:
    1. Ask for their User ID and password in ONE message (e.g. "Please enter your User ID and password").
    2. The user may provide them together (e.g. "niliya, mypass" or "user: niliya pass: mypass").
    3. As soon as you have BOTH user_id AND password, call login_flow immediately — do NOT wait.
    4. Relay the returned "message" field word-for-word to the user.
• PROFILE: if user asks what you know about them, call get_my_profile.
• REGISTER: if user wants a new account, call registration_flow immediately.
• FINANCIAL PLAN: if user is not logged in and asks for a personalised plan,
  tell them to log in first, then offer to help.
• Keep responses focused on financial well-being.
"""

    def _guardrail_response(self, result: Dict) -> str:
        actions = result.get("actions", [])
        if "authenticate" in actions:
            return "For personalised financial advice please log in first. Say 'login' to start."
        if "block" in actions:
            return "I'm unable to process that request. Please ask a financial question."
        return "Let's keep our conversation focused on your financial goals. How can I help?"

    # ── Session management ───────────────────────────────────────────────

    def login(self, user_id: str) -> None:
        self.current_user_id = user_id
        self.guardrail_agent.authenticate_session(self.current_session_id, user_id)

    def logout(self) -> None:
        self.guardrail_agent.end_session(self.current_session_id)
        self.current_user_id = "guest"
        self.current_session_id = str(uuid.uuid4())[:8]
        self.conversation_history = []
        self.registration_agent.reset()
