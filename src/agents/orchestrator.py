"""
Orchestrator Agent - Main coordinator using tool-based architecture
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json
import traceback

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool

from agents.registration_agent import RegistrationAgent
from agents.financial_planner import FinancialPlannerAgent
from agents.market_agent import MarketAgent
from agents.monitor_agent import MonitorAgent
from agents.summarizer_agent import SummarizerAgent
from agents.guardrail_agent import GuardrailAgent
from tools.dateTime import TIME_TOOLS
from tools.calendar_tools import CALENDAR_TOOLS

from core.state_manager import StateManager, SessionPhase
from core.memory_manager import MemoryManager
from core.adaptive_control import AdaptiveController
from core.evaluation import EvaluationMetrics
from core.checkpoint import CheckpointManager

from utils.logger import AgentLogger
from RAG.retriever import create_retriever

class CoFinaOrchestrator:
    """
    Main orchestrator that exposes specialized agents as tools to the LLM
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.logger = AgentLogger()
        
        # Initialize state management
        self.state_manager = StateManager()
        self.memory_manager = MemoryManager()
        self.adaptive_controller = AdaptiveController()
        self.evaluator = EvaluationMetrics()
        self.checkpoint_manager = CheckpointManager()
        
        # Initialize specialized agents
        self.registration_agent = RegistrationAgent()
        self.financial_planner = FinancialPlannerAgent()
        self.market_agent = MarketAgent()
        self.monitor_agent = MonitorAgent()
        self.summarizer_agent = SummarizerAgent(api_key)
        self.guardrail_agent = GuardrailAgent()
        
        # Initialize RAG
        try:
            from RAG.index import create_vector_store
            from RAG.index import ensure_index
            self.vector_store = ensure_index(api_key)
            from RAG.retriever import create_retriever
            self.retriever = create_retriever(self.vector_store)
            self.financial_planner.rag_retriever = self.retriever
            print("✅ RAG system ready")
            self.vector_store = create_vector_store(api_key)
            self.retriever = create_retriever(self.vector_store)
            self.financial_planner.rag_retriever = self.retriever
        except Exception as e:
            self.logger.log_step("error", f"RAG init failed: {e}")
            print(f"⚠️ RAG init warning: {e}")
            self.retriever = None
        
        # Initialize LLM with tools
        self.llm = ChatOpenAI(
            model="gemini-2.5-flash",  # or "gemini-2.5-flash" depending on your API
            temperature=0,
            api_key=api_key,
            base_url='https://ai-gateway.andrew.cmu.edu/'
        )
        
        # Bind tools (specialized agents as tools + utility tools)
        self.tools = self._create_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Session tracking
        self.current_session_id = str(uuid.uuid4())[:8]
        self.current_user_id = "guest"
        self.conversation_history = []  # Store message history
    
    def _create_tools(self):
        """Create tools for the LLM to use"""
        
        @tool
        def register_user_flow(user_input: str) -> str:
            """Use this tool when the user wants to register, login, or manage their account.
            Handles user registration, authentication, and password management."""
            result = self.registration_agent.process(user_input, {
                "user_id": self.current_user_id,
                "session_id": self.current_session_id
            })
            
            # Update user ID if registration successful
            if result.get("action") == "success" and result.get("data", {}).get("user_id"):
                self.current_user_id = result["data"]["user_id"]
                self.guardrail_agent.authenticate_session(
                    self.current_session_id, self.current_user_id
                )
            
            return json.dumps(result)
        @tool
        def complete_registration_flow(user_input: str) -> str:
            """Complete multi-step registration and profile setup. Use for new user registration."""
            result = self.registration_agent.process(user_input, {
                "user_id": self.current_user_id,
                "session_id": self.current_session_id
            })

            # Handle completion
            if result.get("action") == "complete":
                data = result.get("data", {})
                self.current_user_id = data.get("user_id", self.current_user_id)
                self.guardrail_agent.authenticate_session(
                    self.current_session_id, self.current_user_id
                )

                # Show retirement date if available
                if data.get("retirement_date"):
                    result["message"] += f"\n📅 Estimated retirement date: {data['retirement_date']}"
            return json.dumps(result)
   
        @tool
        def financial_planning_flow(user_input: str) -> str:
            """Use this tool for financial planning, budgeting, savings goals, and investment advice.
            Handles creating plans, adjusting plans, and financial analysis."""
            # Get user profile if authenticated
            profile = {}
            if self.current_user_id != "guest":
                from tools.user_profile import get_user_profile
                profile = get_user_profile(self.current_user_id) or {}
            
            result = self.financial_planner.process(
                query=user_input,
                user_id=self.current_user_id,
                context={
                    "user_profile": profile,
                    "session_id": self.current_session_id
                }
            )
            return json.dumps(result)
        
        @tool
        def market_research_flow(user_input: str) -> str:
            """Use this tool when the user wants to search for products, compare prices, or check if they can afford something.
            Handles product searches, price comparisons, and affordability checks."""
            # Get user profile for affordability check
            profile = {}
            if self.current_user_id != "guest":
                from tools.user_profile import get_user_profile
                profile = get_user_profile(self.current_user_id) or {}
            
            result = self.market_agent.process(
                query=user_input,
                user_id=self.current_user_id,
                context={
                    "user_profile": profile,
                    "session_id": self.current_session_id
                }
            )
            return json.dumps(result)
        
        @tool
        def monitoring_flow(user_input: str) -> str:
            """Use this tool for checking financial alerts, tracking spending, and monitoring goals.
            Handles alerts, reminders, and progress tracking."""
            # Get user profile
            profile = {}
            if self.current_user_id != "guest":
                from tools.user_profile import get_user_profile
                profile = get_user_profile(self.current_user_id) or {}
            
            result = self.monitor_agent.process(
                user_id=self.current_user_id,
                context={
                    "user_profile": profile,
                    "session_id": self.current_session_id
                }
            )
            return json.dumps(result)
        
        @tool
        def get_user_status() -> str:
            """Get the current user's authentication status and basic info."""
            status = {
                "user_id": self.current_user_id,
                "authenticated": self.current_user_id != "guest",
                "session_id": self.current_session_id
            }
            return json.dumps(status)
        
        @tool
        def search_financial_documents(query: str) -> str:
            """Search financial knowledge base for information about financial concepts."""
            if not self.retriever:
                return json.dumps({"error": "Knowledge base not available"})
            
            docs = self.retriever.invoke(query)
            results = [{"content": doc.page_content, "source": doc.metadata.get("source", "unknown")} 
                      for doc in docs[:3]]
            return json.dumps(results)
        
        return [
            register_user_flow,
            financial_planning_flow,
            market_research_flow,
            monitoring_flow,
            get_user_status,
            search_financial_documents,
            complete_registration_flow
        ]
    
    def process(self, user_query: str) -> str:
        """
        Process user query through tool-based LLM orchestration
        """
        self.logger.start_turn(user_query)
        
        # Guardrail check first
        guardrail_result = self.guardrail_agent.process(
            query=user_query,
            session_id=self.current_session_id,
            user_id=self.current_user_id
        )
        
        if not guardrail_result["passed"]:
            response = self._handle_guardrail_failure(guardrail_result)
            self.logger.end_turn(response)
            return response
        
        # Build message history
        messages = [
            SystemMessage(content=self._get_system_prompt()),
        ]
        
        # Add conversation history (last 6 messages for context)
        messages.extend(self.conversation_history[-6:])
        
        # Add current user message
        messages.append(HumanMessage(content=user_query))
        
        # Tool-calling loop
        max_tool_turns = 4
        for turn in range(max_tool_turns):
            try:
                # Invoke LLM with tools
                response = self.llm_with_tools.invoke(messages)
                messages.append(response)
                
                # Check if there are tool calls
                tool_calls = getattr(response, "tool_calls", None) or []
                
                if not tool_calls:
                    # No tool calls, this is the final response
                    final_answer = response.content
                    self.conversation_history.append(HumanMessage(content=user_query))
                    self.conversation_history.append(AIMessage(content=final_answer))
                    
                    # Trim history if needed
                    if len(self.conversation_history) > 20:
                        self.conversation_history = self.conversation_history[-20:]
                    
                    self.logger.end_turn(final_answer)
                    return final_answer
                
                # Execute tool calls
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("args", {})
                    tool_id = tool_call.get("id")
                    
                    self.logger.log_step("tool_call", {
                        "tool": tool_name,
                        "args": tool_args
                    })
                    
                    # Find and execute the tool
                    tool_result = None
                    for tool in self.tools:
                        if tool.name == tool_name:
                            try:
                                tool_result = tool.invoke(tool_args)
                            except Exception as e:
                                tool_result = json.dumps({"error": str(e), "traceback": traceback.format_exc()})
                            break
                    
                    if tool_result is None:
                        tool_result = json.dumps({"error": f"Tool {tool_name} not found"})
                    
                    # Add tool result to messages
                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_id
                    ))
                    
                    # Log for debugging
                    print(f"🛠️ Tool call → {tool_name}({tool_args})")
                    print(f"📦 Tool result → {tool_result[:200]}...")
                    
            except Exception as e:
                error_msg = f"Error in processing: {str(e)}"
                print(traceback.format_exc())
                self.logger.log_step("error", error_msg)
                return f"I encountered an error: {error_msg}"
        
        # If we hit max tool turns, return the last response
        final = response.content if 'response' in locals() else "I need more information to help you."
        self.conversation_history.append(HumanMessage(content=user_query))
        self.conversation_history.append(AIMessage(content=final))
        self.logger.end_turn(final)
        return final
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the LLM"""
        return f"""You are CoFina, an intelligent financial assistant for young professionals.

RULES:
- Keep responses concise when possible
- Use bullet points for lists
- No lengthy explanations
- Be direct and actionable

CURRENT SESSION:
- User ID: {self.current_user_id}
- Authenticated: {self.current_user_id != "guest"}
- Session ID: {self.current_session_id}

AVAILABLE TOOLS:
1. register_user_flow - Use for registration, login, account management
2. financial_planning_flow - Use for creating plans, budgeting, savings goals
3. market_research_flow - Use for product searches, price comparisons
4. monitoring_flow - Use for checking alerts, tracking progress
5. get_user_status - Check current user authentication status
6. search_financial_documents - Search for financial concepts and information

INSTRUCTIONS:
1. Be friendly, helpful, and conversational
2. Use the appropriate tool based on what the user wants
3. If the user needs to register or log in for personalized help, guide them through it
4. After getting tool results, explain them clearly to the user
5. Keep responses concise but informative
6. If you need more information, ask follow-up questions

Remember: You're helping young professionals manage their finances better!
"""
    
    def _handle_guardrail_failure(self, guardrail_result: Dict) -> str:
        """Handle guardrail failures"""
        if "authenticate" in guardrail_result.get("actions", []):
            return "For personalized financial advice, please log in first. You can say 'login' to authenticate."
        elif "block" in guardrail_result.get("actions", []):
            return "I'm unable to process that request. Please ask a financial question."
        else:
            return "I need to ensure our conversation stays focused on financial planning. How can I help with your finances?"
    
    def login(self, user_id: str):
        """Log in a user"""
        self.current_user_id = user_id
        self.guardrail_agent.authenticate_session(self.current_session_id, user_id)
    
    def logout(self):
        """Log out current user"""
        self.guardrail_agent.end_session(self.current_session_id)
        self.current_user_id = "guest"
        self.current_session_id = str(uuid.uuid4())[:8]
        self.conversation_history = []