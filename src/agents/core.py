# src/agent/core.py

from typing import Optional, Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
    AIMessage,
)
from langchain.tools import tool
import re
import json

# Internal imports
from RAG.index import create_vector_store
from RAG.retriever import create_retriever, format_docs
from agents.verifier import verify_response
from tools.user_profile import get_user_profile, update_user_preferences, create_financial_plan
from tools.searchProducts import search_products
from tools.dateTime import TIME_TOOLS
from tools.generatePlan import generate_financial_plan_pdf, create_financial_plan_pdf  # New imports
from utils.logger import AgentLogger
from db.queries import register_user, verify_login, reset_password_with_secret, get_secret_question


# ------------------------------------------------------------------
# Tool wrappers (LLM-callable)
# ------------------------------------------------------------------

@tool
def get_user_info(user_id: str) -> Dict[str, Any]:
    """Fetch user financial profile."""
    profile = get_user_profile(user_id)
    if not profile:
        return {"status": "not_found", "message": f"No profile found for {user_id}"}
    return {"status": "ok", "profile": profile}


@tool
def check_user_exists(user_id: str) -> Dict[str, Any]:
    """Check if a user exists in the system."""
    from db.queries import user_exists
    exists = user_exists(user_id)
    if exists:
        question = get_secret_question(user_id)
        return {"exists": True, "secret_question": question}
    return {"exists": False}


@tool
def verify_user_secret(user_id: str, secret_answer: str) -> Dict[str, Any]:
    """Verify user's secret answer for authentication."""
    from db.queries import verify_secret_answer
    verified = verify_secret_answer(user_id, secret_answer)
    return {"verified": verified}


@tool
def save_user_preference(
    user_id: str,
    risk_profile: str = "Not changed",
    debt_strategy: str = "Not changed",
    savings_priority: str = "Not changed",
):
    """Update user preferences."""
    return update_user_preferences(
        user_id,
        risk_profile,
        debt_strategy,
        savings_priority,
    )


@tool
def create_financial_plan_tool(
    user_id: str,
    plan_name: str,
    short_term_goals: str,
    long_term_goals: str
) -> Dict[str, Any]:
    """Create a financial plan for the user AND automatically generate PDF."""
    # First create the plan
    success = create_financial_plan(user_id, plan_name, short_term_goals, long_term_goals)
    
    result = {"success": success, "user_id": user_id}
    
    if success:
        # Automatically generate PDF after successful plan creation
        try:
            # Get user profile
            profile = get_user_profile(user_id)
            if profile:
                # Generate PDF
                pdf_result = create_financial_plan_pdf(
                    user_id=user_id,
                    profile_data=profile,
                    short_term_goals=short_term_goals,
                    long_term_goals=long_term_goals,
                    plan_name=plan_name
                )
                
                # Add PDF info to result
                if pdf_result.get("success"):
                    result["pdf_generated"] = True
                    result["pdf_filename"] = pdf_result.get("filename")
                    result["pdf_filepath"] = pdf_result.get("filepath")
                    result["message"] = f"Financial plan created and PDF generated: {pdf_result.get('filename')}"
                else:
                    result["pdf_generated"] = False
                    result["pdf_error"] = pdf_result.get("error", "Unknown error")
                    result["message"] = "Financial plan created but PDF generation failed"
            else:
                result["pdf_generated"] = False
                result["message"] = "Financial plan created but could not generate PDF (profile not found)"
                
        except Exception as e:
            result["pdf_generated"] = False
            result["pdf_error"] = str(e)
            result["message"] = f"Financial plan created but PDF generation failed: {str(e)}"
    
    return result


@tool
def generate_pdf_plan(
    user_id: str,
    profile_data: str,
    short_term_goals: str,
    long_term_goals: str,
    plan_name: str = "Financial Plan"
) -> Dict[str, Any]:
    """Generate a PDF financial plan document."""
    return generate_financial_plan_pdf(
        user_id=user_id,
        profile_data=profile_data,
        short_term_goals=short_term_goals,
        long_term_goals=long_term_goals,
        plan_name=plan_name
    )


@tool
def register_new_user(
    user_id: str,
    password: str,
    secret_question: str,
    secret_answer: str,
):
    """Register a new user. All parameters are required."""
    success = register_user(user_id, password, secret_question, secret_answer)
    return {"success": success, "user_id": user_id if success else None}


@tool
def authenticate_user(user_id: str, password: str):
    """Authenticate a user with password."""
    success = verify_login(user_id, password)
    return {"authenticated": success}


@tool
def reset_password(
    user_id: str,
    secret_answer: str,
    new_password: str,
):
    """Reset password using secret answer."""
    success = reset_password_with_secret(
        user_id, secret_answer, new_password
    )
    return {"success": success}


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------

class CoFinaAgent:
    """
    Intent-driven financial agent with RAG verification.
    """

    def __init__(self, api_key: str, logger: Optional[AgentLogger] = None):
        self.logger = logger or AgentLogger()
        self.api_key = api_key
        self.conversation_history = []  # Track conversation for context
        self.registration_context = {}  # Track registration info across messages
        self.new_user_setup_context = {}  # Track new user setup info
        self.last_registered_user = None  # Track last successfully registered user
        self.setup_step = None  # Track current setup step

        self.llm = ChatOpenAI(
            model="gemini-2.5-flash",
            api_key=api_key,
            base_url="https://ai-gateway.andrew.cmu.edu/",
            temperature=0.2,
        )

        # Register toolsana
        self.tools = [
            get_user_info,
            check_user_exists,
            verify_user_secret,
            save_user_preference,
            create_financial_plan_tool,  # This now includes PDF generation
            generate_pdf_plan,  # Still available for manual use
            register_new_user,
            authenticate_user,
            reset_password,
            search_products,
            *TIME_TOOLS,
        ]

        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Optional RAG
        try:
            self.vector_store = create_vector_store(api_key)
            self.retriever = create_retriever(self.vector_store)
        except Exception as e:
            self.logger.log_step("error", f"RAG init failed: {e}")
            self.retriever = None

    # --------------------------------------------------------------

    def retrieve_context(self, query: str) -> str:
        """Retrieve relevant context from RAG."""
        if not self.retriever:
            return ""
        docs = self.retriever.invoke(query)
        self.logger.log_retrieval(query, docs)
        return format_docs(docs)

    # --------------------------------------------------------------

    def _is_conversational_query(self, query: str) -> bool:
        """Check if query is conversational (greetings, registration, etc)."""
        conversational_patterns = [
            "hi", "hello", "hey", "good morning", "good afternoon",
            "thanks", "thank you", "bye", "goodbye"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in conversational_patterns)

    # --------------------------------------------------------------

    def _is_time_query(self, query: str) -> bool:
        """Check if query is about date/time (skip RAG verification)."""
        time_patterns = [
            "time", "date", "today", "tomorrow", "yesterday",
            "when", "how long", "age", "birthday", "deadline",
            "quarter", "year", "month", "week", "day",
            "calendar", "schedule", "deadline", "due date"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in time_patterns)

    # --------------------------------------------------------------

    def _is_new_user_setup_query(self, query: str) -> bool:
        """Check if query is part of new user setup."""
        setup_patterns = [
            "risk", "profile", "preference", "debt", "savings", "investment",
            "goals", "plan", "financial", "budget", "retirement", "emergency"
        ]
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in setup_patterns)

    # --------------------------------------------------------------

    def _extract_registration_info(self, query: str) -> Dict[str, str]:
        """Extract registration information from user query."""
        info = {}
        query_lower = query.lower()
        
        # Extract user ID patterns
        id_patterns = [
            r"user[_\s]?id[:\s]+['\"]?(\w+)['\"]?",  # user id: niliya
            r"id[:\s]+['\"]?(\w+)['\"]?",  # id: niliya
            r"my (?:chosen|desired|preferred) (?:user )?id (?:is|as)?['\"]?(\w+)['\"]?",  # my id is niliya
            r"i want to use (\w+) as my id",  # I want to use niliya as my ID
            r"let's use (\w+)",  # let's use niliya
        ]
        
        # Extract password patterns
        password_patterns = [
            r"password[:\s]+['\"]?([^'\"]+)['\"]?",  # password: 4321Open
            r"pass[:\s]+['\"]?([^'\"]+)['\"]?",  # pass: 4321Open
            r"my password (?:is|as)?['\"]?([^'\"]+)['\"]?",  # my password is 4321Open
        ]
        
        # Extract security question patterns
        question_patterns = [
            r"security question[:\s]+['\"]?([^'\"]+?)['\"]?(?=answer|$)",  # security question: What is...
            r"secret question[:\s]+['\"]?([^'\"]+?)['\"]?(?=answer|$)",  # secret question: What is...
            r"question[:\s]+['\"]?([^'\"]+?)['\"]?(?=answer|$)",  # question: What is...
        ]
        
        # Extract security answer patterns
        answer_patterns = [
            r"answer[:\s]+['\"]?([^'\"]+)['\"]?$",  # answer: CMUAfrica
            r"secret answer[:\s]+['\"]?([^'\"]+)['\"]?$",  # secret answer: CMUAfrica
            r"security answer[:\s]+['\"]?([^'\"]+)['\"]?$",  # security answer: CMUAfrica
        ]
        
        # Try to extract each piece
        for pattern in id_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["user_id"] = match.group(1)
                break
        
        for pattern in password_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["password"] = match.group(1)
                break
        
        for pattern in question_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["secret_question"] = match.group(1).strip()
                break
        
        for pattern in answer_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["secret_answer"] = match.group(1).strip()
                break
        
        return info

    # --------------------------------------------------------------

    def _extract_financial_preferences(self, query: str) -> Dict[str, str]:
        """Extract financial preferences from user query."""
        info = {}
        query_lower = query.lower()
        
        # Extract risk profile
        risk_patterns = [
            r"risk (?:profile|tolerance)[:\s]+['\"]?(\w+)['\"]?",  # risk profile: moderate
            r"(?:i am|i'm) (low|medium|high|moderate|conservative|aggressive) risk",  # I am moderate risk
            r"my risk (?:is|as)?['\"]?(low|medium|high|moderate|conservative|aggressive)['\"]?",  # my risk is moderate
        ]
        
        # Extract debt strategy
        debt_patterns = [
            r"debt (?:strategy|approach)[:\s]+['\"]?(\w+)['\"]?",  # debt strategy: snowball
            r"(?:i prefer|i use|i follow) (snowball|avalanche|consolidation) method",  # I prefer snowball
            r"debt[:\s]+['\"]?(snowball|avalanche|consolidation)['\"]?",  # debt: snowball
        ]
        
        # Extract savings priority
        savings_patterns = [
            r"savings (?:priority|goal)[:\s]+['\"]?([^'\"]+)['\"]?",  # savings priority: retirement
            r"(?:i want|i need|my priority is) to save for['\"]?([^'\"]+)['\"]?",  # I want to save for retirement
            r"(?:emergency|retirement|education|house|car|travel) fund",  # emergency fund
        ]
        
        # Try to extract each piece
        for pattern in risk_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                risk_map = {
                    "low": "Low", "conservative": "Low",
                    "medium": "Moderate", "moderate": "Moderate",
                    "high": "High", "aggressive": "High"
                }
                info["risk_profile"] = risk_map.get(match.group(1).lower(), match.group(1).title())
                break
        
        for pattern in debt_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["debt_strategy"] = match.group(1).title()
                break
        
        for pattern in savings_patterns:
            match = re.search(pattern, query_lower, re.IGNORECASE)
            if match:
                info["savings_priority"] = match.group(1).title()
                break
        
        return info

    # --------------------------------------------------------------

    def _has_all_registration_info(self) -> bool:
        """Check if we have all 4 pieces of registration info."""
        required = {"user_id", "password", "secret_question", "secret_answer"}
        return all(key in self.registration_context for key in required)

    # --------------------------------------------------------------

    def _has_basic_financial_preferences(self) -> bool:
        """Check if we have basic financial preferences."""
        return all(key in self.new_user_setup_context for key in ["risk_profile", "debt_strategy", "savings_priority"])

    # --------------------------------------------------------------

    def _build_registration_summary(self) -> str:
        """Build summary of collected registration info."""
        if not self.registration_context:
            return "No registration info collected yet."
        
        summary = "COLLECTED REGISTRATION INFO:\n"
        for key, value in self.registration_context.items():
            summary += f"- {key}: {value}\n"
        
        missing = {"user_id", "password", "secret_question", "secret_answer"} - set(self.registration_context.keys())
        if missing:
            summary += f"\nSTILL NEED:\n"
            for item in missing:
                summary += f"- {item}\n"
        
        return summary

    # --------------------------------------------------------------

    def _build_new_user_setup_summary(self) -> str:
        """Build summary of collected financial preferences."""
        if not self.new_user_setup_context:
            return "No financial preferences collected yet."
        
        summary = "COLLECTED FINANCIAL PREFERENCES:\n"
        for key, value in self.new_user_setup_context.items():
            summary += f"- {key}: {value}\n"
        
        return summary

    # --------------------------------------------------------------

    def run(self, user_query: str, user_id: str = "guest") -> str:
        """
        Intelligent agent loop with smart RAG verification.
        """

        self.logger.start_turn(user_query)

        # Check if this is from a newly registered user who needs setup
        is_new_user_setup = False
        if self.last_registered_user and user_id == "guest":
            # User just registered but hasn't logged in yet
            is_new_user_setup = True

        # Update registration context with any new info from this query
        extracted_info = self._extract_registration_info(user_query)
        if extracted_info:
            self.registration_context.update(extracted_info)
            self.logger.log_step("registration_info", {
                "extracted": extracted_info,
                "current_context": self.registration_context
            })

        # Check if this is a conversational/time query (skip RAG verification)
        skip_verification = (
            self._is_conversational_query(user_query) or 
            self._is_time_query(user_query)
        )

        # Check if this is part of registration flow
        is_registration_flow = any(
            term in user_query.lower() for term in [
                "register", "sign up", "create account", "new account",
                "user id", "password", "secret question", "security question"
            ]
        ) or bool(self.registration_context)

        # Check if this is part of new user setup
        if not is_registration_flow and (is_new_user_setup or self._is_new_user_setup_query(user_query)):
            is_new_user_setup = True
            
            # Determine which setup step we're on
            if not self.setup_step:
                self.setup_step = "risk_profile"
            
            # Extract based on current step
            if self.setup_step == "risk_profile":
                risk_info = self._extract_financial_preferences(user_query)
                if risk_info.get("risk_profile"):
                    self.new_user_setup_context.update(risk_info)
                    self.setup_step = "debt_strategy"
            elif self.setup_step == "debt_strategy":
                debt_info = self._extract_financial_preferences(user_query)
                if debt_info.get("debt_strategy"):
                    self.new_user_setup_context.update(debt_info)
                    self.setup_step = "savings_priority"
            elif self.setup_step == "savings_priority":
                savings_info = self._extract_financial_preferences(user_query)
                if savings_info.get("savings_priority"):
                    self.new_user_setup_context.update(savings_info)
                    self.setup_step = "short_term_goals"
            elif self.setup_step == "short_term_goals":
                if "short" in user_query.lower() or "term" in user_query.lower():
                    self.new_user_setup_context["short_term_goals"] = user_query
                    self.setup_step = "long_term_goals"
            elif self.setup_step == "long_term_goals":
                if "long" in user_query.lower() or "term" in user_query.lower():
                    self.new_user_setup_context["long_term_goals"] = user_query
                    self.setup_step = "plan_name"

        # Retrieve RAG context (skip for time/registration/setup queries)
        rag_context = ""
        if not self._is_time_query(user_query) and not is_registration_flow and not is_new_user_setup:
            rag_context = self.retrieve_context(user_query)

        # Get user profile if authenticated
        user_context = ""
        if user_id != "guest":
            profile = get_user_profile(user_id)
            if profile:
                user_context = f"\n\nUSER PROFILE DATA:\n{profile}"

        # Combined context for both agent and verifier
        full_context = f"{rag_context}{user_context}"

        # Build registration context string for system prompt
        reg_summary = ""
        if is_registration_flow:
            reg_summary = self._build_registration_summary()
        
        # Build new user setup context string
        setup_summary = ""
        current_step_instruction = ""
        if is_new_user_setup:
            setup_summary = self._build_new_user_setup_summary()
            
            # Add current step instruction
            step_instructions = {
                "risk_profile": "ASK USER: What's your investment risk tolerance? (Low/Moderate/High)",
                "debt_strategy": "ASK USER: How do you prefer to handle debt? (Snowball/Avalanche/Consolidation)",
                "savings_priority": "ASK USER: What's your top savings priority? (Emergency Fund/Retirement/Education/House Downpayment/etc)",
                "short_term_goals": "ASK USER: What are your SHORT-TERM financial goals? (1-2 years) Examples: 'Save $5000 emergency fund', 'Pay off credit card debt', 'Save for vacation'",
                "long_term_goals": "ASK USER: What are your LONG-TERM financial goals? (5+ years) Examples: 'Buy a house', 'Save for retirement', 'Start a business'",
                "plan_name": "ASK USER: What would you like to name your financial plan? (Default: 'My Financial Plan')"
            }
            
            if self.setup_step in step_instructions:
                current_step_instruction = f"\nCURRENT SETUP STEP: {step_instructions[self.setup_step]}"

        # Create setup mode status string safely
        setup_mode_status = "INACTIVE"
        if is_new_user_setup and self.setup_step:
            setup_mode_status = f"ACTIVE - Step {self.setup_step}"
        elif is_new_user_setup:
            setup_mode_status = "ACTIVE"

        system_prompt = f"""
You are CoFina, an intelligent conversational financial agent.

CORE BEHAVIOR:
- Be friendly, helpful, and ACTION-ORIENTED
- Understand user intent and TAKE ACTION using tools
- Don't keep asking questions - USE THE TOOLS when you have enough information
- Provide personalized financial advice when user context is available

SESSION INFO:
- Current user_id: {user_id}
- Status: {"Authenticated" if user_id != "guest" else "Guest (unauthenticated)"}
- New User Setup Mode: {setup_mode_status}

CRITICAL REGISTRATION INSTRUCTIONS:
1. When user wants to register, guide them to provide: user_id, password, secret_question, secret_answer
2. COLLECT INFORMATION ACROSS MULTIPLE MESSAGES. Users may provide info piece by piece.
3. Use the check_user_exists tool to verify if a user_id is available BEFORE attempting registration.
4. When you have ALL 4 pieces, call register_new_user IMMEDIATELY.
5. After successful registration, START NEW USER FINANCIAL SETUP WORKFLOW.

NEW USER FINANCIAL SETUP WORKFLOW (STEP-BY-STEP):
After registration, guide user through these steps ONE AT A TIME:

STEP 1: Risk Profile
Ask: "What's your investment risk tolerance? (Low/Moderate/High)"
→ User answers → Store in new_user_setup_context

STEP 2: Debt Strategy  
Ask: "How do you prefer to handle debt? (Snowball/Avalanche/Consolidation)"
→ User answers → Store in new_user_setup_context

STEP 3: Savings Priority
Ask: "What's your top savings priority? (Emergency Fund/Retirement/Education/House Downpayment/etc)"
→ User answers → Store in new_user_setup_context

STEP 4: Short-term Goals (SEPARATE QUESTION)
Ask: "What are your SHORT-TERM financial goals? (1-2 years) Examples: 'Save $5000 emergency fund', 'Pay off credit card debt', 'Save for vacation'"
→ User answers → Store in new_user_setup_context

STEP 5: Long-term Goals (SEPARATE QUESTION)  
Ask: "What are your LONG-TERM financial goals? (5+ years) Examples: 'Buy a house', 'Save for retirement', 'Start a business'"
→ User answers → Store in new_user_setup_context

STEP 6: Plan Name (Optional)
Ask: "What would you like to name your financial plan? (Default: 'My Financial Plan')"

STEP 7: Save Preferences
When you have risk_profile, debt_strategy, savings_priority → Call save_user_preference(user_id, risk, debt, savings)

STEP 8: Create Financial Plan AND Generate PDF (AUTOMATIC)
When you have short_term_goals AND long_term_goals AND plan_name → Call create_financial_plan_tool(user_id, plan_name, short_term_goals, long_term_goals)

IMPORTANT: The create_financial_plan_tool AUTOMATICALLY generates a PDF after creating the plan. No separate PDF generation step needed!

STEP 9: Final Message
After plan creation, check the tool result. If it contains "pdf_generated": true, tell user:
"✅ Setup complete! Your financial plan has been created and a PDF has been generated at: allPlans/[filename].pdf"
If PDF generation failed, still acknowledge plan creation.

CURRENT REGISTRATION CONTEXT:
{reg_summary}

CURRENT FINANCIAL PREFERENCES CONTEXT:
{setup_summary}
{current_step_instruction}

TIME & DATE HANDLING:
You now have time/date tools! Use them when users ask about:
- Current time/date: Use get_current_time()
- Date differences: Use get_date_difference(start_date, end_date)
- Adding to dates: Use add_to_date(date, days, weeks, months, years)
- Date information: Use get_day_info(date)
- Age calculation: Use calculate_age(birth_date)
- Financial dates: Use get_financial_dates()
- Compound interest: Use calculate_compounding(principal, rate, years)

DATE FORMAT: Use YYYY-MM-DD format unless specified otherwise.

AUTHENTICATION WORKFLOW:
1. User wants personalized advice → ask for User ID
2. Call check_user_exists(user_id)
3. If exists → ask secret question → call verify_user_secret(user_id, answer)
4. If verified → call get_user_info(user_id) → provide personalized advice

TOOLS USAGE:
- check_user_exists: Check if user_id exists
- verify_user_secret: Verify secret answer
- get_user_info: Fetch profile AFTER verification
- register_new_user: Register user (needs: user_id, password, secret_question, secret_answer)
- save_user_preference: Update preferences (risk_profile, debt_strategy, savings_priority)
- create_financial_plan_tool: Create financial plan AND generate PDF (user_id, plan_name, short_term_goals, long_term_goals)
- generate_pdf_plan: Generate PDF financial plan manually (user_id, profile_data, short_term_goals, long_term_goals, plan_name)
- search_products: Product search
- TIME TOOLS: get_current_time, get_date_difference, add_to_date, get_day_info, 
              calculate_age, get_financial_dates, calculate_compounding

FINANCIAL ADVICE:
- Use RAG context for general knowledge
- Use user profile for personalized advice
- Use time tools for date-related financial calculations
- Be helpful and provide actionable guidance

CONTEXT:
{full_context if full_context.strip() else "No specific context - use general knowledge."}
"""

        messages = [
            SystemMessage(content=system_prompt),
        ]
        
        # Add conversation history for context (last 3 messages)
        if hasattr(self, 'previous_messages'):
            messages.extend(self.previous_messages[-3:])
        
        messages.append(HumanMessage(content=user_query))

        # First LLM reasoning step
        response = self.llm_with_tools.invoke(messages)

        # ----------------------------------------------------------
        # Tool execution loop
        # ----------------------------------------------------------
        if response.tool_calls:
            tool_messages = []
            tool_map = {tool.name: tool for tool in self.tools}

            for call in response.tool_calls:
                tool_name = call["name"]
                tool_args = call["args"]

                self.logger.log_step("tool_call", {
                    "tool": tool_name,
                    "args": tool_args
                })

                tool_fn = tool_map[tool_name]
                tool_result = tool_fn.invoke(tool_args)

                tool_messages.append(
                    ToolMessage(
                        tool_call_id=call["id"],
                        content=str(tool_result),
                    )
                )

                # Handle successful registration
                if tool_name == "register_new_user" and tool_result.get("success"):
                    self.last_registered_user = tool_args.get("user_id")
                    self.registration_context = {}  # Clear registration context
                    # Start new user setup context
                    self.new_user_setup_context = {"user_id": tool_args.get("user_id")}
                    self.setup_step = "risk_profile"
                    self.logger.log_step("setup_start", {
                        "user_id": tool_args.get("user_id"),
                        "step": "risk_profile"
                    })
                
                # Handle successful preference save
                if tool_name == "save_user_preference":
                    # Mark that preferences are set
                    self.new_user_setup_context["preferences_set"] = True
                    self.logger.log_step("preferences_saved", {
                        "user_id": tool_args.get("user_id"),
                        "risk_profile": tool_args.get("risk_profile"),
                        "debt_strategy": tool_args.get("debt_strategy"),
                        "savings_priority": tool_args.get("savings_priority")
                    })
                
                # Handle successful plan creation (which now includes PDF generation)
                if tool_name == "create_financial_plan_tool":
                    if tool_result.get("success"):
                        # Mark that plan is created
                        self.new_user_setup_context["plan_created"] = True
                        
                        # Setup is complete
                        self.setup_step = None
                        self.last_registered_user = None
                        
                        self.logger.log_step("plan_created", {
                            "user_id": tool_args.get("user_id"),
                            "plan_name": tool_args.get("plan_name"),
                            "pdf_generated": tool_result.get("pdf_generated", False),
                            "pdf_filename": tool_result.get("pdf_filename")
                        })

            # Send tool results back to LLM
            messages.append(response)
            messages.extend(tool_messages)

            final_response = self.llm_with_tools.invoke(messages)
            final_answer = final_response.content

        else:
            final_answer = response.content

        # ----------------------------------------------------------
        # RAG Verification (skip for time/registration/setup queries)
        # ----------------------------------------------------------
        if not skip_verification and not is_registration_flow and not is_new_user_setup and rag_context and self.retriever:
            try:
                verification = verify_response(
                    question=user_query,
                    answer=final_answer,
                    context=full_context if full_context.strip() else "General knowledge response",
                    api_key=self.api_key
                )

                self.logger.log_step("verification", verification)

                score = verification.get("score", 0.0)
                action = verification.get("action", "accept")

                # Only add verification badge if score is meaningful
                if action == "accept" and score >= 0.7:
                    final_answer += f"\n Grounded: (Confidence: {score:.1%})"
                elif action == "retry" and score >= 0.5:
                    final_answer += f"\n Partially Grounded: (Confidence: {score:.1%})"
                
            except Exception as e:
                self.logger.log_step("verification_error", str(e))

        # Store conversation history
        if not hasattr(self, 'previous_messages'):
            self.previous_messages = []
        self.previous_messages.append(HumanMessage(content=user_query))
        self.previous_messages.append(AIMessage(content=final_answer))
        
        # Keep only last 6 messages (3 exchanges)
        if len(self.previous_messages) > 6:
            self.previous_messages = self.previous_messages[-6:]

        self.logger.end_turn(final_answer)
        return final_answer


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        exit(1)

    agent = CoFinaAgent(api_key)

    print("\n" + "-" * 110)
    print("🤖 CoFina Agent — Intelligent Financial Assistant (type 'exit' to quit)")
    print("-" * 110 + "\n")

    user_id = "guest"

    while True:
        try:
            question = input("Question 👤: ").strip()
            if question.lower() in ["exit", "quit", "end"]:
                break
            if not question:
                continue

            response = agent.run(question, user_id)
            print(f"\n🤖 Response:\n{response}\n")

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}\n")