"""
Agent Planner - Plans the agent's own actions and goals (meta-cognition)
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta, date
from enum import Enum

class AgentGoal(Enum):
    UNDERSTAND_USER = "understand_user"
    GATHER_INFO = "gather_information"
    CREATE_PLAN = "create_financial_plan"
    CHECK_PROGRESS = "check_progress"
    PROVIDE_ADVICE = "provide_advice"
    ESCALATE = "escalate_to_human"

class AgentPlanner:
    """
    Plans the agent's own actions based on current state and goals
    This is meta-planning - the agent planning WHAT to do next
    """
    
    def __init__(self):
        self.current_goal = None
        self.subgoals = []
        self.goal_history = []
    
    def plan_next_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decide what the agent should do next based on current state
        
        Returns: Action plan for the agent
        """
        user_id = state.get("user_id", "guest")
        conversation_turns = state.get("turn_count", 0)
        pending_goals = state.get("pending_goals", [])
        
        # Planning logic - what should agent do next?
        if user_id == "guest" and conversation_turns > 2:
            # Guest user chatting - suggest registration
            return {
                "goal": AgentGoal.GATHER_INFO,
                "action": "suggest_registration",
                "reason": "User is guest and engaged",
                "priority": "medium"
            }
        
        elif pending_goals:
            # User has financial goals - help with them
            return {
                "goal": AgentGoal.CREATE_PLAN,
                "action": "initiate_planning",
                "reason": f"User has {len(pending_goals)} pending goals",
                "priority": "high"
            }
        
        elif conversation_turns % 5 == 0 and conversation_turns > 0:
            # Every 5 turns, check if user needs help
            return {
                "goal": AgentGoal.CHECK_PROGRESS,
                "action": "check_in",
                "reason": "Periodic check-in",
                "priority": "low"
            }
        
        else:
            # Default - continue conversation
            return {
                "goal": AgentGoal.UNDERSTAND_USER,
                "action": "continue_conversation",
                "reason": "No specific trigger",
                "priority": "normal"
            }
    
    def decompose_goal(self, goal: AgentGoal) -> List[Dict[str, Any]]:
        """
        Break down a high-level agent goal into steps
        """
        if goal == AgentGoal.GATHER_INFO:
            return [
                {"step": "ask_user_id", "tool": None},
                {"step": "ask_password", "tool": None},
                {"step": "verify_credentials", "tool": "authenticate_user"},
                {"step": "load_profile", "tool": "get_user_info"}
            ]
        
        elif goal == AgentGoal.CREATE_PLAN:
            return [
                {"step": "get_income", "tool": "query_database"},
                {"step": "get_goals", "tool": "query_database"},
                {"step": "calculate_allocations", "tool": "calculate_budget"},
                {"step": "generate_plan", "tool": "create_financial_plan_tool"},
                {"step": "present_to_user", "tool": None}
            ]
        
        elif goal == AgentGoal.CHECK_PROGRESS:
            return [
                {"step": "fetch_goals", "tool": "query_database"},
                {"step": "calculate_progress", "tool": "calculate_goal_progress"},
                {"step": "format_update", "tool": None},
                {"step": "present_to_user", "tool": None}
            ]
        
        else:  # UNDERSTAND_USER
            return [
                {"step": "analyze_query", "tool": None},
                {"step": "check_intent", "tool": None},
                {"step": "formulate_response", "tool": None}
            ]
    
    def should_escalate(self, state: Dict[str, Any]) -> bool:
        """Decide if agent should escalate to human"""
        failed_attempts = state.get("failed_tool_calls", 0)
        user_frustration = state.get("user_frustration_score", 0)
        
        if failed_attempts > 3:
            return True
        if user_frustration > 0.8:
            return True
        return False
    
    def update_from_feedback(self, success: bool, goal: AgentGoal):
        """Learn from previous actions"""
        self.goal_history.append({
            "goal": goal.value,
            "success": success,
            "timestamp": datetime.now().isoformat()
        })