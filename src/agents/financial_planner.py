"""
Financial Planner Agent - Concise financial planning and analysis
"""

from typing import Dict, Any, List, Optional
import json
from datetime import datetime

from tools.user_profile import get_user_profile, update_user_preferences
from tools.generatePlan import create_financial_plan_pdf
from tools.financial_calculator import calculate_simple_interest, calculate_compound_interest
from RAG.retriever import format_docs
from utils.cache import get_cache

class FinancialPlannerAgent:
    """
    Specialized agent for financial planning with concise responses
    """
    
    def __init__(self, rag_retriever=None):
        self.rag_retriever = rag_retriever
        self.planning_context = {}
        self.cache = get_cache()
    
    def process(self, query: str, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process financial planning request with concise responses"""
        intent = self._detect_intent(query, context)
        
        if intent == "create_plan":
            return self._create_plan(query, user_id, context)
        elif intent == "calculate_interest":
            return self._calculate_interest(query)
        elif intent == "car_plan":
            return self._car_buying_plan(query)
        else:
            return self._quick_advice(query, user_id)
    
    def _detect_intent(self, query: str, context: Dict) -> str:
        """Detect financial planning intent"""
        q = query.lower()
        
        if any(kw in q for kw in ["car", "vehicle", "auto"]):
            return "car_plan"
        elif any(kw in q for kw in ["interest", "compound", "simple"]):
            return "calculate_interest"
        elif any(kw in q for kw in ["plan", "budget", "goal"]):
            return "create_plan"
        else:
            return "advice"
    
    def _calculate_interest(self, query: str) -> Dict[str, Any]:
        """Calculate simple/compound interest"""
        # Extract numbers: principal, rate, time
        import re
        numbers = re.findall(r"(\d+\.?\d*)", query)
        
        if len(numbers) >= 3:
            p = float(numbers[0])
            r = float(numbers[1])
            t = float(numbers[2])
            
            result = calculate_simple_interest(p, r, t)
            return {
                "action": "calculation",
                "message": f"Simple interest: ${result['interest_earned']} | Total: ${result['total_amount']}",
                "data": result
            }
        
        return {
            "action": "calculation",
            "message": "Please provide: principal, rate%, and years. Example: '1000 principal, 5% rate, 2 years'",
            "data": {}
        }
    
    def _car_buying_plan(self, query: str) -> Dict[str, Any]:
        """Concise car buying plan"""
        return {
            "action": "plan",
            "message": """🚗 **Car Buying Plan**

**1. Budget:** 20% down payment + monthly payment ≤15% income
**2. Research:** Toyota Corolla ($22k-$28k new), good fuel economy
**3. Financing:** Compare rates (credit unions often best)
**4. Total Cost:** Include insurance ($100-200/mo) + maintenance

Need specific numbers? Tell me your monthly income and down payment.""",
            "data": {"topic": "car_buying"}
        }
    
    def _quick_advice(self, query: str, user_id: str) -> Dict[str, Any]:
        """Quick, concise financial advice"""
        
        # Simple interest calculation from query
        if "interest" in query.lower():
            import re
            nums = re.findall(r"(\d+\.?\d*)", query)
            if len(nums) >= 3:
                p, r, t = float(nums[0]), float(nums[1]), float(nums[2])
                interest = p * (r/100) * t
                return {
                    "action": "advice",
                    "message": f"Interest: ${interest:.2f} | Total: ${p + interest:.2f}",
                    "data": {}
                }
        
        # Generic concise advice by category
        q = query.lower()
        
        if "budget" in q:
            msg = "**50/30/20 Rule:** 50% Needs, 30% Wants, 20% Savings"
        elif "save" in q or "saving" in q:
            msg = "**Save:** Automate transfers on payday. Start with 20% of income."
        elif "invest" in q:
            msg = "**Invest:** Low-cost index funds, start early, stay consistent."
        elif "debt" in q:
            msg = "**Debt:** Pay highest interest first (avalanche) or smallest balances first (snowball)."
        elif "emergency" in q:
            msg = "**Emergency Fund:** 3-6 months of expenses in high-yield savings."
        elif "retirement" in q:
            msg = "**Retirement:** Save 15% of income. Use 401k match + Roth IRA."
        else:
            msg = "**Track spending → Build emergency fund → Pay debt → Invest**"
        
        return {
            "action": "advice",
            "message": msg,
            "data": {}
        }
    
    def _create_plan(self, query: str, user_id: str, context: Dict) -> Dict[str, Any]:
        """Create a new financial plan"""
        profile = get_user_profile(user_id)
        
        if not profile:
            return {
                "action": "error",
                "message": "Please complete your profile first.",
                "data": {}
            }
        
        return {
            "action": "plan_created",
            "message": "✅ Plan ready! Check your 'financial_plans' folder for PDF.",
            "data": {"profile": profile}
        }