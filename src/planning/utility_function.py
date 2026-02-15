"""
Utility Function - Quantifies preferences for decision-making
"""

from typing import Dict, Any, List
import numpy as np

class UtilityFunction:
    """
    Minimal utility function for financial decisions
    Utility = Goal_Progress + Savings_Stability - Overspending_Risk - Liquidity_Risk
    """
    
    def __init__(self):
        self.weights = {
            "goal_progress": 0.4,
            "savings_stability": 0.3,
            "overspending_risk": 0.2,
            "liquidity_risk": 0.1
        }
    
    def calculate(self, state: Dict[str, Any]) -> float:
        """Calculate utility of current state"""
        utility = 0.0
        
        # Goal progress (0-1)
        goal_progress = self._calculate_goal_progress(state)
        utility += self.weights["goal_progress"] * goal_progress
        
        # Savings stability (0-1)
        savings_stability = self._calculate_savings_stability(state)
        utility += self.weights["savings_stability"] * savings_stability
        
        # Overspending risk (0-1, lower is better)
        overspending_risk = self._calculate_overspending_risk(state)
        utility -= self.weights["overspending_risk"] * overspending_risk
        
        # Liquidity risk (0-1, lower is better)
        liquidity_risk = self._calculate_liquidity_risk(state)
        utility -= self.weights["liquidity_risk"] * liquidity_risk
        
        return max(0.0, min(1.0, utility))
    
    def compare_actions(self, action_a: Dict, action_b: Dict, state: Dict) -> str:
        """Compare two actions and return the better one"""
        # Simulate outcome of each action
        utility_a = self._simulate_action(action_a, state)
        utility_b = self._simulate_action(action_b, state)
        
        if utility_a > utility_b:
            return "A"
        elif utility_b > utility_a:
            return "B"
        else:
            return "EQUAL"
    
    def _calculate_goal_progress(self, state: Dict) -> float:
        """Calculate average progress toward goals"""
        goals = state.get("goals", [])
        if not goals:
            return 0.5  # Neutral if no goals
        
        progress = []
        for goal in goals:
            target = goal.get("target", 1)
            current = goal.get("current", 0)
            if target > 0:
                progress.append(min(1.0, current / target))
        
        return np.mean(progress) if progress else 0.5
    
    def _calculate_savings_stability(self, state: Dict) -> float:
        """Calculate stability of savings (low volatility)"""
        savings = state.get("savings_history", [])
        if len(savings) < 2:
            return 0.7
        
        # Lower coefficient of variation = more stable
        mean_savings = np.mean(savings)
        if mean_savings == 0:
            return 0.5
        
        cv = np.std(savings) / mean_savings
        return max(0.0, 1.0 - min(1.0, cv))
    
    def _calculate_overspending_risk(self, state: Dict) -> float:
        """Calculate risk of overspending"""
        budget = state.get("budget", {})
        actual = state.get("actual_spending", {})
        
        if not budget or not actual:
            return 0.3
        
        overages = []
        for category, planned in budget.items():
            spent = actual.get(category, 0)
            if planned > 0:
                overage = max(0, (spent - planned) / planned)
                overages.append(min(1.0, overage))
        
        return np.mean(overages) if overages else 0.0
    
    def _calculate_liquidity_risk(self, state: Dict) -> float:
        """Calculate risk of being unable to meet short-term obligations"""
        cash = state.get("cash_on_hand", 0)
        monthly_expenses = state.get("monthly_expenses", 1)
        
        if monthly_expenses <= 0:
            return 0.0
        
        months_of_expenses = cash / monthly_expenses
        
        # Risk is high if less than 3 months of expenses
        if months_of_expenses >= 6:
            return 0.0
        elif months_of_expenses >= 3:
            return 0.3
        elif months_of_expenses >= 1:
            return 0.7
        else:
            return 1.0
    
    def _simulate_action(self, action: Dict, state: Dict) -> float:
        """Simulate outcome of an action (simplified)"""
        # Copy state and apply action effects
        new_state = state.copy()
        
        action_type = action.get("type", "")
        if action_type == "save":
            new_state["savings"] = state.get("savings", 0) + action.get("amount", 0)
        elif action_type == "spend":
            new_state["savings"] = state.get("savings", 0) - action.get("amount", 0)
        elif action_type == "invest":
            new_state["investments"] = state.get("investments", 0) + action.get("amount", 0)
        
        return self.calculate(new_state)