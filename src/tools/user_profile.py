"""
User Profile Tools - Complete user data management
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Optional, List, Any

# Path logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "cofina.db")

def get_connection():
    """Get database connection with row factory"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# ==================== PROFILE MANAGEMENT ====================

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get complete user profile"""
    from db.queries import get_user_profile as db_get_profile
    return db_get_profile(user_id)

def update_user_profile(user_id: str, **kwargs) -> bool:
    """Update user profile information."""
    if not kwargs:
        return True
    
    # Filter out None values
    valid_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    
    if not valid_kwargs:
        return True
    
    with get_connection() as conn:
        try:
            # Check if profile exists
            cur = conn.execute("SELECT 1 FROM user_profiles WHERE user_id = ?", (user_id,))
            exists = cur.fetchone() is not None
            
            if exists:
                # Update existing profile
                set_clause = ", ".join([f"{k} = ?" for k in valid_kwargs.keys()])
                values = list(valid_kwargs.values()) + [user_id]
                query = f"UPDATE user_profiles SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?"
                conn.execute(query, values)
            else:
                # Insert new profile
                columns = ", ".join(['user_id'] + list(valid_kwargs.keys()))
                placeholders = ", ".join(['?'] + ['?'] * len(valid_kwargs))
                values = [user_id] + list(valid_kwargs.values())
                query = f"INSERT INTO user_profiles ({columns}) VALUES ({placeholders})"
                conn.execute(query, values)
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Profile update error: {e}")
            return False

def update_user_preferences(
    user_id: str,
    risk_profile: str = None,
    debt_strategy: str = None,
    savings_priority: str = None,
    investment_horizon: str = None
) -> bool:
    """Update user financial preferences."""
    # Build dict of provided values
    prefs = {}
    if risk_profile is not None:
        prefs['risk_profile'] = risk_profile
    if debt_strategy is not None:
        prefs['debt_strategy'] = debt_strategy
    if savings_priority is not None:
        prefs['savings_priority'] = savings_priority
    if investment_horizon is not None:
        prefs['investment_horizon'] = investment_horizon
    
    if not prefs:
        return True
    
    with get_connection() as conn:
        try:
            # Check if preferences exist
            cur = conn.execute("SELECT 1 FROM user_preferences WHERE user_id = ?", (user_id,))
            exists = cur.fetchone() is not None
            
            if exists:
                # Update existing preferences
                set_clause = ", ".join([f"{k} = ?" for k in prefs.keys()])
                values = list(prefs.values()) + [user_id]
                query = f"UPDATE user_preferences SET {set_clause}, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?"
                conn.execute(query, values)
            else:
                # Insert new preferences
                columns = ", ".join(['user_id'] + list(prefs.keys()))
                placeholders = ", ".join(['?'] + ['?'] * len(prefs))
                values = [user_id] + list(prefs.values())
                query = f"INSERT INTO user_preferences ({columns}) VALUES ({placeholders})"
                conn.execute(query, values)
            
            conn.commit()
            return True
        except Exception as e:
            print(f"Preferences update error: {e}")
            return False

# ==================== DEBT MANAGEMENT ====================

def add_user_debt(user_id: str, debt_data: Dict[str, Any]) -> bool:
    """Add a new debt for user"""
    from db.queries import add_user_debt as db_add_debt
    return db_add_debt(user_id, debt_data)

def get_user_debts(user_id: str) -> List[Dict]:
    """Get all active debts for user"""
    from db.queries import get_user_debts
    return get_user_debts(user_id)

def calculate_total_debt(user_id: str) -> Dict[str, float]:
    """Calculate total debt statistics"""
    debts = get_user_debts(user_id)
    
    total = sum(d['remaining_amount'] for d in debts)
    monthly_min = sum(d['minimum_payment'] for d in debts if d.get('minimum_payment'))
    weighted_rate = sum(d['remaining_amount'] * d['interest_rate'] for d in debts) / total if total > 0 else 0
    
    return {
        "total_debt": round(total, 2),
        "monthly_minimum": round(monthly_min, 2),
        "weighted_interest_rate": round(weighted_rate, 2),
        "debt_count": len(debts)
    }

# ==================== FINANCIAL PLANS ====================

def create_financial_plan(
    user_id: str,
    plan_name: str,
    short_goals: dict,
    long_goals: dict,
    plan_type: str = "Comprehensive"
) -> bool:
    """Create a new financial plan"""
    from db.queries import create_financial_plan as db_create_plan
    return db_create_plan(user_id, plan_name, short_goals, long_goals, plan_type)

def get_active_plan(user_id: str) -> Optional[Dict]:
    """Get user's active financial plan"""
    from db.queries import get_active_plan
    return get_active_plan(user_id)

# ==================== GOAL MANAGEMENT ====================

def add_user_goal(
    user_id: str,
    goal_type: str,
    goal_name: str,
    target_amount: float,
    target_date: str = None,
    priority: int = 1
) -> int:
    """Add a new financial goal"""
    with get_connection() as conn:
        cur = conn.execute("""
            INSERT INTO user_goals (user_id, goal_type, goal_name, target_amount, target_date, priority)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (user_id, goal_type, goal_name, target_amount, target_date, priority))
        conn.commit()
        return cur.lastrowid

def update_goal_progress(user_id: str, goal_id: int, current_amount: float) -> bool:
    """Update progress toward a goal"""
    with get_connection() as conn:
        conn.execute("""
            UPDATE user_goals
            SET current_amount = ?
            WHERE user_id = ? AND goal_id = ?
        """, (current_amount, user_id, goal_id))
        conn.commit()
        return True

# ==================== TRANSACTIONS ====================

def add_transaction(
    user_id: str,
    amount: float,
    category: str,
    description: str = "",
    is_expense: bool = True
) -> bool:
    """Add a transaction"""
    from db.queries import add_transaction
    return add_transaction(user_id, amount, category, description, is_expense)

def get_recent_transactions(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent transactions"""
    from db.queries import get_recent_transactions
    return get_recent_transactions(user_id, limit)

# ==================== SUMMARY & UTILITY ====================

def get_user_summary(user_id: str) -> Dict[str, Any]:
    """Get comprehensive user summary"""
    profile = get_user_profile(user_id)
    
    if not profile:
        return {"error": "User not found"}
    
    debt_stats = calculate_total_debt(user_id)
    
    summary = {
        "user_id": user_id,
        "name": f"{profile.get('first_name', '')} {profile.get('other_names', '')}",
        "email": profile.get('email'),
        "profile_completeness": 0,
        "has_active_plan": profile.get('active_plan') is not None,
        "debt_summary": debt_stats,
        "financial_health_score": 0
    }
    
    # Calculate completeness
    completeness = 0
    total_fields = 7
    
    if profile.get('profile'): completeness += 1
    if profile.get('preferences'): completeness += 1
    if debt_stats['debt_count'] > 0: completeness += 1
    if profile.get('active_plan'): completeness += 1
    if profile.get('profile', {}).get('monthly_income'): completeness += 1
    if profile.get('profile', {}).get('age'): completeness += 1
    if profile.get('profile', {}).get('retirement_age_target'): completeness += 1
    
    summary['profile_completeness'] = int((completeness / total_fields) * 100)
    
    # Calculate financial health score
    score = 50  # Base
    
    if debt_stats['total_debt'] == 0:
        score += 20
    elif debt_stats['total_debt'] < profile.get('profile', {}).get('annual_income', 0) * 0.3:
        score += 10
    
    if profile.get('profile', {}).get('monthly_income', 0) > 3000:
        score += 10
    
    if summary['has_active_plan']:
        score += 15
    
    summary['financial_health_score'] = min(100, score)
    
    return summary

def calculate_retirement_date(
    employment_start: str,
    target_age: int,
    current_age: int
) -> str:
    """Calculate estimated retirement date"""
    from db.queries import calculate_retirement_date
    return calculate_retirement_date(employment_start, target_age, current_age)

def delete_user_data(user_id: str) -> bool:
    """Delete all user data"""
    from db.queries import delete_user_data
    return delete_user_data(user_id)