"""
Database queries for CoFina - Complete CRUD operations
Extends existing authentication with profile management
"""

import sqlite3
import bcrypt
import json
import os
from datetime import datetime
from typing import Optional, Dict, List, Any

# Path to database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cofina.db")

def get_connection():
    """Establishes connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Add row factory for dict-like access
    return conn

# ==================== EXISTING AUTHENTICATION FUNCTIONS ====================

def user_exists(user_id: str) -> bool:
    """Check if a user exists in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        return cur.fetchone() is not None

def email_exists(email: str) -> bool:
    """Check if email already registered."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        return cur.fetchone() is not None

def get_secret_question(user_id: str) -> str:
    """Retrieve the secret question for a user."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT secret_question FROM users WHERE user_id = ?", (user_id,))
        result = cur.fetchone()
        return result[0] if result else None

def verify_secret_answer(user_id: str, provided_answer: str) -> bool:
    """Verify the secret answer for a user."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT secret_answer_hash FROM users WHERE user_id = ?", (user_id,))
        result = cur.fetchone()
        
        if result and bcrypt.checkpw(provided_answer.lower().strip().encode('utf-8'), result[0]):
            return True
        return False

def register_user(user_id, first_name, other_names, email, password, secret_question, secret_answer):
    """
    Securely registers a new user with complete information.
    Extended from original to include name and email.
    """
    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    ans_hash = bcrypt.hashpw(secret_answer.lower().strip().encode('utf-8'), bcrypt.gensalt())

    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (
                    user_id, first_name, other_names, email, 
                    password_hash, secret_question, secret_answer_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, first_name, other_names, email, pwd_hash, secret_question, ans_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            print(f"Error: Registration failed - {e}")
            return False

def verify_login(user_id, provided_password):
    """Verifies a user's password against the stored hash."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,))
        result = cur.fetchone()
        
        if result and bcrypt.checkpw(provided_password.encode('utf-8'), result[0]):
            # Update last login
            cur.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?", (user_id,))
            conn.commit()
            return True
        return False

def reset_password_with_secret(user_id, provided_answer, new_password):
    """Allows password reset if the secret answer matches."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT secret_answer_hash FROM users WHERE user_id = ?", (user_id,))
        result = cur.fetchone()

        if result and bcrypt.checkpw(provided_answer.lower().strip().encode('utf-8'), result[0]):
            new_pwd_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
            cur.execute("UPDATE users SET password_hash = ? WHERE user_id = ?", (new_pwd_hash, user_id))
            conn.commit()
            return True
        return False

# ==================== NEW: USER PROFILE FUNCTIONS ====================

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get complete user profile including personal info, preferences, and debts."""
    with get_connection() as conn:
        # Basic user info
        cur = conn.execute("""
            SELECT user_id, first_name, other_names, email, created_at, last_login
            FROM users WHERE user_id = ?
        """, (user_id,))
        user = cur.fetchone()
        if not user:
            return None
        
        result = dict(user)
        
        # Profile data
        cur = conn.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,))
        profile = cur.fetchone()
        result['profile'] = dict(profile) if profile else None
        
        # Preferences
        cur = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,))
        prefs = cur.fetchone()
        result['preferences'] = dict(prefs) if prefs else None
        
        # Debts
        cur = conn.execute("""
            SELECT * FROM user_debts 
            WHERE user_id = ? AND status = 'active'
            ORDER BY interest_rate DESC
        """, (user_id,))
        result['debts'] = [dict(row) for row in cur.fetchall()]
        
        # Active plan
        cur = conn.execute("""
            SELECT * FROM financial_plans 
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        plan = cur.fetchone()
        if plan:
            plan_dict = dict(plan)
            # Parse JSON fields
            plan_dict['short_term_goals'] = json.loads(plan_dict['short_term_goals']) if plan_dict['short_term_goals'] else {}
            plan_dict['long_term_goals'] = json.loads(plan_dict['long_term_goals']) if plan_dict['long_term_goals'] else {}
            result['active_plan'] = plan_dict
        else:
            result['active_plan'] = None
        
        return result

def update_user_profile(user_id: str, **kwargs) -> bool:
    """Update user profile information."""
    fields = []
    values = []
    
    for key, value in kwargs.items():
        if value is not None:
            fields.append(f"{key} = ?")
            values.append(value)
    
    if not fields:
        return True
    
    values.append(user_id)
    query = f"""
        INSERT INTO user_profiles (user_id, {', '.join(kwargs.keys())}, updated_at)
        VALUES (?, {', '.join(['?'] * len(kwargs))}, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            {', '.join(fields)},
            updated_at = CURRENT_TIMESTAMP
    """
    
    try:
        with get_connection() as conn:
            conn.execute(query, [user_id] + list(kwargs.values()))
            conn.commit()
            return True
    except Exception as e:
        print(f"Profile update error: {e}")
        return False

# ==================== NEW: USER PREFERENCES FUNCTIONS ====================

def update_user_preferences(user_id: str, **kwargs) -> bool:
    """Update user financial preferences."""
    fields = []
    values = []
    
    for key, value in kwargs.items():
        if value is not None:
            fields.append(f"{key} = ?")
            values.append(value)
    
    if not fields:
        return True
    
    values.append(user_id)
    query = f"""
        INSERT INTO user_preferences (user_id, {', '.join(kwargs.keys())}, updated_at)
        VALUES (?, {', '.join(['?'] * len(kwargs))}, CURRENT_TIMESTAMP)
        ON CONFLICT(user_id) DO UPDATE SET
            {', '.join(fields)},
            updated_at = CURRENT_TIMESTAMP
    """
    
    try:
        with get_connection() as conn:
            conn.execute(query, [user_id] + list(kwargs.values()))
            conn.commit()
            return True
    except Exception as e:
        print(f"Preferences update error: {e}")
        return False

# ==================== NEW: DEBT MANAGEMENT FUNCTIONS ====================

def add_user_debt(user_id: str, debt_data: Dict[str, Any]) -> bool:
    """Add a new debt for user."""
    try:
        with get_connection() as conn:
            conn.execute("""
                INSERT INTO user_debts (
                    user_id, debt_type, creditor, total_amount,
                    remaining_amount, interest_rate, minimum_payment, due_date
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                debt_data.get('debt_type'),
                debt_data.get('creditor'),
                debt_data.get('total_amount'),
                debt_data.get('remaining_amount'),
                debt_data.get('interest_rate'),
                debt_data.get('minimum_payment'),
                debt_data.get('due_date')
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"Add debt error: {e}")
        return False

def get_user_debts(user_id: str) -> List[Dict]:
    """Get all active debts for user."""
    with get_connection() as conn:
        cur = conn.execute("""
            SELECT * FROM user_debts 
            WHERE user_id = ? AND status = 'active'
            ORDER BY interest_rate DESC
        """, (user_id,))
        return [dict(row) for row in cur.fetchall()]

def update_debt_status(debt_id: int, status: str) -> bool:
    """Update debt status (active/paid)."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE user_debts SET status = ? WHERE debt_id = ?",
            (status, debt_id)
        )
        conn.commit()
        return True

# ==================== NEW: FINANCIAL PLANS FUNCTIONS ====================

def create_financial_plan(
    user_id: str,
    plan_name: str,
    short_term_goals: Dict,
    long_term_goals: Dict,
    plan_type: str = "Comprehensive"
) -> bool:
    """Create a new financial plan."""
    try:
        with get_connection() as conn:
            # Archive old active plans
            conn.execute("""
                UPDATE financial_plans 
                SET status = 'archived' 
                WHERE user_id = ? AND status = 'active'
            """, (user_id,))
            
            # Create new plan
            conn.execute("""
                INSERT INTO financial_plans (
                    user_id, plan_name, plan_type,
                    short_term_goals, long_term_goals, status
                ) VALUES (?, ?, ?, ?, ?, 'active')
            """, (
                user_id, plan_name, plan_type,
                json.dumps(short_term_goals),
                json.dumps(long_term_goals)
            ))
            conn.commit()
            return True
    except Exception as e:
        print(f"Create plan error: {e}")
        return False

def get_active_plan(user_id: str) -> Optional[Dict]:
    """Get user's active financial plan."""
    with get_connection() as conn:
        cur = conn.execute("""
            SELECT * FROM financial_plans 
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
        """, (user_id,))
        row = cur.fetchone()
        if row:
            plan = dict(row)
            plan['short_term_goals'] = json.loads(plan['short_term_goals'])
            plan['long_term_goals'] = json.loads(plan['long_term_goals'])
            return plan
        return None

def update_plan_status(plan_id: int, status: str) -> bool:
    """Update plan status."""
    with get_connection() as conn:
        conn.execute(
            "UPDATE financial_plans SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE plan_id = ?",
            (status, plan_id)
        )
        conn.commit()
        return True

# ==================== NEW: TRANSACTION FUNCTIONS ====================

def add_transaction(
    user_id: str,
    amount: float,
    category: str,
    description: str = "",
    is_expense: bool = True
) -> bool:
    """Add a transaction."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO user_transactions (
                user_id, amount, category, description, 
                is_expense, transaction_date
            ) VALUES (?, ?, ?, ?, ?, DATE('now'))
        """, (user_id, amount, category, description, is_expense))
        conn.commit()
        return True

def get_recent_transactions(user_id: str, limit: int = 10) -> List[Dict]:
    """Get recent transactions."""
    with get_connection() as conn:
        cur = conn.execute("""
            SELECT * FROM user_transactions 
            WHERE user_id = ? 
            ORDER BY transaction_date DESC, created_at DESC
            LIMIT ?
        """, (user_id, limit))
        return [dict(row) for row in cur.fetchall()]

# ==================== NEW: AGENT LOGS ====================

def log_agent_decision(
    user_id: str,
    session_id: str,
    decision_type: str,
    summary: str,
    confidence: float
) -> bool:
    """Log agent decision for analysis."""
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO agent_decisions_log (
                user_id, session_id, decision_type, summary, confidence_score
            ) VALUES (?, ?, ?, ?, ?)
        """, (user_id, session_id, decision_type, summary, confidence))
        conn.commit()
        return True

# ==================== NEW: UTILITY FUNCTIONS ====================

def calculate_retirement_date(
    employment_start: str,
    target_age: int,
    current_age: int
) -> str:
    """Calculate estimated retirement date."""
    from datetime import datetime
    
    try:
        start = datetime.strptime(employment_start, "%Y-%m-%d")
        years_to_retirement = target_age - current_age
        # Rough estimate: assume working years + 35 years of savings growth
        retirement_year = start.year + years_to_retirement + 35
        retirement_date = datetime(retirement_year, start.month, start.day)
        return retirement_date.strftime("%Y-%m-%d")
    except:
        return datetime.now().replace(year=datetime.now().year + 30).strftime("%Y-%m-%d")

def get_user_summary(user_id: str) -> Dict[str, Any]:
    """Get quick summary of user's financial health."""
    profile = get_user_profile(user_id)
    
    if not profile:
        return {"error": "User not found"}
    
    # Calculate total debt
    total_debt = sum(d['remaining_amount'] for d in profile.get('debts', []))
    monthly_min = sum(d.get('minimum_payment', 0) for d in profile.get('debts', []))
    
    # Calculate profile completeness
    completeness = 0
    total_fields = 5
    
    if profile.get('profile'): completeness += 1
    if profile.get('preferences'): completeness += 1
    if profile.get('debts'): completeness += 1
    if profile.get('active_plan'): completeness += 1
    if profile.get('profile', {}).get('monthly_income'): completeness += 1
    
    return {
        "user_id": user_id,
        "name": f"{profile.get('first_name', '')} {profile.get('other_names', '')}",
        "email": profile.get('email'),
        "profile_completeness": int((completeness / total_fields) * 100),
        "has_active_plan": profile.get('active_plan') is not None,
        "total_debt": round(total_debt, 2),
        "monthly_minimum": round(monthly_min, 2),
        "debt_count": len(profile.get('debts', []))
    }

def delete_user_data(user_id: str) -> bool:
    """Delete all user data (GDPR compliance)."""
    with get_connection() as conn:
        # Delete in correct order due to foreign keys
        conn.execute("DELETE FROM agent_decisions_log WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_transactions WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM plan_milestones WHERE plan_id IN (SELECT plan_id FROM financial_plans WHERE user_id = ?)", (user_id,))
        conn.execute("DELETE FROM financial_plans WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_debts WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        return True