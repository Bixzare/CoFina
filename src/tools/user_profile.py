import sqlite3
import json
import os

# Path logic
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "cofina.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def get_user_profile(user_id: str):
    """Fetches the full context for a user including preferences and plans."""
    with get_connection() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        
        cur.execute("""
            SELECT u.user_id, p.risk_profile, p.debt_strategy, p.savings_priority
            FROM users u
            LEFT JOIN user_preferences p ON u.user_id = p.user_id
            WHERE u.user_id = ?
        """, (user_id,))
        
        profile = cur.fetchone()
        if not profile:
            return None
            
        cur.execute("""
            SELECT plan_name, short_term_goals, long_term_goals, status
            FROM financial_plans
            WHERE user_id = ? AND status = 'active'
        """, (user_id,))
        
        plan = cur.fetchone()
        
        return {
            "user_id": profile["user_id"],
            "preferences": {
                "risk_profile": profile["risk_profile"],
                "debt_strategy": profile["debt_strategy"],
                "savings_priority": profile["savings_priority"]
            },
            "active_plan": {
                "name": plan["plan_name"],
                "short_term": json.loads(plan["short_term_goals"]) if plan else {},
                "long_term": json.loads(plan["long_term_goals"]) if plan else {}
            } if plan else None
        }

def update_user_preferences(user_id: str, risk_profile: str, debt_strategy: str, savings_priority: str):
    """Updates or sets the user's financial personality."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO user_preferences (user_id, risk_profile, debt_strategy, savings_priority, last_updated)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                risk_profile=excluded.risk_profile,
                debt_strategy=excluded.debt_strategy,
                savings_priority=excluded.savings_priority,
                last_updated=CURRENT_TIMESTAMP
        """, (user_id, risk_profile, debt_strategy, savings_priority))
        conn.commit()
        return True

def create_financial_plan(user_id: str, plan_name: str, short_goals: dict, long_goals: dict):
    """Archives old plans and sets a new active plan."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("UPDATE financial_plans SET status = 'archived' WHERE user_id = ? AND status = 'active'", (user_id,))
        cur.execute("""
            INSERT INTO financial_plans (user_id, plan_name, short_term_goals, long_term_goals, status)
            VALUES (?, ?, ?, ?, 'active')
        """, (user_id, plan_name, json.dumps(short_goals), json.dumps(long_goals)))
        conn.commit()
        return True