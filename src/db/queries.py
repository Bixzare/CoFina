"""
Database queries for CoFina — complete CRUD layer over cofina.db (SQLite).

All public functions return plain Python types (bool, str, dict, list).
Callers should never need to import sqlite3 directly.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

import bcrypt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cofina.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ═══════════════════════════════════════════════════════════════════
# Authentication
# ═══════════════════════════════════════════════════════════════════

def user_exists(user_id: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
        return cur.fetchone() is not None


def email_exists(email: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute("SELECT 1 FROM users WHERE email = ?", (email,))
        return cur.fetchone() is not None


def register_user(
    user_id: str,
    first_name: str,
    last_name: str,          # stored in `other_names` column
    email: str,
    password: str,
    secret_question: str,
    secret_answer: str,
) -> bool:
    """
    Insert a new user row.  Passwords and secret answers are bcrypt-hashed.
    Returns True on success, False if the user_id or email already exists.
    """
    pwd_hash = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
    ans_hash = bcrypt.hashpw(
        secret_answer.lower().strip().encode(), bcrypt.gensalt()
    )
    with get_connection() as conn:
        try:
            conn.execute(
                """
                INSERT INTO users
                    (user_id, first_name, other_names, email,
                     password_hash, secret_question, secret_answer_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, first_name, last_name, email,
                 pwd_hash, secret_question, ans_hash),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError as exc:
            print(f"[register_user] IntegrityError: {exc}")
            return False


def verify_login(user_id: str, password: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT password_hash FROM users WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        if row and bcrypt.checkpw(password.encode(), row[0]):
            conn.execute(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE user_id = ?",
                (user_id,),
            )
            conn.commit()
            return True
        return False


def get_secret_question(user_id: str) -> Optional[str]:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT secret_question FROM users WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        return row[0] if row else None


def verify_secret_answer(user_id: str, provided_answer: str) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT secret_answer_hash FROM users WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        if row:
            return bcrypt.checkpw(
                provided_answer.lower().strip().encode(), row[0]
            )
        return False


def reset_password_with_secret(
    user_id: str, provided_answer: str, new_password: str
) -> bool:
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT secret_answer_hash FROM users WHERE user_id = ?", (user_id,)
        )
        row = cur.fetchone()
        if row and bcrypt.checkpw(
            provided_answer.lower().strip().encode(), row[0]
        ):
            new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt())
            conn.execute(
                "UPDATE users SET password_hash = ? WHERE user_id = ?",
                (new_hash, user_id),
            )
            conn.commit()
            return True
        return False


# ═══════════════════════════════════════════════════════════════════
# User profile
# ═══════════════════════════════════════════════════════════════════

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Return a fully joined profile dict or None if user not found."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT user_id, first_name, other_names, email, created_at, last_login
            FROM users WHERE user_id = ?
            """,
            (user_id,),
        )
        user_row = cur.fetchone()
        if not user_row:
            return None

        result: Dict[str, Any] = dict(user_row)

        cur = conn.execute(
            "SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)
        )
        p = cur.fetchone()
        result["profile"] = dict(p) if p else None

        cur = conn.execute(
            "SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)
        )
        pr = cur.fetchone()
        result["preferences"] = dict(pr) if pr else None

        cur = conn.execute(
            """
            SELECT * FROM user_debts
            WHERE user_id = ? AND status = 'active'
            ORDER BY interest_rate DESC
            """,
            (user_id,),
        )
        result["debts"] = [dict(r) for r in cur.fetchall()]

        cur = conn.execute(
            """
            SELECT * FROM financial_plans
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id,),
        )
        plan_row = cur.fetchone()
        if plan_row:
            plan = dict(plan_row)
            plan["short_term_goals"] = (
                json.loads(plan["short_term_goals"])
                if plan.get("short_term_goals") else {}
            )
            plan["long_term_goals"] = (
                json.loads(plan["long_term_goals"])
                if plan.get("long_term_goals") else {}
            )
            result["active_plan"] = plan
        else:
            result["active_plan"] = None

        return result


def update_user_profile(user_id: str, **kwargs) -> bool:
    """Upsert profile fields.  Only non-None kwargs are written."""
    valid = {k: v for k, v in kwargs.items() if v is not None}
    if not valid:
        return True
    columns = list(valid.keys())
    values = list(valid.values())
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    try:
        with get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO user_profiles (user_id, {', '.join(columns)})
                VALUES (?, {', '.join('?' * len(columns))})
                ON CONFLICT(user_id) DO UPDATE SET
                    {set_clause}, updated_at = CURRENT_TIMESTAMP
                """,
                [user_id] + values + values,
            )
            conn.commit()
            return True
    except Exception as exc:
        print(f"[update_user_profile] {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════
# Preferences
# ═══════════════════════════════════════════════════════════════════

def update_user_preferences(user_id: str, **kwargs) -> bool:
    """Upsert preference fields."""
    valid = {k: v for k, v in kwargs.items() if v is not None}
    if not valid:
        return True
    columns = list(valid.keys())
    values = list(valid.values())
    set_clause = ", ".join(f"{c} = ?" for c in columns)
    try:
        with get_connection() as conn:
            conn.execute(
                f"""
                INSERT INTO user_preferences (user_id, {', '.join(columns)})
                VALUES (?, {', '.join('?' * len(columns))})
                ON CONFLICT(user_id) DO UPDATE SET
                    {set_clause}, updated_at = CURRENT_TIMESTAMP
                """,
                [user_id] + values + values,
            )
            conn.commit()
            return True
    except Exception as exc:
        print(f"[update_user_preferences] {exc}")
        return False


# ═══════════════════════════════════════════════════════════════════
# Debts
# ═══════════════════════════════════════════════════════════════════

def add_user_debt(user_id: str, debt_data: Dict[str, Any]) -> bool:
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO user_debts
                    (user_id, debt_type, creditor, total_amount,
                     remaining_amount, interest_rate, minimum_payment, due_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    debt_data.get("debt_type"),
                    debt_data.get("creditor"),
                    debt_data.get("total_amount"),
                    debt_data.get("remaining_amount"),
                    debt_data.get("interest_rate"),
                    debt_data.get("minimum_payment"),
                    debt_data.get("due_date"),
                ),
            )
            conn.commit()
            return True
    except Exception as exc:
        print(f"[add_user_debt] {exc}")
        return False


def get_user_debts(user_id: str) -> List[Dict]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT * FROM user_debts
            WHERE user_id = ? AND status = 'active'
            ORDER BY interest_rate DESC
            """,
            (user_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def update_debt_status(debt_id: int, status: str) -> bool:
    with get_connection() as conn:
        conn.execute(
            "UPDATE user_debts SET status = ? WHERE debt_id = ?",
            (status, debt_id),
        )
        conn.commit()
        return True


# ═══════════════════════════════════════════════════════════════════
# Financial plans
# ═══════════════════════════════════════════════════════════════════

def create_financial_plan(
    user_id: str,
    plan_name: str,
    short_term_goals: Dict,
    long_term_goals: Dict,
    plan_type: str = "Comprehensive",
) -> bool:
    try:
        with get_connection() as conn:
            # Archive previous active plan
            conn.execute(
                """
                UPDATE financial_plans SET status = 'archived'
                WHERE user_id = ? AND status = 'active'
                """,
                (user_id,),
            )
            conn.execute(
                """
                INSERT INTO financial_plans
                    (user_id, plan_name, plan_type,
                     short_term_goals, long_term_goals, status)
                VALUES (?, ?, ?, ?, ?, 'active')
                """,
                (
                    user_id, plan_name, plan_type,
                    json.dumps(short_term_goals),
                    json.dumps(long_term_goals),
                ),
            )
            conn.commit()
            return True
    except Exception as exc:
        print(f"[create_financial_plan] {exc}")
        return False


def get_active_plan(user_id: str) -> Optional[Dict]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT * FROM financial_plans
            WHERE user_id = ? AND status = 'active'
            ORDER BY created_at DESC LIMIT 1
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        plan = dict(row)
        plan["short_term_goals"] = (
            json.loads(plan["short_term_goals"]) if plan.get("short_term_goals") else {}
        )
        plan["long_term_goals"] = (
            json.loads(plan["long_term_goals"]) if plan.get("long_term_goals") else {}
        )
        return plan


def update_plan_status(plan_id: int, status: str) -> bool:
    with get_connection() as conn:
        conn.execute(
            "UPDATE financial_plans SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE plan_id = ?",
            (status, plan_id),
        )
        conn.commit()
        return True


# ═══════════════════════════════════════════════════════════════════
# Transactions
# ═══════════════════════════════════════════════════════════════════

def add_transaction(
    user_id: str,
    amount: float,
    category: str,
    description: str = "",
    is_expense: bool = True,
) -> bool:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO user_transactions
                (user_id, amount, category, description, is_expense, transaction_date)
            VALUES (?, ?, ?, ?, ?, DATE('now'))
            """,
            (user_id, amount, category, description, is_expense),
        )
        conn.commit()
        return True


def get_recent_transactions(user_id: str, limit: int = 10) -> List[Dict]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT * FROM user_transactions
            WHERE user_id = ?
            ORDER BY transaction_date DESC, created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        return [dict(r) for r in cur.fetchall()]


# ═══════════════════════════════════════════════════════════════════
# Agent decision log
# ═══════════════════════════════════════════════════════════════════

def log_agent_decision(
    user_id: str,
    session_id: str,
    decision_type: str,
    summary: str,
    confidence: float,
) -> bool:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO agent_decisions_log
                (user_id, session_id, decision_type, summary, confidence_score)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, session_id, decision_type, summary, confidence),
        )
        conn.commit()
        return True


# ═══════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════

def calculate_retirement_date(
    employment_start: str, target_age: int, current_age: int
) -> str:
    """Return an ISO date string for the estimated retirement date."""
    try:
        years_left = max(target_age - current_age, 0)
        retirement_year = datetime.now().year + years_left
        start = datetime.strptime(employment_start, "%Y-%m-%d")
        return datetime(retirement_year, start.month, start.day).strftime("%Y-%m-%d")
    except Exception:
        return f"{datetime.now().year + 30}-01-01"


def get_user_summary(user_id: str) -> Dict[str, Any]:
    profile = get_user_profile(user_id)
    if not profile:
        return {"error": "User not found"}

    debts = profile.get("debts", [])
    total_debt = sum(d.get("remaining_amount", 0) for d in debts)
    monthly_min = sum(d.get("minimum_payment", 0) for d in debts)

    completeness = 0
    total_fields = 5
    if profile.get("profile"):         completeness += 1
    if profile.get("preferences"):     completeness += 1
    if debts:                          completeness += 1
    if profile.get("active_plan"):     completeness += 1
    if profile.get("profile", {}) and profile["profile"].get("monthly_income"):
        completeness += 1

    return {
        "user_id": user_id,
        "name": f"{profile.get('first_name', '')} {profile.get('other_names', '')}".strip(),
        "email": profile.get("email"),
        "profile_completeness": int((completeness / total_fields) * 100),
        "has_active_plan": profile.get("active_plan") is not None,
        "total_debt": round(total_debt, 2),
        "monthly_minimum": round(monthly_min, 2),
        "debt_count": len(debts),
    }


def delete_user_data(user_id: str) -> bool:
    """Hard-delete all data for a user (GDPR / test teardown)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM agent_decisions_log WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_transactions WHERE user_id = ?", (user_id,))
        conn.execute(
            "DELETE FROM plan_milestones WHERE plan_id IN "
            "(SELECT plan_id FROM financial_plans WHERE user_id = ?)",
            (user_id,),
        )
        conn.execute("DELETE FROM financial_plans WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_debts WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_profiles WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        conn.commit()
        return True