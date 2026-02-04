import sqlite3
from typing import Dict, Any, List
from pathlib import Path

# Assuming the DB is in the project root, two levels up from src/tools
DB_NAME = Path(__file__).parent.parent.parent / "user_data.db"

def get_db_connection():
    """Create a database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def get_user_profile(user_id: str) -> str:
    """
    Retrieve the full profile and preferences for a user.
    Useful for understanding user context before answering questions.
    """
    conn = get_db_connection()
    try:
        # Get basic info
        user = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not user:
            return "User not found."
            
        # Get preferences
        prefs = conn.execute("SELECT key, value FROM preferences WHERE user_id = ?", (user_id,)).fetchall()
        
        # Format as a readable string for the LLM
        profile_str = f"User Profile for {user['username']} (ID: {user_id}):\n"
        if prefs:
            profile_str += "Preferences:\n"
            for p in prefs:
                profile_str += f"- {p['key']}: {p['value']}\n"
        else:
            profile_str += "No specific preferences recorded.\n"
            
        return profile_str
    except Exception as e:
        return f"Error retrieving profile: {e}"
    finally:
        conn.close()

def update_preference(user_id: str, key: str, value: str) -> str:
    """
    Update or set a specific user preference.
    Use this when the user explicitly states a preference (e.g., "I'm risk averse", "I want to save for a house").
    """
    conn = get_db_connection()
    try:
        # Ensure user exists (simple check)
        user = conn.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not user:
            # Auto-create for simplicity in this demo
            conn.execute("INSERT INTO users (user_id, username) VALUES (?, ?)", (user_id, "Guest"))
            
        conn.execute("""
            INSERT INTO preferences (user_id, key, value, updated_at) 
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, key) DO UPDATE SET 
            value=excluded.value, 
            updated_at=CURRENT_TIMESTAMP
        """, (user_id, key, value))
        
        conn.commit()
        return f"Successfully updated preference: {key} = {value}"
    except Exception as e:
        return f"Error updating preference: {e}"
    finally:
        conn.close()

# LangChain Tool Definitions
from langchain_core.tools import tool

@tool
def get_user_info(user_id: str = "default_user") -> str:
    """
    Get the current user's profile and preferences. 
    ALWAYS call this at the start of a conversation to understand the user's context.
    """
    return get_user_profile(user_id)

@tool
def save_user_preference(key: str, value: str, user_id: str = "default_user") -> str:
    """
    Save a user preference to the database.
    Call this when the user expresses a preference about their financial goals, risk tolerance, or situation.
    Example: key="risk_tolerance", value="low"
    """
    return update_preference(user_id, key, value)
