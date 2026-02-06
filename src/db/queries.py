import sqlite3
import bcrypt
import os

# Path to database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "cofina.db")

def get_connection():
    """Establishes connection to the SQLite database."""
    return sqlite3.connect(DB_PATH)

def user_exists(user_id: str) -> bool:
    """Check if a user exists in the database."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
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

def register_user(user_id, password, secret_question, secret_answer):
    """Securely registers a new user with hashed credentials."""
    pwd_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    ans_hash = bcrypt.hashpw(secret_answer.lower().strip().encode('utf-8'), bcrypt.gensalt())

    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO users (user_id, password_hash, secret_question, secret_answer_hash)
                VALUES (?, ?, ?, ?)
            """, (user_id, pwd_hash, secret_question, ans_hash))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            print(f"Error: User {user_id} already exists.")
            return False

def verify_login(user_id, provided_password):
    """Verifies a user's password against the stored hash."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT password_hash FROM users WHERE user_id = ?", (user_id,))
        result = cur.fetchone()
        
        if result and bcrypt.checkpw(provided_password.encode('utf-8'), result[0]):
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