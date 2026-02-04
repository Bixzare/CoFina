import sqlite3
import os

DB_NAME = "user_data.db"

def initialize_db():
    if os.path.exists(DB_NAME):
        print(f"Database {DB_NAME} already exists.")
        return

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Create users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            username TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create preferences table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            user_id TEXT,
            key TEXT,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, key),
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # Create conversation_summary table (optional, but good for context)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            user_id TEXT PRIMARY KEY,
            summary TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # Insert a default test user
    cursor.execute("INSERT OR IGNORE INTO users (user_id, username) VALUES (?, ?)", ("default_user", "Guest"))
    cursor.execute("INSERT OR IGNORE INTO preferences (user_id, key, value) VALUES (?, ?, ?)", 
                   ("default_user", "risk_tolerance", "medium"))

    conn.commit()
    conn.close()
    print(f"Database {DB_NAME} initialized successfully.")

if __name__ == "__main__":
    initialize_db()
