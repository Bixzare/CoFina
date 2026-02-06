import sqlite3
import os

# Define paths
SRC_DIR = "src"
DB_DIR = os.path.join(SRC_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "cofina.db")

def main():
    """Initialize database schema."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        print(f"Created directory: {DB_DIR}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print(f"Configuring CoFina SQLite database at: {DB_PATH}")

    # Create Tables
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        password_hash TEXT NOT NULL,
        secret_question TEXT NOT NULL,
        secret_answer_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_preferences (
        user_id TEXT PRIMARY KEY,
        risk_profile TEXT,
        debt_strategy TEXT,
        savings_priority TEXT,
        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS financial_plans (
        plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        plan_name TEXT,
        short_term_goals TEXT, 
        long_term_goals TEXT,  
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_decisions_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        decision_type TEXT,
        summary TEXT,
        confidence_score REAL,
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    );
    """)

    conn.commit()
    cur.close()
    conn.close()

    print(" CoFina database schema created successfully.")

def registerDefaultUsers():
    """Test the system with sample users."""
    print("\n" + "="*60)
    print("Testing CoFina Auth & Profile System")
    print("="*60 + "\n")
    
    from src.db.queries import register_user, user_exists, get_secret_question
    from src.tools.user_profile import update_user_preferences, create_financial_plan, get_user_profile

    users_to_create = [
        {
            "id": "iliya0003",
            "name": "Nasiru Iliya",
            "pwd": "4321open!",
            "q": "What is your birth city?",
            "a": "Kaduna",
            "pref": {"risk": "Low", "debt": "Snowball", "save": "Education Fund"},
            "plan": {
                "name": "Nasiru's 2026 Growth Plan",
                "short": {"emergency_fund": 3000, "buy_laptop": 1200},   
                "long": {"buy_house": 2030, "retirement_target": 2055}
            }
        },
        {
            "id": "djibrilla",
            "name": "Djibrilla Boubacar",
            "pwd": "4321open!",
            "q": "What is your favorite food?",
            "a": "Jollof",
            "pref": {"risk": "High", "debt": "Avalanche", "save": "Stock Investment"},
            "plan": {
                "name": "Djibrilla's Wealth Engine",
                "short": {"pay_credit_card": 1500, "crypto_bucket": 500},
                "long": {"financial_independence": 2040, "world_trip": 2032}
            }
        }
    ]

    for user in users_to_create:
        print(f" Processing: {user['name']} ({user['id']})")
        print("-" * 60)
        
        # Check if already exists
        if user_exists(user['id']):
            print(f" User {user['id']} already exists, skipping...")
            profile = get_user_profile(user['id'])
            if profile:
                print(f" Verified existing profile")
                print(f"   Risk Profile: {profile['preferences']['risk_profile']}")
                print(f"   Active Plan: {profile['active_plan']['name'] if profile['active_plan'] else 'None'}")
        else:
            # Register
            reg_ok = register_user(user['id'], user['pwd'], user['q'], user['a'])
            print(f"{'Registered' if reg_ok else 'Not registered'} Registration: {'Success' if reg_ok else 'Failed'}")

            if reg_ok:
                # Set preferences
                pref_ok = update_user_preferences(
                    user['id'], 
                    user['pref']['risk'], 
                    user['pref']['debt'], 
                    user['pref']['save']
                )
                print(f"{'Registered' if pref_ok else 'Not Registered'} Preferences: {'Set' if pref_ok else 'Failed'}")

                # Create plan
                plan_ok = create_financial_plan(
                    user['id'], 
                    user['plan']['name'], 
                    user['plan']['short'], 
                    user['plan']['long']
                )
                print(f"{'Successful created' if plan_ok else 'Not successfully created'} Financial Plan: {'Created' if plan_ok else 'Failed'}")

                # Verify
                final_profile = get_user_profile(user['id'])
                if final_profile:
                    print(f" Profile Verified:")
                    print(f"   User ID: {final_profile['user_id']}")
                    print(f"   Risk Profile: {final_profile['preferences']['risk_profile']}")
                    print(f"   Active Plan: {final_profile['active_plan']['name']}")
        
        print()

    print("-"*60)
    print(" Test Complete! You can now run the agent with these users:")
    print("   - iliya0003 (Secret Answer: Kaduna)")
    print("   - djibrilla (Secret Answer: Jollof)")
    print("-"*60 + "\n")

if __name__ == "__main__":
    main()
    registerDefaultUsers()
    print("\n CoFina database ready with default users.\n")