"""
CoFina Database Setup - Complete schema with all required tables
"""

import sqlite3
import os
import sys
from datetime import datetime

# Add src to path so we can import db modules
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# Define paths
SRC_DIR = "src"
DB_DIR = os.path.join(SRC_DIR, "db")
DB_PATH = os.path.join(DB_DIR, "cofina.db")

def main():
    """Initialize complete database schema."""
    if not os.path.exists(DB_DIR):
        os.makedirs(DB_DIR)
        print(f"✅ Created directory: {DB_DIR}")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print(f"🔧 Configuring CoFina database at: {DB_PATH}")

    # --- USERS TABLE (Authentication) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        other_names TEXT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        secret_question TEXT NOT NULL,
        secret_answer_hash TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        account_status TEXT DEFAULT 'active'
    );
    """)

    # --- USER PROFILES TABLE (Personal & Employment) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_profiles (
        profile_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE,
        profession TEXT,
        current_role TEXT,
        employment_start_date TEXT,
        age INTEGER,
        gender TEXT,
        civil_status TEXT,
        number_of_children INTEGER DEFAULT 0,
        monthly_income REAL,
        annual_income REAL,
        retirement_age_target INTEGER DEFAULT 60,
        estimated_retirement_date TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # --- USER PREFERENCES TABLE (Financial Personality) ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_preferences (
        preference_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT UNIQUE,
        risk_profile TEXT CHECK(risk_profile IN ('Low', 'Moderate', 'High', 'Very High')),
        debt_strategy TEXT CHECK(debt_strategy IN ('Snowball', 'Avalanche', 'Consolidation')),
        savings_priority TEXT,
        investment_horizon TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # --- USER DEBTS TABLE ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_debts (
        debt_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        debt_type TEXT CHECK(debt_type IN ('Student Loan', 'Credit Card', 'Personal Loan', 'Mortgage', 'Car Loan', 'Other')),
        creditor TEXT,
        total_amount REAL,
        remaining_amount REAL,
        interest_rate REAL,
        minimum_payment REAL,
        due_date TEXT,
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # --- FINANCIAL PLANS TABLE ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS financial_plans (
        plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        plan_name TEXT,
        plan_type TEXT CHECK(plan_type IN ('Budget', 'Savings', 'Investment', 'Debt Repayment', 'Comprehensive')),
        short_term_goals TEXT,  -- JSON
        long_term_goals TEXT,    -- JSON
        monthly_budget TEXT,      -- JSON
        allocations TEXT,         -- JSON
        status TEXT DEFAULT 'active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # --- PLAN MILESTONES TABLE ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS plan_milestones (
        milestone_id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id INTEGER,
        milestone_name TEXT,
        target_amount REAL,
        current_amount REAL DEFAULT 0,
        target_date TEXT,
        achieved_date TEXT,
        status TEXT DEFAULT 'pending',
        FOREIGN KEY (plan_id) REFERENCES financial_plans(plan_id) ON DELETE CASCADE
    );
    """)

    # --- USER TRANSACTIONS TABLE ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS user_transactions (
        transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        amount REAL,
        category TEXT,
        description TEXT,
        transaction_date TEXT,
        is_expense BOOLEAN DEFAULT 1,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    # --- AGENT DECISIONS LOG ---
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_decisions_log (
        log_id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        session_id TEXT,
        decision_type TEXT,
        summary TEXT,
        confidence_score REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    print("✅ Database schema created successfully.")
    
    # Create indexes for performance
    cur.execute("CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_plans_user ON financial_plans(user_id, status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_debts_user ON user_debts(user_id, status);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_transactions_user ON user_transactions(user_id, transaction_date);")
    
    conn.commit()
    cur.close()
    conn.close()
    
    print("✅ Indexes created successfully.")

def register_default_users():
    """Register default test users."""
    print("\n" + "="*60)
    print("📝 Setting up default users")
    print("="*60 + "\n")
    
    # Import here after path is set
    from src.db.queries import register_user, user_exists
    from src.tools.user_profile import (
        update_user_profile, update_user_preferences, 
        add_user_debt, create_financial_plan
    )
    
    default_users = [
        {
            "user_id": "niliya",
            "first_name": "Nasiru",
            "other_names": "Iliya",
            "email": "niliya@example.com",
            "password": "SecurePass123!",
            "secret_question": "What is your mother's maiden name?",
            "secret_answer": "Ibrahim",
            "profile": {
                "profession": "Software Engineer",
                "current_role": "Senior Developer",
                "employment_start_date": "2020-01-15",
                "age": 28,
                "gender": "Male",
                "civil_status": "Single",
                "number_of_children": 0,
                "monthly_income": 8500,
                "annual_income": 102000,
                "retirement_age_target": 60
            },
            "preferences": {
                "risk_profile": "Moderate",
                "debt_strategy": "Snowball",
                "savings_priority": "Emergency Fund"
            },
            "debts": [
                {
                    "debt_type": "Student Loan",
                    "creditor": "Sallie Mae",
                    "total_amount": 45000,
                    "remaining_amount": 32000,
                    "interest_rate": 4.5,
                    "minimum_payment": 350,
                    "due_date": "15th"
                }
            ],
            "plan": {
                "name": "Niliya's Financial Independence Plan",
                "short_term": {"emergency_fund": 15000, "new_laptop": 2000},
                "long_term": {"house_down_payment": 60000, "retirement": 2000000}
            }
        },
        {
            "user_id": "djibrilla",
            "first_name": "Djibrilla",
            "other_names": "Boubacar",
            "email": "djibrilla@example.com",
            "password": "SecurePass456!",
            "secret_question": "What was your first pet's name?",
            "secret_answer": "Rex",
            "profile": {
                "profession": "Data Scientist",
                "current_role": "ML Engineer",
                "employment_start_date": "2021-06-01",
                "age": 26,
                "gender": "Male",
                "civil_status": "Married",
                "number_of_children": 1,
                "monthly_income": 7200,
                "annual_income": 86400,
                "retirement_age_target": 58
            },
            "preferences": {
                "risk_profile": "High",
                "debt_strategy": "Avalanche",
                "savings_priority": "Investment"
            },
            "debts": [
                {
                    "debt_type": "Credit Card",
                    "creditor": "Chase",
                    "total_amount": 5000,
                    "remaining_amount": 2800,
                    "interest_rate": 18.5,
                    "minimum_payment": 150,
                    "due_date": "10th"
                },
                {
                    "debt_type": "Car Loan",
                    "creditor": "Toyota Financial",
                    "total_amount": 25000,
                    "remaining_amount": 18000,
                    "interest_rate": 3.9,
                    "minimum_payment": 420,
                    "due_date": "5th"
                }
            ],
            "plan": {
                "name": "Djibrilla's Wealth Builder",
                "short_term": {"credit_card_payoff": 2800, "vacation": 3000},
                "long_term": {"investment_portfolio": 100000, "business": 50000}
            }
        }
    ]
    
    for user in default_users:
        print(f"👤 Processing: {user['first_name']} {user.get('other_names', '')} ({user['user_id']})")
        print("-" * 60)
        
        if user_exists(user['user_id']):
            print(f"   ⏭️  User already exists, skipping...")
        else:
            # Register user
            reg_ok = register_user(
                user['user_id'],
                user['first_name'],
                user.get('other_names', ''),
                user['email'],
                user['password'],
                user['secret_question'],
                user['secret_answer']
            )
            print(f"   ✅ Registration: {'Success' if reg_ok else 'Failed'}")
            
            if reg_ok:
                # Update profile
                profile = user['profile']
                retirement_date = calculate_retirement_date(
                    profile.get('employment_start_date'),
                    profile.get('retirement_age_target', 60),
                    profile.get('age', 30)
                )
                
                profile_ok = update_user_profile(
                    user['user_id'],
                    profession=profile['profession'],
                    current_role=profile['current_role'],
                    employment_start_date=profile['employment_start_date'],
                    age=profile['age'],
                    gender=profile['gender'],
                    civil_status=profile['civil_status'],
                    number_of_children=profile['number_of_children'],
                    monthly_income=profile['monthly_income'],
                    annual_income=profile['annual_income'],
                    retirement_age_target=profile['retirement_age_target'],
                    estimated_retirement_date=retirement_date
                )
                print(f"   ✅ Profile: {'Saved' if profile_ok else 'Failed'}")
                
                # Set preferences
                pref = user['preferences']
                pref_ok = update_user_preferences(
                    user['user_id'],
                    risk_profile=pref['risk_profile'],
                    debt_strategy=pref['debt_strategy'],
                    savings_priority=pref['savings_priority']
                )
                print(f"   ✅ Preferences: {'Set' if pref_ok else 'Failed'}")
                
                # Add debts
                for debt in user.get('debts', []):
                    debt_ok = add_user_debt(user['user_id'], debt)
                    print(f"   ✅ Debt ({debt['debt_type']}): {'Added' if debt_ok else 'Failed'}")
                
                # Create plan
                plan = user['plan']
                plan_ok = create_financial_plan(
                    user['user_id'],
                    plan['name'],
                    plan['short_term'],
                    plan['long_term']
                )
                print(f"   ✅ Financial Plan: {'Created' if plan_ok else 'Failed'}")
        
        print()

def calculate_retirement_date(employment_start: str, target_age: int, current_age: int) -> str:
    """Calculate estimated retirement date"""
    from datetime import datetime
    
    if not employment_start or not current_age:
        return datetime.now().replace(year=datetime.now().year + 30).strftime("%Y-%m-%d")
    
    try:
        start = datetime.strptime(employment_start, "%Y-%m-%d")
        years_to_retirement = target_age - current_age
        # Rough estimate: assume working years + 35 years of savings growth
        retirement_year = start.year + years_to_retirement + 35
        # Ensure we don't go beyond reasonable date
        if retirement_year > start.year + 60:
            retirement_year = start.year + 60
        retirement_date = datetime(retirement_year, start.month, start.day)
        return retirement_date.strftime("%Y-%m-%d")
    except:
        return datetime.now().replace(year=datetime.now().year + 30).strftime("%Y-%m-%d")

if __name__ == "__main__":
    main()
    register_default_users()
    print("\n" + "="*60)
    print("✅ CoFina database ready with complete schema!")
    print("="*60 + "\n")