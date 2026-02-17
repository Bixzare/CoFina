"""
Monitor Agent - Monitors financial alerts and updates
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
import sqlite3
import os

class MonitorAgent:
    """
    Specialized agent for monitoring financial activities and sending alerts
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to src/db/cofina.db relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.db_path = os.path.join(base_dir, "db", "cofina.db")
        else:
            self.db_path = db_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
        self._init_monitor_tables()
        
        # Alert thresholds
        self.thresholds = {
            "low_balance": 1000,  # Alert if balance below this
            "large_transaction": 500,  # Alert for transactions above this
            "spike_multiplier": 1.5,  # Alert if spending > 1.5x average
            "bill_reminder_days": 3  # Remind X days before bill due
        }
    
    def _init_monitor_tables(self):
        """Initialize monitoring tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS financial_alerts (
                    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    alert_type TEXT,
                    message TEXT,
                    severity TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    acknowledged BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    amount REAL,
                    category TEXT,
                    description TEXT,
                    transaction_date TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def process(self, user_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process monitoring for a user
        
        Args:
            user_id: User identifier
            context: Current context
        
        Returns:
            Alerts and monitoring results
        """
        alerts = []
        
        # Check for various conditions
        balance_alert = self._check_balance(user_id, context)
        if balance_alert:
            alerts.append(balance_alert)
        
        spending_alert = self._check_spending_patterns(user_id, context)
        if spending_alert:
            alerts.append(spending_alert)
        
        bill_alert = self._check_upcoming_bills(user_id, context)
        if bill_alert:
            alerts.append(bill_alert)
        
        goal_alert = self._check_goal_progress(user_id, context)
        if goal_alert:
            alerts.append(goal_alert)
        
        # Store alerts
        for alert in alerts:
            self._store_alert(user_id, alert)
        
        return {
            "action": "monitoring_results",
            "message": f"Found {len(alerts)} items to review",
            "data": {
                "alerts": alerts,
                "timestamp": datetime.now().isoformat()
            }
        }
    
    def _check_balance(self, user_id: str, context: Dict) -> Optional[Dict]:
        """Check if balance is low"""
        # In production, would get actual balance from bank integration
        balance = context.get("user_profile", {}).get("balance", 0)
        
        if balance < self.thresholds["low_balance"]:
            return {
                "type": "low_balance",
                "severity": "high" if balance < 500 else "medium",
                "message": f"Your balance (${balance}) is below ${self.thresholds['low_balance']}",
                "suggestion": "Consider reviewing upcoming expenses and adjusting spending"
            }
        return None
    
    def _check_spending_patterns(self, user_id: str, context: Dict) -> Optional[Dict]:
        """Check for unusual spending patterns"""
        # Get recent transactions
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT amount, category, transaction_date
                FROM transactions
                WHERE user_id = ? AND transaction_date > ?
                ORDER BY transaction_date DESC
                LIMIT 20
            """, (user_id, (datetime.now() - timedelta(days=30)).isoformat()))
            
            transactions = cur.fetchall()
        
        if len(transactions) < 5:
            return None
        
        # Calculate averages by category
        category_spending = {}
        for trans in transactions:
            amount = trans[0]
            category = trans[1]
            if category not in category_spending:
                category_spending[category] = []
            category_spending[category].append(amount)
        
        # Check for spikes
        alerts = []
        for category, amounts in category_spending.items():
            if len(amounts) >= 3:
                avg = sum(amounts[:-1]) / (len(amounts) - 1)  # Exclude most recent
                latest = amounts[-1]
                if latest > avg * self.thresholds["spike_multiplier"]:
                    return {
                        "type": "spending_spike",
                        "severity": "medium",
                        "message": f"Spending in {category} is {(latest/avg - 1)*100:.0f}% above your average",
                        "suggestion": "Review if this is a one-time expense or a new pattern"
                    }
        
        return None
    
    def _check_upcoming_bills(self, user_id: str, context: Dict) -> Optional[Dict]:
        """Check for upcoming bills"""
        # In production, would get from calendar integration
        # For now, return placeholder
        return {
            "type": "bill_reminder",
            "severity": "low",
            "message": "No upcoming bills in the next 3 days",
            "suggestion": "Set up bill reminders in your calendar"
        }
    
    def _check_goal_progress(self, user_id: str, context: Dict) -> Optional[Dict]:
        """Check progress toward financial goals"""
        profile = context.get("user_profile", {})
        active_plan = profile.get("active_plan")
        
        if not active_plan:
            return None
        
        # Check if behind on goals
        # Simplified - in production would compare actual savings to targets
        return {
            "type": "goal_check",
            "severity": "info",
            "message": "Time to review your financial goals",
            "suggestion": "Check if you're on track with your savings targets"
        }
    
    def _store_alert(self, user_id: str, alert: Dict):
        """Store alert in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO financial_alerts 
                (user_id, alert_type, message, severity, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (
                user_id,
                alert["type"],
                alert["message"],
                alert["severity"],
                json.dumps({"suggestion": alert.get("suggestion")})
            ))
    
    def get_unacknowledged_alerts(self, user_id: str) -> List[Dict]:
        """Get all unacknowledged alerts for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT alert_type, message, severity, created_at, metadata
                FROM financial_alerts
                WHERE user_id = ? AND acknowledged = 0
                ORDER BY created_at DESC
            """, (user_id,))
            
            return [
                {
                    "type": row[0],
                    "message": row[1],
                    "severity": row[2],
                    "created_at": row[3],
                    "metadata": json.loads(row[4]) if row[4] else {}
                }
                for row in cur.fetchall()
            ]
    
    def acknowledge_alert(self, alert_id: int):
        """Mark alert as acknowledged"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE financial_alerts SET acknowledged = 1 WHERE alert_id = ?",
                (alert_id,)
            )