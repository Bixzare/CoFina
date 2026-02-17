"""
Adaptive Control for CoFina - Closed-loop behavior modification based on feedback
"""

from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import json
import sqlite3
import os
class AdaptiveController:
    """
    Monitors agent performance and adapts behavior in real-time
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to src/db/cofina.db relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.db_path = os.path.join(base_dir, "db", "cofina.db")
        else:
            self.db_path = db_path
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.thresholds = {
            "groundedness": 0.7,
            "tool_success_rate": 0.8,
            "max_iterations": 5,
            "confidence": 0.6,
            "hallucination": 0.3
        }
        
        self.adaptation_strategies = {
            "low_groundedness": self._handle_low_groundedness,
            "tool_failure": self._handle_tool_failure,
            "iteration_limit": self._handle_iteration_limit,
            "low_confidence": self._handle_low_confidence,
            "hallucination_detected": self._handle_hallucination
        }
        
        self._init_adaptation_tables()
    
    def _init_adaptation_tables(self):
        """Initialize adaptation tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS adaptation_log (
                    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    trigger_type TEXT,
                    metric_value REAL,
                    threshold REAL,
                    strategy_applied TEXT,
                    outcome TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def observe_and_adapt(self, metrics: Dict[str, Any], 
                          session_id: str,
                          available_actions: List[Callable]) -> Optional[Dict]:
        """
        Observe metrics and adapt behavior if needed
        
        Args:
            metrics: Current performance metrics
            session_id: Current session
            available_actions: List of available adaptation actions
        
        Returns:
            Adaptation decision or None
        """
        adaptations = []
        
        # Check each metric against thresholds
        for metric_name, value in metrics.items():
            if metric_name in self.thresholds:
                threshold = self.thresholds[metric_name]
                
                # Different metrics have different comparison directions
                if metric_name == "hallucination":
                    # Lower is better
                    if value > threshold:
                        trigger = f"high_{metric_name}"
                        strategy = self.adaptation_strategies.get(metric_name)
                        if strategy:
                            action = strategy(value, threshold, available_actions)
                            adaptations.append({
                                "trigger": trigger,
                                "metric": metric_name,
                                "value": value,
                                "threshold": threshold,
                                "action": action
                            })
                else:
                    # Higher is better for most metrics
                    if value < threshold:
                        trigger = f"low_{metric_name}"
                        strategy = self.adaptation_strategies.get(metric_name)
                        if strategy:
                            action = strategy(value, threshold, available_actions)
                            adaptations.append({
                                "trigger": trigger,
                                "metric": metric_name,
                                "value": value,
                                "threshold": threshold,
                                "action": action
                            })
        
        # Log adaptations
        if adaptations:
            self._log_adaptations(session_id, adaptations)
            
            # Return the most critical adaptation
            return adaptations[0]
        
        return None
    
    def _handle_low_groundedness(self, value: float, threshold: float, 
                                   actions: List[Callable]) -> Dict:
        """Handle low groundedness by re-retrieving with better context"""
        return {
            "type": "re_retrieve",
            "params": {
                "expand_query": True,
                "use_more_sources": True,
                "min_confidence": threshold
            },
            "message": f"Groundedness low ({value:.2f}). Re-retrieving with broader context."
        }
    
    def _handle_tool_failure(self, value: float, threshold: float,
                              actions: List[Callable]) -> Dict:
        """Handle tool failures by retrying or using alternatives"""
        return {
            "type": "tool_fallback",
            "params": {
                "max_retries": 3,
                "use_alternative": True,
                "exponential_backoff": True
            },
            "message": f"Tool success rate low ({value:.2f}). Activating fallback mechanisms."
        }
    
    def _handle_iteration_limit(self, value: float, threshold: float,
                                 actions: List[Callable]) -> Dict:
        """Handle hitting iteration limits by escalating"""
        return {
            "type": "escalate",
            "params": {
                "escalation_level": "human_review",
                "simplify_task": True
            },
            "message": f"Iteration limit reached. Escalating for review."
        }
    
    def _handle_low_confidence(self, value: float, threshold: float,
                                actions: List[Callable]) -> Dict:
        """Handle low confidence by requesting clarification"""
        return {
            "type": "request_clarification",
            "params": {
                "ask_for_confirmation": True,
                "provide_options": True
            },
            "message": f"Confidence low ({value:.2f}). Requesting user clarification."
        }
    
    def _handle_hallucination(self, value: float, threshold: float,
                               actions: List[Callable]) -> Dict:
        """Handle detected hallucinations by regenerating with stricter grounding"""
        return {
            "type": "regenerate",
            "params": {
                "stricter_grounding": True,
                "require_citations": True
            },
            "message": f"Hallucination detected ({value:.2f}). Regenerating with stricter controls."
        }
    
    def _log_adaptations(self, session_id: str, adaptations: List[Dict]):
        """Log adaptation decisions for analysis"""
        with sqlite3.connect(self.db_path) as conn:
            for adapt in adaptations:
                conn.execute("""
                    INSERT INTO adaptation_log 
                    (session_id, trigger_type, metric_value, threshold, strategy_applied)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session_id,
                    adapt["trigger"],
                    adapt["value"],
                    adapt["threshold"],
                    json.dumps(adapt["action"])
                ))
    
    def update_thresholds(self, new_thresholds: Dict[str, float]):
        """Dynamically update thresholds based on performance"""
        self.thresholds.update(new_thresholds)
    
    def get_adaptation_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get adaptation history for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT trigger_type, metric_value, threshold, strategy_applied, timestamp
                FROM adaptation_log
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (session_id, limit))
            
            return [
                {
                    "trigger": row[0],
                    "value": row[1],
                    "threshold": row[2],
                    "strategy": json.loads(row[3]),
                    "timestamp": row[4]
                }
                for row in cur.fetchall()
            ]