"""
Evaluation Framework for CoFina - Track verification scores and agent metrics.
"""

from __future__ import annotations

import json
import sqlite3
import numpy as np
import os
class EvaluationMetrics:
    """
    Tracks verification scores, tool success rates, and other agent metrics.
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
            
        self._init_metrics_tables()
        
        self.metric_definitions = {
            "groundedness_score": {
                "description": "How well responses are grounded in retrieved context",
                "range": [0, 1],
                "higher_is_better": True
            },
            "tool_selection_accuracy": {
                "description": "Correctness of tool selection for given tasks",
                "range": [0, 1],
                "higher_is_better": True
            },
            "task_completion_rate": {
                "description": "Percentage of tasks successfully completed",
                "range": [0, 1],
                "higher_is_better": True
            },
            "iterations_before_convergence": {
                "description": "Number of iterations needed to complete task",
                "range": [1, 10],
                "higher_is_better": False
            },
            "hallucination_frequency": {
                "description": "Frequency of ungrounded claims",
                "range": [0, 1],
                "higher_is_better": False
            },
            "plan_adherence_score": {
                "description": "How well user adheres to financial plan",
                "range": [0, 1],
                "higher_is_better": True
            },
            "human_escalation_rate": {
                "description": "Frequency of human intervention needed",
                "range": [0, 1],
                "higher_is_better": False
            },
            "response_time": {
                "description": "Average response time in seconds",
                "range": [0, 10],
                "higher_is_better": False
            },
            "user_satisfaction": {
                "description": "User satisfaction score from feedback",
                "range": [1, 5],
                "higher_is_better": True
            }
        }
    
    def _init_metrics_tables(self):
        """Initialize metrics tracking tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verification_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    user_id TEXT,
                    turn_number INTEGER,
                    query TEXT,
                    response TEXT,
                    rag_context TEXT,
                    score REAL,
                    action TEXT,
                    reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_execution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    tool_name TEXT,
                    success BOOLEAN,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def log_verification(
        self,
        session_id: str,
        user_id: str,
        turn_number: int,
        query: str,
        response: str,
        rag_context: str,
        verification: Dict[str, Any],
    ) -> None:
        """Store a verification result."""
        score = verification.get("score", 0.0)
        self._recent_verifications.append(score)
        if len(self._recent_verifications) > 10:
            self._recent_verifications.pop(0)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO verification_log
                    (session_id, user_id, turn_number, query, response,
                     rag_context, score, action, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, user_id, turn_number, query, response,
                    rag_context[:500],  # truncate context
                    score,
                    verification.get("action", "accept"),
                    verification.get("reason", ""),
                ),
            )

    def log_tool_call(self, session_id: str, tool_name: str, success: bool) -> None:
        """Log tool execution success/failure."""
        self._recent_tool_calls.append(success)
        if len(self._recent_tool_calls) > 10:
            self._recent_tool_calls.pop(0)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO tool_execution_log (session_id, tool_name, success) VALUES (?, ?, ?)",
                (session_id, tool_name, success),
            )

    def get_recent_success_rate(self) -> float:
        """Return tool success rate over the last 10 calls."""
        if not self._recent_tool_calls:
            return 1.0
        return sum(self._recent_tool_calls) / len(self._recent_tool_calls)

    def get_recent_avg_verification_score(self) -> float:
        """Return average verification score over the last 10 turns."""
        if not self._recent_verifications:
            return 0.9  # default high
        return sum(self._recent_verifications) / len(self._recent_verifications)

    def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get aggregated stats for a session."""
        with sqlite3.connect(self.db_path) as conn:
            # Verification stats
            cur = conn.execute(
                """
                SELECT COUNT(*), AVG(score), MIN(score), MAX(score)
                FROM verification_log WHERE session_id = ?
                """,
                (session_id,),
            )
            v_row = cur.fetchone()
            v_count, v_avg, v_min, v_max = v_row if v_row else (0, 0, 0, 0)

            # Tool stats
            cur = conn.execute(
                """
                SELECT COUNT(*), SUM(CASE WHEN success THEN 1 ELSE 0 END)
                FROM tool_execution_log WHERE session_id = ?
                """,
                (session_id,),
            )
            t_row = cur.fetchone()
            t_count, t_success = t_row if t_row else (0, 0)

            return {
                "verification_count": v_count or 0,
                "avg_score": round(v_avg or 0.0, 3),
                "min_score": round(v_min or 0.0, 3),
                "max_score": round(v_max or 0.0, 3),
                "tool_count": t_count or 0,
                "tool_success_rate": round((t_success / t_count) if t_count else 1.0, 3),
            }