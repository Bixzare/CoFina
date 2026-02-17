"""
Evaluation Framework for CoFina - Automated metrics computation
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import sqlite3
import numpy as np
import os
class EvaluationMetrics:
    """
    Computes and tracks evaluation metrics for agent performance
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
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_results (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_case_id TEXT,
                    session_id TEXT,
                    metric_name TEXT,
                    metric_value REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS test_cases (
                    test_case_id TEXT PRIMARY KEY,
                    description TEXT,
                    expected_outcome TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def compute_groundedness(self, responses: List[Dict[str, Any]], 
                              contexts: List[str]) -> float:
        """
        Compute groundedness score based on response-context alignment
        
        Args:
            responses: List of agent responses with metadata
            contexts: List of retrieved contexts used
        
        Returns:
            Groundedness score (0-1)
        """
        if not responses:
            return 0.0
        
        scores = []
        for response, context in zip(responses, contexts):
            # Check for direct citations
            has_citations = "source:" in response.get("content", "").lower()
            
            # Check for factual alignment (simplified)
            response_words = set(response.get("content", "").lower().split())
            context_words = set(context.lower().split())
            
            if context_words:
                overlap = len(response_words.intersection(context_words))
                total = len(response_words)
                alignment = overlap / total if total > 0 else 0
            else:
                alignment = 0
            
            # Combine signals
            score = (alignment * 0.7) + (0.3 if has_citations else 0)
            scores.append(min(score, 1.0))
        
        return np.mean(scores) if scores else 0.0
    
    def compute_tool_selection_accuracy(self, tool_calls: List[Dict], 
                                         expected_tools: List[str]) -> float:
        """
        Compute accuracy of tool selection
        
        Args:
            tool_calls: Actual tool calls made
            expected_tools: Expected tool calls for the task
        
        Returns:
            Accuracy score (0-1)
        """
        if not expected_tools:
            return 1.0 if not tool_calls else 0.0
        
        actual_tools = [call.get("tool") for call in tool_calls]
        
        # Check if all expected tools were called
        expected_set = set(expected_tools)
        actual_set = set(actual_tools)
        
        # Precision and recall
        true_positives = len(expected_set.intersection(actual_set))
        
        if not actual_set:
            return 0.0
        
        precision = true_positives / len(actual_set)
        recall = true_positives / len(expected_set) if expected_set else 0
        
        # F1 score
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        return f1
    
    def compute_task_completion(self, task_log: Dict[str, Any]) -> float:
        """
        Compute task completion rate based on objectives and outcomes
        
        Args:
            task_log: Log of task execution with objectives and completion status
        
        Returns:
            Completion score (0-1)
        """
        score = 0.0
        components = 0
        
        # Check if goal was met
        if task_log.get("goal_met"):
            score += 0.4
        elif task_log.get("partial_progress"):
            score += 0.2
        components += 1
        
        # Check user satisfaction
        if task_log.get("user_rating", 0) >= 4:
            score += 0.2
        elif task_log.get("user_rating", 0) >= 3:
            score += 0.1
        components += 1
        
        # Check efficiency
        expected_calls = task_log.get("expected_tool_calls", 5)
        actual_calls = task_log.get("actual_tool_calls", 0)
        if actual_calls <= expected_calls:
            score += 0.1
        components += 1
        
        # Check correctness (hallucination)
        hallucination_score = task_log.get("hallucination_score", 0)
        if hallucination_score < 0.2:
            score += 0.2
        elif hallucination_score < 0.4:
            score += 0.1
        components += 1
        
        # Check state preservation
        if task_log.get("state_preserved"):
            score += 0.1
        components += 1
        
        return min(score / components, 1.0) if components > 0 else 0.0
    
    def compute_hallucination_frequency(self, verification_logs: List[Dict]) -> float:
        """
        Compute frequency of hallucinations from verification logs
        
        Args:
            verification_logs: Logs from response verification
        
        Returns:
            Hallucination frequency (0-1)
        """
        if not verification_logs:
            return 0.0
        
        low_quality = 0
        for log in verification_logs:
            score = log.get("score", 1.0)
            if score < 0.5:  # Hallucination threshold
                low_quality += 1
        
        return low_quality / len(verification_logs)
    
    def evaluate_test_case(self, test_case_id: str, 
                            execution_logs: List[Dict]) -> Dict[str, float]:
        """
        Run complete evaluation on a test case
        
        Args:
            test_case_id: Identifier for the test case
            execution_logs: Logs from test execution
        
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Extract relevant data from logs
        responses = [log for log in execution_logs if log.get("type") == "response"]
        contexts = [log.get("context", "") for log in execution_logs if log.get("type") == "retrieval"]
        tool_calls = [log for log in execution_logs if log.get("type") == "tool_call"]
        verifications = [log for log in execution_logs if log.get("type") == "verification"]
        
        # Compute metrics
        if responses and contexts:
            metrics["groundedness_score"] = self.compute_groundedness(responses, contexts)
        
        if tool_calls:
            expected = execution_logs[0].get("expected_tools", []) if execution_logs else []
            metrics["tool_selection_accuracy"] = self.compute_tool_selection_accuracy(
                tool_calls, expected
            )
        
        # Task completion requires structured task log
        task_log = execution_logs[0].get("task", {}) if execution_logs else {}
        metrics["task_completion_rate"] = self.compute_task_completion(task_log)
        
        # Iterations count
        metrics["iterations_before_convergence"] = len([l for l in execution_logs 
                                                        if l.get("type") == "turn"])
        
        # Hallucination frequency
        metrics["hallucination_frequency"] = self.compute_hallucination_frequency(verifications)
        
        # Response time
        if responses:
            times = [r.get("duration", 0) for r in responses if r.get("duration")]
            metrics["response_time"] = np.mean(times) if times else 0
        
        # Store results
        self._store_results(test_case_id, metrics, execution_logs)
        
        return metrics
    
    def _store_results(self, test_case_id: str, metrics: Dict[str, float],
                        logs: List[Dict]):
        """Store evaluation results in database"""
        with sqlite3.connect(self.db_path) as conn:
            for metric_name, value in metrics.items():
                conn.execute("""
                    INSERT INTO evaluation_results 
                    (test_case_id, metric_name, metric_value, metadata)
                    VALUES (?, ?, ?, ?)
                """, (
                    test_case_id,
                    metric_name,
                    value,
                    json.dumps({"log_count": len(logs)})
                ))
    
    def get_test_case_results(self, test_case_id: str) -> Dict[str, Any]:
        """Retrieve results for a specific test case"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT metric_name, metric_value, timestamp
                FROM evaluation_results
                WHERE test_case_id = ?
                ORDER BY timestamp DESC
            """, (test_case_id,))
            
            results = {}
            for row in cur.fetchall():
                results[row[0]] = {
                    "value": row[1],
                    "timestamp": row[2]
                }
            
            return results