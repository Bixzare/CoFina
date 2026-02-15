"""
Failure Analysis - Deep dive into system failures
"""

from typing import Dict, Any, List
import json
from datetime import datetime

class FailureAnalyzer:
    """
    Analyzes system failures and provides root cause analysis
    """
    
    def __init__(self):
        self.failure_patterns = []
    
    def analyze_failure(self, test_case: Dict, execution_logs: List[Dict]) -> Dict[str, Any]:
        """
        Perform deep dive analysis on a failure
        
        Args:
            test_case: The test case that failed
            execution_logs: Logs from test execution
        
        Returns:
            Detailed failure analysis
        """
        analysis = {
            "test_case": test_case["id"],
            "name": test_case["name"],
            "timestamp": datetime.now().isoformat(),
            "failure_point": None,
            "root_cause": None,
            "technical_explanation": None,
            "fix_applied": None,
            "before_vs_after": None
        }
        
        # Find failure point
        failure_point = self._identify_failure_point(execution_logs)
        analysis["failure_point"] = failure_point
        
        # Determine root cause
        root_cause = self._determine_root_cause(failure_point, execution_logs)
        analysis["root_cause"] = root_cause
        analysis["technical_explanation"] = self._generate_technical_explanation(root_cause)
        
        # Suggest fix
        fix = self._suggest_fix(root_cause)
        analysis["fix_applied"] = fix
        
        return analysis
    
    def _identify_failure_point(self, logs: List[Dict]) -> Dict[str, Any]:
        """Identify where in the execution the failure occurred"""
        for i, log in enumerate(logs):
            # Check for error indicators
            if "error" in str(log).lower():
                return {
                    "turn": i,
                    "log": log,
                    "type": "explicit_error"
                }
            
            # Check for unexpected responses
            if log.get("type") == "turn":
                expected = log.get("expected_intent")
                actual = self._detect_actual_intent(log.get("agent", ""))
                
                if expected and actual and expected != actual:
                    return {
                        "turn": i,
                        "log": log,
                        "type": "intent_mismatch",
                        "expected": expected,
                        "actual": actual
                    }
        
        return {
            "turn": len(logs) - 1,
            "type": "unknown",
            "log": logs[-1] if logs else {}
        }
    
    def _determine_root_cause(self, failure_point: Dict, logs: List[Dict]) -> Dict[str, Any]:
        """Determine root cause of failure"""
        failure_type = failure_point.get("type")
        
        if failure_type == "explicit_error":
            return self._analyze_error_log(failure_point["log"])
        
        elif failure_type == "intent_mismatch":
            return self._analyze_intent_mismatch(failure_point, logs)
        
        else:
            return {
                "category": "unknown",
                "description": "Could not determine root cause",
                "confidence": 0.3
            }
    
    def _analyze_error_log(self, log: Dict) -> Dict[str, Any]:
        """Analyze error log to find cause"""
        error_text = str(log)
        
        if "API" in error_text or "api" in error_text.lower():
            return {
                "category": "external_service",
                "subcategory": "api_failure",
                "description": "External API call failed",
                "confidence": 0.9
            }
        elif "database" in error_text.lower() or "db" in error_text.lower():
            return {
                "category": "data_layer",
                "subcategory": "database_error",
                "description": "Database connection or query failed",
                "confidence": 0.9
            }
        elif "timeout" in error_text.lower():
            return {
                "category": "performance",
                "subcategory": "timeout",
                "description": "Operation timed out",
                "confidence": 0.8
            }
        else:
            return {
                "category": "system",
                "subcategory": "unhandled_exception",
                "description": f"Unhandled exception: {error_text[:100]}",
                "confidence": 0.7
            }
    
    def _analyze_intent_mismatch(self, failure_point: Dict, logs: List[Dict]) -> Dict[str, Any]:
        """Analyze intent mismatch to find cause"""
        expected = failure_point.get("expected")
        actual = failure_point.get("actual")
        
        # Check if RAG might have failed
        rag_attempts = [l for l in logs if l.get("type") == "retrieval"]
        if not rag_attempts and expected == "financial":
            return {
                "category": "rag_failure",
                "subcategory": "no_context",
                "description": "RAG retrieval failed or wasn't attempted for financial query",
                "confidence": 0.8
            }
        
        # Check for tool selection issues
        tool_calls = [l for l in logs if l.get("type") == "tool_call"]
        if not tool_calls and expected in ["market", "financial"]:
            return {
                "category": "tool_selection",
                "subcategory": "no_tool_called",
                "description": "Agent failed to call required tools",
                "confidence": 0.7
            }
        
        return {
            "category": "reasoning",
            "subcategory": "incorrect_intent_classification",
            "description": f"Expected {expected} but got {actual}",
            "confidence": 0.6
        }
    
    def _generate_technical_explanation(self, root_cause: Dict) -> str:
        """Generate technical explanation of the failure"""
        category = root_cause.get("category", "unknown")
        
        explanations = {
            "external_service": """
                Technical Explanation:
                The agent attempted to call an external API but the call failed. 
                This could be due to network issues, API rate limiting, or authentication problems.
                In production, this would trigger retry logic with exponential backoff.
            """,
            "data_layer": """
                Technical Explanation:
                Database operation failed. This could be due to connection pool exhaustion,
                schema mismatch, or transaction conflicts. The agent should implement connection
                retry and circuit breaker patterns.
            """,
            "rag_failure": """
                Technical Explanation:
                The RAG retrieval system failed to return relevant context. This could be due to
                embedding generation failure, vector store unavailability, or poor query formulation.
                The summarizer agent should have been triggered to handle this case.
            """,
            "tool_selection": """
                Technical Explanation:
                The agent failed to select the appropriate tool for the task. This indicates a
                reasoning failure where the LLM didn't recognize the need for specialized tools.
                The orchestrator's routing logic may need refinement.
            """,
            "reasoning": """
                Technical Explanation:
                The agent's reasoning chain produced an incorrect intent classification.
                This could be due to ambiguous user input, missing context, or model limitations.
                Adaptive control should detect this and adjust prompting strategies.
            """
        }
        
        return explanations.get(category, "Technical details unavailable.")
    
    def _suggest_fix(self, root_cause: Dict) -> Dict[str, Any]:
        """Suggest fix for the root cause"""
        category = root_cause.get("category")
        
        fixes = {
            "external_service": {
                "immediate": "Implement retry with exponential backoff",
                "short_term": "Add circuit breaker pattern",
                "long_term": "Cache API responses and implement fallback providers",
                "code_change": """
                    def call_with_retry(func, max_retries=3):
                        for attempt in range(max_retries):
                            try:
                                return func()
                            except Exception as e:
                                if attempt == max_retries - 1:
                                    return fallback_handler()
                                time.sleep(2 ** attempt)
                """
            },
            "data_layer": {
                "immediate": "Add connection retry logic",
                "short_term": "Implement connection pooling",
                "long_term": "Add read replicas for scaling",
                "code_change": """
                    def get_db_connection(max_attempts=3):
                        for attempt in range(max_attempts):
                            try:
                                return create_connection()
                            except sqlite3.OperationalError:
                                if attempt == max_attempts - 1:
                                    raise
                                time.sleep(1)
                """
            },
            "rag_failure": {
                "immediate": "Trigger summarizer agent with cached content",
                "short_term": "Add fallback to general knowledge",
                "long_term": "Improve query expansion and retrieval",
                "code_change": """
                    if not rag_results:
                        # Trigger summarizer with cached content
                        summary = summarizer_agent.process(
                            text=cached_knowledge,
                            max_length=500
                        )
                        return generate_response(summary)
                """
            },
            "tool_selection": {
                "immediate": "Add explicit tool selection prompt",
                "short_term": "Enhance orchestrator routing rules",
                "long_term": "Implement few-shot examples for tool selection",
                "code_change": """
                    # Add to system prompt
                    TOOL_SELECTION_GUIDELINES = {
                        'market': 'Use for product searches',
                        'financial': 'Use for planning and budgeting',
                        'registration': 'Use for account management'
                    }
                """
            }
        }
        
        return fixes.get(category, {
            "immediate": "Review logs and add specific error handling",
            "short_term": "Add more comprehensive testing",
            "long_term": "Implement monitoring and alerting"
        })
    
    def _detect_actual_intent(self, response: str) -> str:
        """Detect actual intent from response"""
        response_lower = response.lower()
        
        if any(word in response_lower for word in ["register", "login", "password"]):
            return "registration"
        elif any(word in response_lower for word in ["budget", "plan", "save", "goal"]):
            return "financial"
        elif any(word in response_lower for word in ["product", "buy", "price", "shop"]):
            return "market"
        elif any(word in response_lower for word in ["alert", "remind", "monitor"]):
            return "monitor"
        else:
            return "general"
    
    def generate_report(self, analysis: Dict[str, Any]) -> str:
        """Generate formatted failure analysis report"""
        report = f"""
{'='*80}
FAILURE ANALYSIS REPORT
{'='*80}

Test Case: {analysis['name']} ({analysis['test_case']})
Timestamp: {analysis['timestamp']}

{'='*80}
FAILURE POINT
{'='*80}
Turn: {analysis['failure_point'].get('turn', 'Unknown')}
Type: {analysis['failure_point'].get('type', 'Unknown')}
Details: {json.dumps(analysis['failure_point'].get('log', {}), indent=2)}

{'='*80}
ROOT CAUSE ANALYSIS
{'='*80}
Category: {analysis['root_cause'].get('category', 'Unknown')}
Subcategory: {analysis['root_cause'].get('subcategory', 'Unknown')}
Description: {analysis['root_cause'].get('description', 'Unknown')}
Confidence: {analysis['root_cause'].get('confidence', 0):.0%}

{analysis.get('technical_explanation', 'No technical explanation available.')}

{'='*80}
RECOMMENDED FIXES
{'='*80}
Immediate: {analysis['fix_applied'].get('immediate', 'N/A')}
Short-term: {analysis['fix_applied'].get('short_term', 'N/A')}
Long-term: {analysis['fix_applied'].get('long_term', 'N/A')}

Code Change:
{analysis['fix_applied'].get('code_change', '// No code change suggested')}

{'='*80}
BEFORE VS AFTER COMPARISON
{'='*80}
Before Fix:
- System would fail with error or incorrect response
- User experience degraded
- No graceful degradation

After Fix:
- System handles failure gracefully
- User informed of limitations
- Fallback mechanisms activated
- Continuous improvement through learning

{'='*80}
"""
        return report