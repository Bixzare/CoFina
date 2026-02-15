"""
Test Cases for CoFina Evaluation
"""

import json
from datetime import datetime
from typing import Dict, List, Any

class TestCases:
    """Collection of test cases for CoFina evaluation"""
    
    @staticmethod
    def get_test_cases() -> List[Dict[str, Any]]:
        """Get all test cases"""
        return [
            TestCases.test_case_1_new_user(),
            TestCases.test_case_2_impulse_purchase(),
            TestCases.test_case_3_plan_adjustment(),
            TestCases.test_case_4_multi_session(),
            TestCases.test_case_5_system_degradation()
        ]
    
    @staticmethod
    def test_case_1_new_user() -> Dict[str, Any]:
        """Test Case 1: New User Onboarding"""
        return {
            "id": "TC001",
            "name": "New User Onboarding",
            "description": "First-time user wants to start financial planning",
            "scenario": {
                "user_type": "guest",
                "initial_state": {}
            },
            "conversation_flow": [
                {"user": "I need help with my finances", "expected_intent": "registration"},
                {"user": "I'm new", "expected_intent": "registration"},
                {"user": "user123", "expected_intent": "registration"},
                {"user": "password123", "expected_intent": "registration"},
                {"user": "What was your first pet's name?", "expected_intent": "registration"},
                {"user": "Rex", "expected_intent": "registration"},
                {"user": "Low", "expected_intent": "financial"},
                {"user": "Snowball", "expected_intent": "financial"},
                {"user": "Emergency Fund", "expected_intent": "financial"},
                {"user": "Save $5000 for emergency", "expected_intent": "financial"},
                {"user": "Buy a house in 5 years", "expected_intent": "financial"},
                {"user": "My First Plan", "expected_intent": "financial"}
            ],
            "expected_outcomes": {
                "user_registered": True,
                "plan_created": True,
                "pdf_generated": True,
                "session_persisted": True
            },
            "evaluation_criteria": {
                "task_completion_rate": 1.0,
                "tool_selection_accuracy": 0.9,
                "iterations_before_convergence": 12
            }
        }
    
    @staticmethod
    def test_case_2_impulse_purchase() -> Dict[str, Any]:
        """Test Case 2: Impulse Purchase Check"""
        return {
            "id": "TC002",
            "name": "Impulse Purchase Check",
            "description": "User wants to buy expensive item while behind on savings",
            "scenario": {
                "user_type": "authenticated",
                "user_id": "iliya0003",
                "initial_state": {
                    "profile": {
                        "income": 65000,
                        "savings": 2000,
                        "goals": {"emergency_fund": 10000}
                    }
                }
            },
            "conversation_flow": [
                {"user": "Can I buy a MacBook for $2000?", "expected_intent": "market"},
                {"user": "What about a refurbished one?", "expected_intent": "market"},
                {"user": "Okay, I'll wait", "expected_intent": "general"}
            ],
            "expected_outcomes": {
                "product_search_performed": True,
                "affordability_check": True,
                "alternative_suggestions": True,
                "user_decides_to_wait": True
            },
            "evaluation_criteria": {
                "task_completion_rate": 0.8,
                "tool_selection_accuracy": 0.85,
                "groundedness_score": 0.9
            }
        }
    
    @staticmethod
    def test_case_3_plan_adjustment() -> Dict[str, Any]:
        """Test Case 3: Plan Adjustment After Overspending"""
        return {
            "id": "TC003",
            "name": "Plan Adjustment After Overspending",
            "description": "User exceeded budget, system adjusts plan",
            "scenario": {
                "user_type": "authenticated",
                "user_id": "djibrilla",
                "initial_state": {
                    "profile": {
                        "income": 75000,
                        "preferences": {"risk_profile": "High", "debt_strategy": "Avalanche"}
                    },
                    "transactions": [
                        {"amount": 200, "category": "dining", "date": "2026-02-01"},
                        {"amount": 250, "category": "dining", "date": "2026-02-08"},
                        {"amount": 300, "category": "dining", "date": "2026-02-15"}
                    ]
                }
            },
            "conversation_flow": [
                {"user": "Why am I getting alerts?", "expected_intent": "monitor"},
                {"user": "Yes, I've been eating out more", "expected_intent": "financial"},
                {"user": "Accept the adjustment", "expected_intent": "financial"}
            ],
            "expected_outcomes": {
                "spending_pattern_detected": True,
                "alert_generated": True,
                "plan_adjusted": True,
                "user_approved": True
            },
            "evaluation_criteria": {
                "task_completion_rate": 0.9,
                "human_escalation_rate": 0.2,
                "plan_adherence_score": 0.7
            }
        }
    
    @staticmethod
    def test_case_4_multi_session() -> Dict[str, Any]:
        """Test Case 4: Multi-Session Continuity"""
        return {
            "id": "TC004",
            "name": "Multi-Session Continuity",
            "description": "User returns after 1 week, system remembers context",
            "scenario": {
                "user_type": "authenticated",
                "user_id": "iliya0003",
                "initial_state": {
                    "profile": {
                        "preferences": {"risk_profile": "Low", "savings_priority": "Education"}
                    },
                    "previous_session": {
                        "date": "2026-02-07",
                        "last_topic": "saving for education",
                        "progress": {"education_fund": 4500, "target": 10000}
                    }
                }
            },
            "conversation_flow": [
                {"user": "I'm back", "expected_intent": "general"},
                {"user": "How am I doing on my education savings?", "expected_intent": "financial"}
            ],
            "expected_outcomes": {
                "user_recognized": True,
                "context_retained": True,
                "progress_calculated": True,
                "no_repeated_questions": True
            },
            "evaluation_criteria": {
                "task_completion_rate": 1.0,
                "state_preserved": True,
                "memory_read_success": True
            }
        }
    
    @staticmethod
    def test_case_5_system_degradation() -> Dict[str, Any]:
        """Test Case 5: System Degradation"""
        return {
            "id": "TC005",
            "name": "System Degradation",
            "description": "RAG service temporarily unavailable, system degrades gracefully",
            "scenario": {
                "user_type": "authenticated",
                "user_id": "guest",
                "simulated_failures": ["rag_unavailable"]
            },
            "conversation_flow": [
                {"user": "What's the best way to save for retirement?", "expected_intent": "financial"}
            ],
            "expected_outcomes": {
                "fallback_activated": True,
                "user_informed": True,
                "cached_or_general_advice_provided": True,
                "no_hallucination": True
            },
            "evaluation_criteria": {
                "task_completion_rate": 0.7,
                "hallucination_frequency": 0.0,
                "groundedness_score": 0.8
            }
        }

class TestRunner:
    """Runs test cases and collects results"""
    
    def __init__(self, orchestrator, evaluator):
        self.orchestrator = orchestrator
        self.evaluator = evaluator
        self.results = []
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test cases"""
        test_cases = TestCases.get_test_cases()
        
        for test_case in test_cases:
            print(f"\n{'='*60}")
            print(f"Running Test: {test_case['name']}")
            print(f"{'='*60}")
            
            result = self.run_test(test_case)
            self.results.append(result)
            
            print(f"\n✅ Test Complete")
            print(f"   Metrics: {json.dumps(result.get('metrics', {}), indent=2)}")
        
        return self.summarize_results()
    
    def run_test(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case"""
        execution_logs = []
        
        # Setup test scenario
        self._setup_test_scenario(test_case["scenario"])
        
        # Run conversation flow
        for turn in test_case["conversation_flow"]:
            user_input = turn["user"]
            expected_intent = turn["expected_intent"]
            
            # Process through orchestrator
            response = self.orchestrator.process(user_input)
            
            # Log the turn
            execution_logs.append({
                "type": "turn",
                "user": user_input,
                "agent": response,
                "expected_intent": expected_intent,
                "timestamp": datetime.now().isoformat()
            })
        
        # Evaluate results
        metrics = self.evaluator.evaluate_test_case(
            test_case_id=test_case["id"],
            execution_logs=execution_logs
        )
        
        # Check expected outcomes
        outcomes_met = self._check_outcomes(
            test_case["expected_outcomes"],
            execution_logs
        )
        
        return {
            "test_case": test_case["id"],
            "name": test_case["name"],
            "metrics": metrics,
            "outcomes_met": outcomes_met,
            "logs": execution_logs
        }
    
    def _setup_test_scenario(self, scenario: Dict):
        """Setup test scenario"""
        if scenario.get("user_type") == "authenticated":
            self.orchestrator.login(scenario.get("user_id", "test_user"))
        else:
            self.orchestrator.logout()
    
    def _check_outcomes(self, expected: Dict, logs: List) -> Dict:
        """Check if expected outcomes were met"""
        results = {}
        
        for outcome, expected_value in expected.items():
            if isinstance(expected_value, bool):
                # Check if outcome occurred in logs
                occurred = any(
                    outcome in str(log) for log in logs
                )
                results[outcome] = occurred == expected_value
            else:
                # For non-boolean outcomes, just record
                results[outcome] = "check_manually"
        
        return results
    
    def summarize_results(self) -> Dict[str, Any]:
        """Summarize all test results"""
        summary = {
            "total_tests": len(self.results),
            "passed": 0,
            "failed": 0,
            "average_metrics": {},
            "details": []
        }
        
        # Aggregate metrics
        all_metrics = {}
        for result in self.results:
            if all(result.get("outcomes_met", {}).values()):
                summary["passed"] += 1
            else:
                summary["failed"] += 1
            
            for metric, value in result.get("metrics", {}).items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
            
            summary["details"].append({
                "test": result["name"],
                "metrics": result["metrics"],
                "outcomes": result["outcomes_met"]
            })
        
        # Calculate averages
        for metric, values in all_metrics.items():
            summary["average_metrics"][metric] = sum(values) / len(values)
        
        return summary