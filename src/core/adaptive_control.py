"""
Adaptive Control for CoFina - Lightweight performance monitoring.

This module tracks metrics and can trigger warnings/adaptations when
thresholds are crossed. For MVP, it's primarily observational — full
closed-loop adaptation can be added in Phase 2.
"""

from __future__ import annotations

from typing import Any, Dict


class AdaptiveController:
    """
    Monitors performance metrics and flags issues.
    """

    def __init__(self) -> None:
        # Thresholds for triggering warnings/adaptations
        self.thresholds = {
            "min_groundedness": 0.7,
            "min_tool_success_rate": 0.8,
            "max_iterations": 5,
        }

    def check_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check metrics against thresholds and return warnings/recommendations.

        Args:
            metrics: Current performance metrics (e.g. from EvaluationMetrics)

        Returns:
            {"status": "ok" | "warning", "issues": [...], "recommendations": [...]}
        """
        issues = []
        recommendations = []

        # Check groundedness
        score = metrics.get("avg_verification_score", 1.0)
        if score < self.thresholds["min_groundedness"]:
            issues.append(f"Low groundedness: {score:.2f}")
            recommendations.append("Consider re-retrieving with more context")

        # Check tool success rate
        success_rate = metrics.get("tool_success_rate", 1.0)
        if success_rate < self.thresholds["min_tool_success_rate"]:
            issues.append(f"Tool failure rate high: {1-success_rate:.2%}")
            recommendations.append("Check tool implementations and retry logic")

        # Check iteration count (for planning loops)
        iterations = metrics.get("iterations", 0)
        if iterations > self.thresholds["max_iterations"]:
            issues.append(f"Excessive iterations: {iterations}")
            recommendations.append("Simplify task or escalate to human")

        return {
            "status": "warning" if issues else "ok",
            "issues": issues,
            "recommendations": recommendations,
        }

    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """Dynamically adjust thresholds based on observed performance."""
        self.thresholds.update(new_thresholds)