"""
HTN Planner - Hierarchical Task Network for agent's own tasks
"""

from typing import Dict, Any, List, Optional
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class HTNPlanner:
    """
    Plans the agent's own task decomposition
    HTN = Hierarchical Task Network - breaks complex agent tasks into subtasks
    """
    
    def __init__(self):
        self.task_library = {
            "handle_user_query": self._decompose_handle_query,
            "authenticate_user": self._decompose_authenticate,
            "retrieve_information": self._decompose_retrieve,
            "generate_response": self._decompose_generate_response
        }
    
    def decompose(self, task: str, context: Dict) -> List[Dict]:
        """Break down a high-level agent task into subtasks"""
        if task in self.task_library:
            return self.task_library[task](context)
        return [{"name": task, "status": TaskStatus.PENDING}]
    
    def _decompose_handle_query(self, context: Dict) -> List[Dict]:
        """Handle user query task decomposition"""
        return [
            {"name": "classify_intent", "tool": "intent_classifier", "status": TaskStatus.PENDING},
            {"name": "check_authentication", "tool": "get_user_status", "status": TaskStatus.PENDING},
            {"name": "route_to_specialist", "tool": None, "status": TaskStatus.PENDING},
            {"name": "execute_specialist", "tool": "delegate", "status": TaskStatus.PENDING},
            {"name": "format_response", "tool": None, "status": TaskStatus.PENDING}
        ]
    
    def _decompose_authenticate(self, context: Dict) -> List[Dict]:
        """Authentication task decomposition"""
        return [
            {"name": "extract_credentials", "tool": None, "status": TaskStatus.PENDING},
            {"name": "validate_user_id", "tool": "check_user_exists", "status": TaskStatus.PENDING},
            {"name": "verify_password", "tool": "authenticate_user", "status": TaskStatus.PENDING},
            {"name": "load_user_data", "tool": "get_user_info", "status": TaskStatus.PENDING}
        ]
    
    def _decompose_retrieve(self, context: Dict) -> List[Dict]:
        """Information retrieval task decomposition"""
        return [
            {"name": "check_cache", "tool": "get_cached", "status": TaskStatus.PENDING},
            {"name": "query_rag", "tool": "search_documents", "status": TaskStatus.PENDING},
            {"name": "query_database", "tool": "query_db", "status": TaskStatus.PENDING},
            {"name": "merge_results", "tool": None, "status": TaskStatus.PENDING}
        ]
    
    def _decompose_generate_response(self, context: Dict) -> List[Dict]:
        """Response generation task decomposition"""
        return [
            {"name": "get_template", "tool": None, "status": TaskStatus.PENDING},
            {"name": "fill_template", "tool": None, "status": TaskStatus.PENDING},
            {"name": "verify_groundedness", "tool": "verify_response", "status": TaskStatus.PENDING},
            {"name": "add_citations", "tool": None, "status": TaskStatus.PENDING}
        ]
    
    def get_next_task(self, current_tasks: List[Dict]) -> Optional[Dict]:
        """Get the next executable task based on dependencies"""
        completed = {t["name"] for t in current_tasks if t["status"] == TaskStatus.COMPLETED}
        
        for task in current_tasks:
            if task["status"] != TaskStatus.PENDING:
                continue
            
            dependencies = task.get("dependencies", [])
            if all(dep in completed for dep in dependencies):
                return task
        
        return None