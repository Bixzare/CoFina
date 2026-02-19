"""
State Manager for CoFina - Handles persistent state across sessions with explicit schemas
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
from enum import Enum
import hashlib

class SessionPhase(Enum):
    GUEST = "guest"
    REGISTRATION = "registration"
    AUTHENTICATED = "authenticated"
    PLANNING = "planning"
    EXECUTION = "execution"
    REVIEW = "review"

class StateSchema:
    """Schema definitions for all state types"""
    
    CONVERSATION_SCHEMA = {
        "session_id": str,
        "user_id": (str, type(None)),
        "turn_count": int,
        "current_phase": str,
        "last_interaction": str,
        "constraints": list
    }
    
    TASK_SCHEMA = {
        "objective": str,
        "subtasks": list,
        "dependencies": dict,
        "completion_criteria": list
    }
    
    WORLD_SCHEMA = {
        "tool_outputs": dict,
        "external_facts": dict,
        "last_update": str
    }
    
    INTERNAL_SCHEMA = {
        "assumptions": list,
        "reasoning_scratchpad": list,
        "decision_trace": list,
        "confidence_scores": dict
    }

class StateManager:
    """
    Manages all agent state with persistence, validation, and checkpointing
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
            
        self._init_state_tables()
        self.current_state = self._create_empty_state()
        self.checkpoint_counter = 0
        
    def _init_state_tables(self):
        """Initialize state persistence tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    turn_count INTEGER,
                    current_phase TEXT,
                    last_interaction TEXT,
                    constraints TEXT,
                    checkpoint_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS task_state (
                    session_id TEXT,
                    objective TEXT,
                    subtasks TEXT,
                    dependencies TEXT,
                    completion_criteria TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, objective)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    state_snapshot TEXT,
                    checkpoint_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def _create_empty_state(self) -> Dict[str, Any]:
        """Create empty state structure"""
        return {
            "conversation": {
                "session_id": None,
                "user_id": "guest",
                "turn_count": 0,
                "current_phase": "guest",
                "last_interaction": datetime.now().isoformat(),
                "constraints": []
            },
            "task": {
                "objective": None,
                "subtasks": [],
                "dependencies": {},
                "completion_criteria": []
            },
            "world": {
                "tool_outputs": {},
                "external_facts": {},
                "last_update": datetime.now().isoformat()
            },
            "internal": {
                "assumptions": [],
                "reasoning_scratchpad": [],
                "decision_trace": [],
                "confidence_scores": {}
            }
        }
    
    def validate_state(self, state: Dict[str, Any]) -> bool:
        """Validate state against schemas"""
        try:
            # Check conversation state
            conv = state.get("conversation", {})
            for key, expected_type in StateSchema.CONVERSATION_SCHEMA.items():
                if key not in conv:
                    return False
                if not isinstance(conv[key], expected_type):
                    return False
            
            # Check task state
            task = state.get("task", {})
            for key, expected_type in StateSchema.TASK_SCHEMA.items():
                if key not in task:
                    return False
                if not isinstance(task[key], expected_type):
                    return False
            
            return True
        except Exception:
            return False
    
    def checkpoint(self, reason: str = "periodic") -> str:
        """Create a checkpoint of current state"""
        self.checkpoint_counter += 1
        checkpoint_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.checkpoint_counter}"
        
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "conversation": self.current_state["conversation"],
            "task": self.current_state["task"],
            "world": {k: v for k, v in self.current_state["world"].items() 
                     if k != "tool_outputs"},  # Skip large tool outputs
            "checkpoint_reason": reason
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO checkpoints (checkpoint_id, session_id, state_snapshot, checkpoint_reason) VALUES (?, ?, ?, ?)",
                (checkpoint_id, self.current_state["conversation"]["session_id"], 
                 json.dumps(snapshot), reason)
            )
        
        return checkpoint_id
    
    def restore_from_checkpoint(self, checkpoint_id: str) -> bool:
        """Restore state from a checkpoint"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT state_snapshot FROM checkpoints WHERE checkpoint_id = ?",
                (checkpoint_id,)
            )
            row = cur.fetchone()
            if row:
                snapshot = json.loads(row[0])
                self.current_state["conversation"] = snapshot["conversation"]
                self.current_state["task"] = snapshot["task"]
                # Don't restore world state fully to avoid stale data
                return True
        return False
    
    def update_conversation(self, session_id: str, user_id: str, phase: SessionPhase):
        """Update conversation state"""
        self.current_state["conversation"]["session_id"] = session_id
        self.current_state["conversation"]["user_id"] = user_id
        self.current_state["conversation"]["current_phase"] = phase.value
        self.current_state["conversation"]["turn_count"] += 1
        self.current_state["conversation"]["last_interaction"] = datetime.now().isoformat()
        
        # Persist to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO conversation_state 
                (session_id, user_id, turn_count, current_phase, last_interaction, constraints)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                session_id, user_id, 
                self.current_state["conversation"]["turn_count"],
                phase.value,
                self.current_state["conversation"]["last_interaction"],
                json.dumps(self.current_state["conversation"]["constraints"])
            ))
    
    def update_task(self, objective: str, subtasks: List[Dict], completion_criteria: List[str]):
        """Update task state"""
        self.current_state["task"] = {
            "objective": objective,
            "subtasks": subtasks,
            "dependencies": self._build_dependencies(subtasks),
            "completion_criteria": completion_criteria
        }
    
    def update_world(self, tool_name: str, output: Any):
        """Update world state with tool output"""
        self.current_state["world"]["tool_outputs"][tool_name] = {
            "result": output,
            "timestamp": datetime.now().isoformat()
        }
        self.current_state["world"]["last_update"] = datetime.now().isoformat()
    
    def add_assumption(self, fact: str, confidence: float, source: str):
        """Add an assumption to internal state"""
        self.current_state["internal"]["assumptions"].append({
            "fact": fact,
            "confidence": confidence,
            "source": source,
            "timestamp": datetime.now().isoformat()
        })
    
    def add_decision(self, decision: str):
        """Add decision to trace"""
        self.current_state["internal"]["decision_trace"].append({
            "decision": decision,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_session_history(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Retrieve conversation history for a session"""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT state_snapshot FROM checkpoints 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (session_id, limit))
            return [json.loads(row[0]) for row in cur.fetchall()]
    
    def _build_dependencies(self, subtasks: List[Dict]) -> Dict:
        """Build dependency graph from subtasks"""
        dependencies = {}
        for i, task in enumerate(subtasks):
            deps = task.get("depends_on", [])
            dependencies[str(i)] = deps
        return dependencies
    
    def clear_session(self, session_id: str):
        """Clear all state for a session (logout)"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM conversation_state WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM task_state WHERE session_id = ?", (session_id,))
        self.current_state = self._create_empty_state()