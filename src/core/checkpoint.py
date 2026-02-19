"""
Checkpoint Manager for CoFina - Handles system checkpointing for recovery
"""

import json
import sqlite3
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import uuid

class CheckpointManager:
    """
    Manages system checkpoints for fault tolerance and recovery
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
            
        self._init_checkpoint_tables()
    
    def _init_checkpoint_tables(self):
        """Initialize checkpoint tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS system_checkpoints (
                    checkpoint_id TEXT PRIMARY KEY,
                    session_id TEXT,
                    user_id TEXT,
                    state_snapshot TEXT,
                    checkpoint_reason TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    restored_at TIMESTAMP,
                    metadata TEXT
                )
            """)
    
    def create_checkpoint(self, session_id: str, user_id: str, 
                          state: Dict[str, Any], 
                          reason: str = "periodic") -> str:
        """
        Create a new checkpoint
        
        Args:
            session_id: Current session ID
            user_id: Current user ID
            state: Current system state
            reason: Reason for checkpoint
        
        Returns:
            Checkpoint ID
        """
        checkpoint_id = f"ckpt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Create a serializable snapshot
        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "session_id": session_id,
            "user_id": user_id,
            "conversation": state.get("conversation", {}),
            "task": state.get("task", {}),
            "world": self._sanitize_world_state(state.get("world", {})),
            "checkpoint_reason": reason
        }
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO system_checkpoints 
                (checkpoint_id, session_id, user_id, state_snapshot, checkpoint_reason, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                checkpoint_id,
                session_id,
                user_id,
                json.dumps(snapshot),
                reason,
                json.dumps({"created_from": "checkpoint_manager"})
            ))
        
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """
        Restore state from a checkpoint
        
        Args:
            checkpoint_id: ID of checkpoint to restore
        
        Returns:
            Restored state or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT state_snapshot FROM system_checkpoints 
                WHERE checkpoint_id = ?
            """, (checkpoint_id,))
            
            row = cur.fetchone()
            if row:
                # Update restored_at timestamp
                conn.execute("""
                    UPDATE system_checkpoints 
                    SET restored_at = CURRENT_TIMESTAMP 
                    WHERE checkpoint_id = ?
                """, (checkpoint_id,))
                
                return json.loads(row[0])
        
        return None
    
    def get_latest_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the most recent checkpoint for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Latest checkpoint or None
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT checkpoint_id, state_snapshot, created_at 
                FROM system_checkpoints 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 1
            """, (session_id,))
            
            row = cur.fetchone()
            if row:
                return {
                    "checkpoint_id": row[0],
                    "state": json.loads(row[1]),
                    "created_at": row[2]
                }
        
        return None
    
    def list_checkpoints(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        List checkpoints for a session
        
        Args:
            session_id: Session ID
            limit: Maximum number of checkpoints to return
        
        Returns:
            List of checkpoint metadata
        """
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute("""
                SELECT checkpoint_id, checkpoint_reason, created_at, restored_at
                FROM system_checkpoints 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (session_id, limit))
            
            return [
                {
                    "checkpoint_id": row[0],
                    "reason": row[1],
                    "created_at": row[2],
                    "restored_at": row[3]
                }
                for row in cur.fetchall()
            ]
    
    def delete_old_checkpoints(self, days: int = 7):
        """
        Delete checkpoints older than specified days
        
        Args:
            days: Delete checkpoints older than this many days
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                DELETE FROM system_checkpoints 
                WHERE created_at < datetime('now', '-' || ? || ' days')
                AND restored_at IS NULL
            """, (days,))
    
    def _sanitize_world_state(self, world_state: Dict) -> Dict:
        """Remove large objects from world state for checkpointing"""
        sanitized = {}
        
        for key, value in world_state.items():
            if key == "tool_outputs":
                # Only keep metadata about tool outputs, not full outputs
                sanitized[key] = {
                    k: {
                        "timestamp": v.get("timestamp") if isinstance(v, dict) else None,
                        "size": len(str(v)) if v else 0
                    }
                    for k, v in value.items()
                }
            elif key == "external_facts":
                # Keep important facts but truncate if too large
                if isinstance(value, dict):
                    sanitized[key] = {k: v for k, v in value.items() 
                                     if len(str(v)) < 1000}
                else:
                    sanitized[key] = str(value)[:1000]
            else:
                # Keep other state as is if not too large
                if len(str(value)) < 10000:
                    sanitized[key] = value
                else:
                    sanitized[key] = f"[Truncated: {len(str(value))} chars]"
        
        return sanitized