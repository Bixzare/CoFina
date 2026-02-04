import json
import time
import os
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path

class AgentLogger:
    def __init__(self, log_dir: str = "logs"):
        """Initialize the agent logger."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a new log file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trace_{timestamp}.json"
        
        self.current_turn_id: Optional[str] = None
        self.session_id = timestamp
        
        # Initialize the log file with session start
        self._write_entry({
            "event": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat()
        })

    def start_turn(self, user_query: str):
        """Start a new turn with a user query."""
        self.current_turn_id = datetime.now().strftime("%H%M%S_%f")
        self._write_entry({
            "event": "turn_start",
            "turn_id": self.current_turn_id,
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })

    def log_step(self, step_type: str, content: Any, metadata: Dict[str, Any] = None):
        """Log a specific step (decision, tool call, etc)."""
        entry = {
            "event": "step",
            "turn_id": self.current_turn_id,
            "step_type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        if metadata:
            entry["metadata"] = metadata
        self._write_entry(entry)

    def log_decision(self, decision: str, reason: str):
        """Log a specialized routing decision."""
        self.log_step("decision", {"action": decision, "reason": reason})

    def log_retrieval(self, query: str, chunks: list):
        """Log retrieved chunks."""
        # Sanitize chunks to store just IDs and sources if possible, or truncate content
        sanitized_chunks = []
        for c in chunks:
            chunk_data = {
                "source": c.metadata.get("source", "unknown"),
                "chunk_id": c.metadata.get("chunk_id", "unknown"),
                "score": c.metadata.get("score", None)
            }
            sanitized_chunks.append(chunk_data)
            
        self.log_step("retrieval", {
            "query": query,
            "chunks": sanitized_chunks
        })

    def end_turn(self, final_answer: str):
        """End the current turn."""
        self._write_entry({
            "event": "turn_end",
            "turn_id": self.current_turn_id,
            "content": final_answer,
            "timestamp": datetime.now().isoformat()
        })
        self.current_turn_id = None

    def _write_entry(self, entry: Dict[str, Any]):
        """Append a JSON entry to the log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

# Global instance for easy import
# In a real app, you might want dependency injection or a singleton pattern
# For now, we'll instantiate when needed or rely on app.py to create one
