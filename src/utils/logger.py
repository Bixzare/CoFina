"""
Logger for CoFina - Ensures trace logs are saved properly
"""

import json
import os
from datetime import datetime
from typing import Any, Dict
from pathlib import Path

class AgentLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"trace_{timestamp}.json"
        self.current_turn_id = None
        
        self._write_entry({
            "event": "session_start",
            "timestamp": datetime.now().isoformat()
        })
        print(f"📝 Logging to: {self.log_file}")  # Debug print
    
    def start_turn(self, user_query: str):
        self.current_turn_id = datetime.now().strftime("%H%M%S_%f")
        self._write_entry({
            "event": "turn_start",
            "turn_id": self.current_turn_id,
            "content": user_query,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_step(self, step_type: str, content: Any):
        self._write_entry({
            "event": "step",
            "turn_id": self.current_turn_id,
            "step_type": step_type,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_retrieval(self, query: str, chunks: list):
        sanitized = []
        for c in chunks:
            sanitized.append({
                "source": c.metadata.get("source", "unknown"),
                "chunk_id": c.metadata.get("chunk_id", "unknown"),
            })
        self.log_step("retrieval", {"query": query, "chunks": sanitized})
    
    def end_turn(self, final_answer: str):
        self._write_entry({
            "event": "turn_end",
            "turn_id": self.current_turn_id,
            "content": final_answer,
            "timestamp": datetime.now().isoformat()
        })
        self.current_turn_id = None
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Write entry to log file"""
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
            f.flush()  # Force write to disk