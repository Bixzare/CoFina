"""
Memory Manager for CoFina - Handles memory read/write with policies and pruning
"""

import json
import sqlite3
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import numpy as np
import os
class MemoryManager:
    """
    Manages agent memory with read/write policies and pruning strategies
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
        
        self.short_term = deque(maxlen=50)  # Recent interactions
        self.working_memory = {}  # Current context
        self._init_memory_tables()
        
        # Memory policies
        self.write_policy = {
            "user_provides_personal_info": 0.9,  # Importance score
            "user_confirms_plan": 0.8,
            "goal_created_or_updated": 0.85,
            "significant_transaction": 0.7,
            "user_correction": 0.95,
            "conversational": 0.3,
            "tool_output": 0.5
        }
        
        self.read_policy = {
            "personalization": ["user_profile", "recent_goals", "preferences"],
            "planning": ["financial_history", "similar_situations", "constraints"],
            "reminder": ["unmet_goals", "deadlines", "commitments"],
            "default": ["short_term", "key_facts"]
        }
    
    def _init_memory_tables(self):
        """Initialize memory storage tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS long_term_memory (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    memory_type TEXT,
                    content TEXT,
                    importance REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_index (
                    memory_id TEXT,
                    keyword TEXT,
                    relevance REAL,
                    PRIMARY KEY (memory_id, keyword)
                )
            """)
    
    def write(self, user_id: str, memory_type: str, content: Any, 
              context: str, importance_override: Optional[float] = None) -> str:
        """
        Write to memory with importance scoring
        
        Args:
            user_id: User identifier
            memory_type: Type of memory (personal_info, goal, plan, etc.)
            content: Memory content
            context: Context of the memory (what triggered it)
            importance_override: Optional manual importance score
        
        Returns:
            memory_id: Unique identifier for the memory
        """
        # Determine importance
        if importance_override is not None:
            importance = importance_override
        else:
            importance = self.write_policy.get(memory_type, 0.5)
        
        # Generate memory ID
        memory_id = hashlib.md5(
            f"{user_id}:{memory_type}:{datetime.now().isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Store in long-term if important enough
        if importance > 0.6:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO long_term_memory 
                    (memory_id, user_id, memory_type, content, importance, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    memory_id, user_id, memory_type, 
                    json.dumps(content) if not isinstance(content, str) else content,
                    importance, json.dumps({"context": context})
                ))
                
                # Index keywords for retrieval
                self._index_memory(conn, memory_id, content)
        
        # Always add to short-term
        self.short_term.append({
            "memory_id": memory_id,
            "type": memory_type,
            "content": content,
            "importance": importance,
            "timestamp": datetime.now().isoformat(),
            "context": context
        })
        
        return memory_id
    
    def read(self, user_id: str, query: str, purpose: str = "default", 
             limit: int = 10) -> List[Dict]:
        """
        Read from memory based on purpose
        
        Args:
            user_id: User identifier
            query: Search query
            purpose: Reading purpose (personalization, planning, reminder, default)
            limit: Maximum results
        
        Returns:
            List of relevant memories
        """
        memories = []
        
        # Get memory types to retrieve based on purpose
        memory_types = self.read_policy.get(purpose, self.read_policy["default"])
        
        # 1. Check short-term first (recency bias)
        short_term_results = []
        for mem in reversed(self.short_term):  # Most recent first
            if self._relevance_score(mem["content"], query) > 0.3:
                short_term_results.append(mem)
                if len(short_term_results) >= limit // 2:
                    break
        
        # 2. Check long-term for important memories
        long_term_results = []
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ",".join(["?"] * len(memory_types))
            cur = conn.execute(f"""
                SELECT memory_id, memory_type, content, importance, 
                       access_count, created_at
                FROM long_term_memory 
                WHERE user_id = ? AND memory_type IN ({placeholders})
                ORDER BY importance DESC, last_accessed DESC
                LIMIT ?
            """, (user_id, *memory_types, limit))
            
            for row in cur.fetchall():
                memory = {
                    "memory_id": row[0],
                    "type": row[1],
                    "content": json.loads(row[2]) if row[2].startswith("{") else row[2],
                    "importance": row[3],
                    "access_count": row[4],
                    "created_at": row[5]
                }
                if self._relevance_score(memory["content"], query) > 0.2:
                    long_term_results.append(memory)
                    
                    # Update access stats
                    conn.execute("""
                        UPDATE long_term_memory 
                        SET last_accessed = CURRENT_TIMESTAMP, 
                            access_count = access_count + 1
                        WHERE memory_id = ?
                    """, (memory["memory_id"],))
        
        # Merge and deduplicate
        seen_ids = set()
        merged = []
        
        for mem in short_term_results + long_term_results:
            mem_id = mem.get("memory_id", hashlib.md5(str(mem["content"]).encode()).hexdigest()[:16])
            if mem_id not in seen_ids:
                seen_ids.add(mem_id)
                merged.append(mem)
        
        return merged[:limit]
    
    def prune(self, older_than_days: int = 30, min_importance: float = 0.3):
        """
        Prune old, low-importance memories
        
        Args:
            older_than_days: Prune memories older than this
            min_importance: Minimum importance to keep
        """
        cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Archive or delete old memories
            cur = conn.execute("""
                SELECT memory_id FROM long_term_memory 
                WHERE created_at < ? AND importance < ?
            """, (cutoff_date, min_importance))
            
            to_prune = cur.fetchall()
            
            for (memory_id,) in to_prune:
                # Archive before deleting (optional)
                conn.execute("""
                    INSERT INTO memory_archive SELECT * FROM long_term_memory 
                    WHERE memory_id = ?
                """, (memory_id,))
                conn.execute("DELETE FROM long_term_memory WHERE memory_id = ?", (memory_id,))
                conn.execute("DELETE FROM memory_index WHERE memory_id = ?", (memory_id,))
        
        # Prune short-term (already limited by deque maxlen)
        self.short_term = deque(
            [m for m in self.short_term if 
             datetime.fromisoformat(m["timestamp"]) > datetime.now() - timedelta(days=7)],
            maxlen=50
        )
    
    def summarize_old_memories(self, user_id: str, older_than_days: int = 60):
        """
        Summarize very old memories to save space
        
        Args:
            user_id: User identifier
            older_than_days: Summarize memories older than this
        """
        cutoff_date = (datetime.now() - timedelta(days=older_than_days)).isoformat()
        
        with sqlite3.connect(self.db_path) as conn:
            # Get old memories
            cur = conn.execute("""
                SELECT memory_id, memory_type, content, importance
                FROM long_term_memory 
                WHERE user_id = ? AND created_at < ?
                ORDER BY created_at
            """, (user_id, cutoff_date))
            
            old_memories = cur.fetchall()
            
            if len(old_memories) > 20:  # Only summarize if enough memories
                # Group by type
                by_type = {}
                for mem in old_memories:
                    mem_type = mem[1]
                    if mem_type not in by_type:
                        by_type[mem_type] = []
                    by_type[mem_type].append(mem)
                
                # Create summary for each type
                for mem_type, memories in by_type.items():
                    summary = self._create_summary(memories)
                    
                    # Store summary
                    summary_id = hashlib.md5(
                        f"{user_id}:summary:{mem_type}:{datetime.now().isoformat()}".encode()
                    ).hexdigest()[:16]
                    
                    conn.execute("""
                        INSERT INTO long_term_memory 
                        (memory_id, user_id, memory_type, content, importance, metadata)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (
                        summary_id, user_id, f"summary_{mem_type}",
                        json.dumps(summary), 0.7,
                        json.dumps({"summarized_count": len(memories)})
                    ))
                    
                    # Delete original memories
                    for mem in memories:
                        conn.execute("DELETE FROM long_term_memory WHERE memory_id = ?", (mem[0],))
    
    def _index_memory(self, conn, memory_id: str, content: Any):
        """Index memory for keyword search"""
        if isinstance(content, dict):
            text = " ".join(str(v) for v in content.values())
        else:
            text = str(content)
        
        # Simple keyword extraction
        words = set(text.lower().split())
        for word in list(words)[:20]:  # Limit to 20 keywords
            if len(word) > 3:  # Ignore short words
                conn.execute(
                    "INSERT INTO memory_index (memory_id, keyword, relevance) VALUES (?, ?, ?)",
                    (memory_id, word, 1.0)
                )
    
    def _relevance_score(self, memory_content: Any, query: str) -> float:
        """Compute relevance score between memory and query"""
        if isinstance(memory_content, dict):
            mem_str = " ".join(str(v) for v in memory_content.values())
        else:
            mem_str = str(memory_content)
        
        query_words = set(query.lower().split())
        mem_words = set(mem_str.lower().split())
        
        if not query_words or not mem_words:
            return 0.0
        
        intersection = query_words.intersection(mem_words)
        return len(intersection) / len(query_words)
    
    def _create_summary(self, memories: List[tuple]) -> Dict:
        """Create a summary of multiple memories"""
        summary = {
            "count": len(memories),
            "types": list(set(m[1] for m in memories)),
            "earliest": memories[0][4] if len(memories[0]) > 4 else None,
            "latest": memories[-1][4] if len(memories[-1]) > 4 else None,
            "key_points": []
        }
        
        # Extract key points (simplified - in production would use LLM)
        for mem in memories[:5]:  # Top 5 by importance
            content = json.loads(mem[2]) if mem[2].startswith("{") else mem[2]
            summary["key_points"].append(str(content)[:100])
        
        return summary