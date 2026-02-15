"""
Simple caching utility for RAG and other expensive operations
"""

import hashlib
import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, List
import pickle

class SimpleCache:
    """Simple file-based cache with TTL"""
    
    def __init__(self, cache_dir: str = "cache", default_ttl_hours: int = 24):
        self.cache_dir = cache_dir
        self.default_ttl = timedelta(hours=default_ttl_hours)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "embeddings"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "rag"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "responses"), exist_ok=True)
    
    def _get_cache_path(self, key: str, subdir: str = "") -> str:
        """Get cache file path for a key"""
        hashed = hashlib.md5(key.encode()).hexdigest()
        
        if subdir:
            full_dir = os.path.join(self.cache_dir, subdir)
            os.makedirs(full_dir, exist_ok=True)
            return os.path.join(full_dir, f"{hashed}.json")
        else:
            return os.path.join(self.cache_dir, f"{hashed}.json")
    
    def get(self, key: str, subdir: str = "") -> Optional[Any]:
        """Get value from cache if not expired"""
        cache_path = self._get_cache_path(key, subdir)
        
        if not os.path.exists(cache_path):
            return None
        
        try:
            with open(cache_path, 'r') as f:
                data = json.load(f)
            
            # Check expiration
            cached_time = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - cached_time > self.default_ttl:
                os.remove(cache_path)  # Remove expired cache
                return None
            
            return data['value']
        except Exception as e:
            print(f"Cache read error: {e}")
            return None
    
    def set(self, key: str, value: Any, subdir: str = ""):
        """Set value in cache"""
        cache_path = self._get_cache_path(key, subdir)
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'key': key,
            'value': value
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Cache write error: {e}")
    
    def get_or_set(self, key: str, func, *args, subdir: str = "", **kwargs) -> Any:
        """Get from cache or compute and store"""
        cached = self.get(key, subdir)
        if cached is not None:
            return cached
        
        result = func(*args, **kwargs)
        self.set(key, result, subdir)
        return result
    
    def clear_expired(self, subdir: str = ""):
        """Clear all expired cache entries"""
        target_dir = os.path.join(self.cache_dir, subdir) if subdir else self.cache_dir
        if not os.path.exists(target_dir):
            return
        
        now = datetime.now()
        cleared = 0
        
        for filename in os.listdir(target_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(target_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                cached_time = datetime.fromisoformat(data['timestamp'])
                if now - cached_time > self.default_ttl:
                    os.remove(filepath)
                    cleared += 1
            except Exception:
                # If we can't read it, delete it
                os.remove(filepath)
                cleared += 1
        
        if cleared > 0:
            print(f"🧹 Cleared {cleared} expired cache entries from {target_dir}")
    
    def clear_all(self, subdir: str = ""):
        """Clear all cache entries in a subdirectory"""
        target_dir = os.path.join(self.cache_dir, subdir) if subdir else self.cache_dir
        if not os.path.exists(target_dir):
            return
        
        cleared = 0
        for filename in os.listdir(target_dir):
            if filename.endswith('.json'):
                os.remove(os.path.join(target_dir, filename))
                cleared += 1
        
        if cleared > 0:
            print(f"🧹 Cleared {cleared} cache entries from {target_dir}")
    
    def get_stats(self, subdir: str = "") -> Dict[str, Any]:
        """Get cache statistics"""
        target_dir = os.path.join(self.cache_dir, subdir) if subdir else self.cache_dir
        if not os.path.exists(target_dir):
            return {"entry_count": 0, "total_size_bytes": 0, "total_size_mb": 0}
        
        cache_files = [f for f in os.listdir(target_dir) if f.endswith('.json')]
        
        total_size = 0
        for f in cache_files:
            path = os.path.join(target_dir, f)
            total_size += os.path.getsize(path)
        
        # Get age info
        ages = []
        for f in cache_files[:10]:  # Check first 10 files
            try:
                path = os.path.join(target_dir, f)
                with open(path, 'r') as file:
                    data = json.load(file)
                    cached_time = datetime.fromisoformat(data['timestamp'])
                    age_hours = (datetime.now() - cached_time).total_seconds() / 3600
                    ages.append(age_hours)
            except:
                pass
        
        avg_age = sum(ages) / len(ages) if ages else 0
        
        return {
            "entry_count": len(cache_files),
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "avg_age_hours": round(avg_age, 1),
            "cache_dir": target_dir
        }

# Global cache instance
_global_cache = None

def get_cache():
    """Get or create global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SimpleCache()
    return _global_cache

# Decorator for easy caching
def cached(ttl_hours: int = 24, subdir: str = ""):
    """Decorator to cache function results"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_parts = [func.__name__]
            key_parts.extend([str(arg) for arg in args])
            key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
            cache_key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            cache = get_cache()
            # Temporarily use different TTL
            original_ttl = cache.default_ttl
            cache.default_ttl = timedelta(hours=ttl_hours)
            
            try:
                result = cache.get_or_set(cache_key, func, *args, subdir=subdir, **kwargs)
            finally:
                cache.default_ttl = original_ttl
            
            return result
        return wrapper
    return decorator