"""
Async processing for long-running operations
"""

import threading
import queue
import time
import uuid
from typing import Callable, Any, Optional, Dict
from datetime import datetime
import traceback

class AsyncProcessor:
    """Process long-running tasks asynchronously"""
    
    def __init__(self, max_workers: int = 2):
        self.task_queue = queue.Queue()
        self.results: Dict[str, Any] = {}
        self.status: Dict[str, str] = {}
        self.max_workers = max_workers
        self.workers = []
        self.running = True
        
        # Start worker threads
        for i in range(max_workers):
            worker = threading.Thread(target=self._worker, daemon=True, name=f"AsyncWorker-{i}")
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        """Worker thread to process tasks"""
        while self.running:
            try:
                task_id, func, args, kwargs, timeout = self.task_queue.get(timeout=1)
                
                self.status[task_id] = "running"
                
                try:
                    # Execute with timeout if specified
                    if timeout:
                        # Create a result container
                        result_container = []
                        error_container = []
                        
                        def target():
                            try:
                                result = func(*args, **kwargs)
                                result_container.append(result)
                            except Exception as e:
                                error_container.append(e)
                        
                        thread = threading.Thread(target=target)
                        thread.daemon = True
                        thread.start()
                        thread.join(timeout)
                        
                        if thread.is_alive():
                            # Thread still running after timeout
                            self.results[task_id] = {
                                'status': 'timeout',
                                'error': f'Task timed out after {timeout} seconds',
                                'time': time.time()
                            }
                            self.status[task_id] = "timeout"
                        elif error_container:
                            # Error occurred
                            self.results[task_id] = {
                                'status': 'failed',
                                'error': str(error_container[0]),
                                'traceback': traceback.format_exc(),
                                'time': time.time()
                            }
                            self.status[task_id] = "failed"
                        else:
                            # Success
                            self.results[task_id] = {
                                'status': 'completed',
                                'result': result_container[0],
                                'time': time.time()
                            }
                            self.status[task_id] = "completed"
                    else:
                        # No timeout
                        result = func(*args, **kwargs)
                        self.results[task_id] = {
                            'status': 'completed',
                            'result': result,
                            'time': time.time()
                        }
                        self.status[task_id] = "completed"
                        
                except Exception as e:
                    self.results[task_id] = {
                        'status': 'failed',
                        'error': str(e),
                        'traceback': traceback.format_exc(),
                        'time': time.time()
                    }
                    self.status[task_id] = "failed"
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Async worker error: {e}")
                traceback.print_exc()
    
    def submit(self, func: Callable, *args, timeout: Optional[float] = None, **kwargs) -> str:
        """
        Submit a task for async processing
        
        Args:
            func: Function to execute
            *args: Function arguments
            timeout: Timeout in seconds (optional)
            **kwargs: Function keyword arguments
        
        Returns:
            task_id: Unique task identifier
        """
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, func, args, kwargs, timeout))
        self.status[task_id] = "queued"
        return task_id
    
    def get_result(self, task_id: str, wait: bool = False, wait_timeout: float = 5.0) -> Optional[Any]:
        """
        Get result of a task
        
        Args:
            task_id: Task identifier
            wait: Whether to wait for completion
            wait_timeout: How long to wait in seconds
        
        Returns:
            Task result or None if not ready
        """
        if wait:
            start = time.time()
            while task_id not in self.results:
                if time.time() - start > wait_timeout:
                    return None
                time.sleep(0.1)
        
        return self.results.get(task_id)
    
    def get_status(self, task_id: str) -> str:
        """Get status of a task"""
        return self.status.get(task_id, "unknown")
    
    def is_ready(self, task_id: str) -> bool:
        """Check if task result is ready"""
        return task_id in self.results
    
    def wait_for_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """Wait for task to complete and return result"""
        start = time.time()
        while task_id not in self.results:
            if time.time() - start > timeout:
                return None
            time.sleep(0.1)
        
        result = self.results[task_id]
        if result['status'] == 'completed':
            return result['result']
        else:
            raise Exception(result.get('error', 'Task failed'))
    
    def cleanup_old_results(self, max_age_seconds: float = 3600):
        """Remove results older than max_age_seconds"""
        now = time.time()
        to_delete = []
        
        for task_id, result in self.results.items():
            if now - result.get('time', 0) > max_age_seconds:
                to_delete.append(task_id)
        
        for task_id in to_delete:
            del self.results[task_id]
            if task_id in self.status:
                del self.status[task_id]
    
    def shutdown(self):
        """Shutdown the async processor"""
        self.running = False
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)

# Global async processor
_async_processor = None

def get_async_processor():
    """Get or create global async processor"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncProcessor()
    return _async_processor

# Decorator for easy async execution
def async_task(timeout: Optional[float] = None):
    """Decorator to make a function run asynchronously"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            processor = get_async_processor()
            return processor.submit(func, *args, timeout=timeout, **kwargs)
        return wrapper
    return decorator

# Example usage class
class AsyncTask:
    """Wrapper for async tasks with callbacks"""
    
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.task_id = None
        self.callbacks = []
        self.error_callbacks = []
    
    def start(self, timeout: Optional[float] = None):
        """Start the async task"""
        processor = get_async_processor()
        self.task_id = processor.submit(self.func, *self.args, timeout=timeout, **self.kwargs)
        
        # Start monitoring thread
        monitor = threading.Thread(target=self._monitor, daemon=True)
        monitor.start()
        
        return self.task_id
    
    def _monitor(self):
        """Monitor task completion"""
        processor = get_async_processor()
        
        while not processor.is_ready(self.task_id):
            time.sleep(0.1)
        
        result = processor.get_result(self.task_id)
        
        if result['status'] == 'completed':
            for callback in self.callbacks:
                try:
                    callback(result['result'])
                except Exception as e:
                    print(f"Callback error: {e}")
        else:
            for callback in self.error_callbacks:
                try:
                    callback(result.get('error', 'Unknown error'))
                except Exception as e:
                    print(f"Error callback error: {e}")
    
    def on_complete(self, callback):
        """Add completion callback"""
        self.callbacks.append(callback)
        return self
    
    def on_error(self, callback):
        """Add error callback"""
        self.error_callbacks.append(callback)
        return self