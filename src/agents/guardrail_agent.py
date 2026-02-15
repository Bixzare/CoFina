"""
Guardrail Agent - Handles security, authentication, and safety
"""

from typing import Dict, Any, Optional, List
import re
import hashlib
from datetime import datetime, timedelta

class GuardrailAgent:
    """
    Specialized agent for security, authentication, and content safety
    """
    
    def __init__(self):
        self.sessions = {}
        self.blocked_patterns = [
            r"DROP\s+TABLE",
            r"DELETE\s+FROM",
            r"UPDATE\s+.*SET",
            r"INSERT\s+INTO",
            r"ALTER\s+TABLE",
            r"TRUNCATE\s+TABLE",
            r"--",  # SQL comment
            r";\s*$"  # SQL injection
        ]
        
        self.sensitive_patterns = [
            r"\b\d{16}\b",  # Credit card
            r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
            r"password[=:]\s*\S+",
            r"api[_-]key[=:]\s*\S+"
        ]
    
    def process(self, query: str, session_id: str, 
                 user_id: str = "guest") -> Dict[str, Any]:
        """
        Process guardrail checks for a query
        
        Args:
            query: User query
            session_id: Session identifier
            user_id: User identifier
        
        Returns:
            Guardrail assessment
        """
        results = {
            "passed": True,
            "session_valid": False,
            "injection_risk": 0.0,
            "sensitive_data_found": False,
            "auth_status": "unknown",
            "warnings": [],
            "actions": []
        }
        
        # Check session
        session_check = self._check_session(session_id, user_id)
        results["session_valid"] = session_check["valid"]
        results["auth_status"] = session_check["status"]
        
        if not session_check["valid"] and user_id != "guest":
            results["passed"] = False
            results["warnings"].append("Session expired or invalid")
            results["actions"].append("reauthenticate")
        
        # Check for prompt injection
        injection_check = self._check_injection(query)
        results["injection_risk"] = injection_check["risk"]
        
        if injection_check["risk"] > 0.7:
            results["passed"] = False
            results["warnings"].append("High risk of prompt injection detected")
            results["actions"].append("block")
        elif injection_check["risk"] > 0.4:
            results["warnings"].append("Possible injection attempt detected")
            results["actions"].append("scrutinize")
        
        # Check for sensitive data
        sensitive_check = self._check_sensitive_data(query)
        results["sensitive_data_found"] = sensitive_check["found"]
        
        if sensitive_check["found"]:
            results["warnings"].extend(sensitive_check["warnings"])
            results["actions"].append("redact")
        
        # Check authorization for personalized actions
        if self._needs_auth(query) and user_id == "guest":
            results["passed"] = False
            results["warnings"].append("Authentication required for this request")
            results["actions"].append("authenticate")
        
        return results
    
    def _check_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        """Check if session is valid"""
        now = datetime.now()
        
        if session_id in self.sessions:
            session = self.sessions[session_id]
            
            # Check expiration (30 minutes default)
            if now - session["created"] > timedelta(minutes=30):
                return {"valid": False, "status": "expired"}
            
            # Update last activity
            session["last_activity"] = now
            
            return {
                "valid": True,
                "status": "authenticated" if session.get("authenticated") else "guest",
                "user_id": session.get("user_id", "guest")
            }
        else:
            # Create new session
            self.sessions[session_id] = {
                "created": now,
                "last_activity": now,
                "authenticated": user_id != "guest",
                "user_id": user_id
            }
            
            return {
                "valid": True,
                "status": "authenticated" if user_id != "guest" else "guest",
                "user_id": user_id
            }
    
    def _check_injection(self, query: str) -> Dict[str, float]:
        """Check for prompt injection attempts"""
        risk = 0.0
        reasons = []
        
        # Check for SQL injection patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                risk += 0.5
                reasons.append(f"SQL pattern: {pattern}")
        
        # Check for prompt injection attempts
        injection_patterns = [
            (r"ignore previous instructions", 0.4),
            (r"forget (everything|all)", 0.3),
            (r"you are now", 0.2),
            (r"act as (?!a financial assistant)", 0.3),
            (r"system prompt", 0.2),
            (r"bypass", 0.3),
            (r"jailbreak", 0.5)
        ]
        
        for pattern, weight in injection_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                risk += weight
                reasons.append(f"Injection pattern: {pattern}")
        
        # Check for excessive length (potential DoS)
        if len(query) > 5000:
            risk += 0.2
        
        # Check for encoded content
        if re.search(r'%[0-9A-Fa-f]{2}', query):  # URL encoded
            risk += 0.1
        
        return {
            "risk": min(risk, 1.0),
            "reasons": reasons[:3]  # Top 3 reasons
        }
    
    def _check_sensitive_data(self, query: str) -> Dict[str, Any]:
        """Check for sensitive data in query"""
        found = False
        warnings = []
        
        for pattern in self.sensitive_patterns:
            matches = re.findall(pattern, query)
            if matches:
                found = True
                warnings.append(f"Potential sensitive data detected")
                # In production, would redact here
        
        return {
            "found": found,
            "warnings": warnings
        }
    
    def _needs_auth(self, query: str) -> bool:
        """Check if query requires authentication"""
        auth_required_patterns = [
            r"my (plan|profile|preferences)",
            r"my (spending|transactions|balance)",
            r"update (my|profile)",
            r"create (plan|goal)",
            r"show (me )?my"
        ]
        
        for pattern in auth_required_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        
        return False
    
    def authenticate_session(self, session_id: str, user_id: str):
        """Mark session as authenticated"""
        if session_id in self.sessions:
            self.sessions[session_id]["authenticated"] = True
            self.sessions[session_id]["user_id"] = user_id
    
    def end_session(self, session_id: str):
        """End a session (logout)"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def redact_pii(self, text: str) -> str:
        """Redact personally identifiable information"""
        # Redact emails
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[EMAIL REDACTED]', text)
        
        # Redact phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE REDACTED]', text)
        
        # Redact SSNs
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REDACTED]', text)
        
        # Redact credit cards
        text = re.sub(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', 
                     '[CREDIT CARD REDACTED]', text)
        
        return text