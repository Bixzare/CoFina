"""
Guardrail Agent - Handles security, authentication, and content safety.

LLM-specific vulnerability checks added:
- Role hijacking / system prompt override
- Jailbreak escalation (DAN, character play, hypothetical framing)
- Indirect prompt injection (instructions embedded in quoted/pasted content)
- Token smuggling (encoding tricks: Base64, hex, unicode homoglyphs, zero-width chars)
- Excessive repetition / token-flooding (DoS)
"""

from typing import Dict, Any, List
import re
import base64
from datetime import datetime, timedelta


class GuardrailAgent:
    """
    Specialised agent for security, authentication, and content safety.
    """

    def __init__(self):
        self.sessions = {}

        # ── SQL injection patterns ────────────────────────────────────────
        self._sql_patterns = [
            r"DROP\s+TABLE",
            r"DELETE\s+FROM",
            r"UPDATE\s+.*SET",
            r"INSERT\s+INTO",
            r"ALTER\s+TABLE",
            r"TRUNCATE\s+TABLE",
            r"--",
            r";\s*$",
        ]

        # ── PII patterns ──────────────────────────────────────────────────
        self._sensitive_patterns = [
            r"\b\d{16}\b",                  # Credit card (no spaces)
            r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b",  # Credit card (spaced)
            r"\b\d{3}-\d{2}-\d{4}\b",       # SSN
            r"password\s*[=:]\s*\S+",
            r"api[_-]?key\s*[=:]\s*\S+",
        ]

        # ── LLM attack patterns (pattern, risk_weight, label) ────────────
        # Kept minimal: each entry earns its place with a distinct attack class.
        self._llm_attack_patterns: List[tuple] = [

            # 1. System prompt / context override
            (r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
             0.6, "system_override"),
            (r"(disregard|forget|override)\s+(your\s+)?(instructions?|rules?|guidelines?|constraints?|training)",
             0.5, "system_override"),
            (r"(new|updated?|real)\s+system\s+prompt",
             0.5, "system_override"),
            (r"you\s+are\s+now\s+(a\s+)?(?!CoFina|an?\s+intelligent|a\s+financial)",
             0.4, "role_hijack"),

            # 2. Jailbreak / character play escalation
            (r"\bDAN\b",                        0.7, "jailbreak"),  # "Do Anything Now"
            (r"jailbreak",                      0.7, "jailbreak"),
            (r"developer\s+mode",               0.5, "jailbreak"),
            (r"(pretend|imagine|roleplay|act)\s+(that\s+)?(you\s+)?(are|have\s+no)\s+(no\s+)?(restrictions?|limits?|guidelines?|rules?|filters?)",
             0.5, "jailbreak"),
            (r"hypothetically[,\s]+if\s+you\s+(had\s+no\s+rules?|could\s+say\s+anything)",
             0.4, "jailbreak"),

            # 3. Indirect prompt injection
            # Instructions embedded in content the user pastes/quotes
            (r"(the\s+)?(document|text|article|email|message)\s+says?\s+[\"']?\s*(ignore|forget|you\s+are|act\s+as)",
             0.5, "indirect_injection"),
            (r"\[INST\]|\[\/INST\]|<\|im_start\|>|<\|im_end\|>",
             0.6, "indirect_injection"),  # LLM control tokens smuggled in user text

            # 4. Token smuggling — encoding tricks
            # Zero-width / invisible characters
            (r"[\u200b\u200c\u200d\u2060\ufeff]",
             0.4, "token_smuggling"),
            # Lookalike / homoglyph latin letters used to spell bypass words
            # (catches ɪɢɴᴏʀᴇ, ΙGΝORE etc.)
            (r"[\u0400-\u04ff\u0370-\u03ff]{3,}",
             0.3, "token_smuggling"),

            # 5. Excessive repetition / token flooding
            (r"(.)\1{200,}",                    0.5, "token_flood"),
        ]

    # ── Public entry point ────────────────────────────────────────────────────

    def process(self, query: str, session_id: str,
                user_id: str = "guest") -> Dict[str, Any]:
        """
        Run all guardrail checks and return a consolidated result.
        """
        results = {
            "passed": True,
            "session_valid": False,
            "injection_risk": 0.0,
            "sensitive_data_found": False,
            "auth_status": "unknown",
            "warnings": [],
            "actions": [],
            "attack_labels": [],
        }

        # ── Session ───────────────────────────────────────────────────────
        session_check = self._check_session(session_id, user_id)
        results["session_valid"] = session_check["valid"]
        results["auth_status"]   = session_check["status"]

        if not session_check["valid"] and user_id != "guest":
            results["passed"] = False
            results["warnings"].append("Session expired or invalid")
            results["actions"].append("reauthenticate")

        # ── SQL injection ─────────────────────────────────────────────────
        sql_risk = self._check_sql(query)
        if sql_risk > 0:
            results["injection_risk"] = min(results["injection_risk"] + sql_risk, 1.0)
            results["warnings"].append("SQL injection pattern detected")
            results["attack_labels"].append("sql_injection")

        # ── LLM-specific attacks ──────────────────────────────────────────
        llm_check = self._check_llm_attacks(query)
        if llm_check["risk"] > 0:
            results["injection_risk"] = min(
                results["injection_risk"] + llm_check["risk"], 1.0
            )
            results["attack_labels"].extend(llm_check["labels"])
            results["warnings"].extend(llm_check["warnings"])

        # ── Token smuggling: Base64 / hex decode attempt ──────────────────
        smuggle_risk = self._check_encoded_payload(query)
        if smuggle_risk > 0:
            results["injection_risk"] = min(
                results["injection_risk"] + smuggle_risk, 1.0
            )
            results["attack_labels"].append("encoded_payload")
            results["warnings"].append("Possibly encoded payload detected")

        # ── Block / flag based on total risk ─────────────────────────────
        total_risk = results["injection_risk"]
        if total_risk > 0.7:
            results["passed"] = False
            results["actions"].append("block")
        elif total_risk > 0.4:
            results["actions"].append("scrutinize")

        # ── PII / sensitive data ──────────────────────────────────────────
        sensitive_check = self._check_sensitive_data(query)
        results["sensitive_data_found"] = sensitive_check["found"]
        if sensitive_check["found"]:
            results["warnings"].extend(sensitive_check["warnings"])
            results["actions"].append("redact")

        # ── Auth gate ─────────────────────────────────────────────────────
        if self._needs_auth(query) and user_id == "guest":
            results["passed"] = False
            results["warnings"].append("Authentication required for this request")
            results["actions"].append("authenticate")

        return results

    # ── SQL injection ─────────────────────────────────────────────────────────

    def _check_sql(self, query: str) -> float:
        risk = 0.0
        for pattern in self._sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                risk += 0.4
        return min(risk, 1.0)

    # ── LLM-specific attacks ──────────────────────────────────────────────────

    def _check_llm_attacks(self, query: str) -> Dict[str, Any]:
        """
        Check for LLM-specific attack patterns.
        Returns cumulative risk, unique labels, and human-readable warnings.
        """
        risk    = 0.0
        labels  = []
        warnings = []

        for pattern, weight, label in self._llm_attack_patterns:
            if re.search(pattern, query, re.IGNORECASE | re.UNICODE):
                risk += weight
                if label not in labels:
                    labels.append(label)

        if labels:
            label_to_msg = {
                "system_override":   "Attempted system prompt override",
                "role_hijack":       "Role hijacking attempt detected",
                "jailbreak":         "Jailbreak attempt detected",
                "indirect_injection":"Indirect prompt injection detected",
                "token_smuggling":   "Token smuggling / encoding attack detected",
                "token_flood":       "Token flooding / DoS pattern detected",
            }
            for label in set(labels):
                if label in label_to_msg:
                    warnings.append(label_to_msg[label])

        return {"risk": min(risk, 1.0), "labels": labels, "warnings": warnings}

    # ── Encoded payload (Base64 / hex smuggling) ──────────────────────────────

    def _check_encoded_payload(self, query: str) -> float:
        """
        Detect Base64 or hex blobs that decode to attack strings.
        Only triggers if the decoded content contains injection keywords —
        avoids false-positives on legitimate Base64 data.
        """
        _injection_keywords = re.compile(
            r"ignore|forget|jailbreak|DAN|system\s+prompt|override|act\s+as",
            re.IGNORECASE,
        )

        risk = 0.0

        # Base64 blobs (≥20 chars of base64 alphabet)
        for match in re.finditer(r"[A-Za-z0-9+/]{20,}={0,2}", query):
            try:
                decoded = base64.b64decode(match.group()).decode("utf-8", errors="ignore")
                if _injection_keywords.search(decoded):
                    risk += 0.6
                    break
            except Exception:
                pass

        # Hex blobs (≥16 hex chars)
        for match in re.finditer(r"(?:0x)?([0-9a-fA-F]{16,})", query):
            try:
                decoded = bytes.fromhex(match.group(1)).decode("utf-8", errors="ignore")
                if _injection_keywords.search(decoded):
                    risk += 0.5
                    break
            except Exception:
                pass

        return min(risk, 1.0)

    # ── PII / sensitive data ──────────────────────────────────────────────────

    def _check_sensitive_data(self, query: str) -> Dict[str, Any]:
        found    = False
        warnings = []
        for pattern in self._sensitive_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                found = True
                warnings.append("Potential sensitive data detected — please avoid sharing credentials or card numbers")
                break  # One warning is enough
        return {"found": found, "warnings": warnings}

    # ── Auth gate ─────────────────────────────────────────────────────────────

    def _needs_auth(self, query: str) -> bool:
        auth_patterns = [
            r"my\s+(plan|profile|preferences|goals)",
            r"my\s+(spending|transactions|balance)",
            r"update\s+(my|profile)",
            r"create\s+(plan|goal)",
            r"show\s+(me\s+)?my",
        ]
        return any(re.search(p, query, re.IGNORECASE) for p in auth_patterns)

    # ── Session management ────────────────────────────────────────────────────

    def _check_session(self, session_id: str, user_id: str) -> Dict[str, Any]:
        now = datetime.now()
        if session_id in self.sessions:
            session = self.sessions[session_id]
            if now - session["created"] > timedelta(minutes=30):
                return {"valid": False, "status": "expired"}
            session["last_activity"] = now
            return {
                "valid":   True,
                "status":  "authenticated" if session.get("authenticated") else "guest",
                "user_id": session.get("user_id", "guest"),
            }
        # New session
        self.sessions[session_id] = {
            "created":       now,
            "last_activity": now,
            "authenticated": user_id != "guest",
            "user_id":       user_id,
        }
        return {
            "valid":   True,
            "status":  "authenticated" if user_id != "guest" else "guest",
            "user_id": user_id,
        }

    def authenticate_session(self, session_id: str, user_id: str) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["authenticated"] = True
            self.sessions[session_id]["user_id"]       = user_id

    def end_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    # ── PII redaction utility ─────────────────────────────────────────────────

    def redact_pii(self, text: str) -> str:
        text = re.sub(r"[\w.\-]+@[\w.\-]+\.\w+", "[EMAIL REDACTED]", text)
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE REDACTED]", text)
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN REDACTED]", text)
        text = re.sub(r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",
                      "[CARD REDACTED]", text)
        return text