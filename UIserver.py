"""
CoFina Web Server — v2.0 with Agent Transparency
Flask wrapper around CoFinaOrchestrator with:
  - /trace/latest  → real-time agent step polling
  - /checkpoint/*  → save, list, restore
  - /chat          → enhanced with verify_score + checkpoint in response
"""

import os
import sys
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import deque
from typing import Dict, Any, List

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR  = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

load_dotenv(dotenv_path=BASE_DIR / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

# ── RAG init ───────────────────────────────────────────────────────────
print("🔍 Initialising RAG knowledge base…")
try:
    from RAG.index import ensure_index
    vector_store = ensure_index(API_KEY, force_reindex=False)
    print("  … RAG ready")
except Exception as e:
    print(f"  … RAG skipped: {e}")

# ── CoFina imports ─────────────────────────────────────────────────────
from agents.orchestrator import CoFinaOrchestrator
from utils.logger import AgentLogger
from utils.async_processor import get_async_processor

# ── Flask ──────────────────────────────────────────────────────────────
app    = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

_orchestrator: "CoFinaOrchestrator | None" = None
_logger  = AgentLogger()
_lock    = threading.Lock()

# ── Step broadcast queue (rolling 100 steps) ───────────────────────────
_trace_steps: deque = deque(maxlen=100)
_step_lock   = threading.Lock()


def push_step(step: Dict[str, Any]) -> None:
    """Push an agent step to the live trace queue."""
    step["ts"]     = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    step["_shown"] = False
    with _step_lock:
        _trace_steps.append(step)


def _patch_orchestrator(orch: "CoFinaOrchestrator") -> None:
    """
    Monkey-patch orchestrator logger to broadcast steps to /trace/latest.
    This avoids modifying the core orchestrator files.
    """
    orig_log = orch.logger.log_step

    def patched_log(step_type: str, data: Any) -> None:
        orig_log(step_type, data)
        step: Dict[str, Any] = {"type": step_type}

        if step_type == "tool_call" and isinstance(data, dict):
            step["tool"] = data.get("tool", "unknown")
            step["args"] = data.get("args", {})

        elif step_type == "verification" and isinstance(data, dict):
            step["score"]  = data.get("score", 0)
            step["reason"] = data.get("reason", "")
            step["action"] = data.get("action", "accept")

        elif step_type == "guardrail" and isinstance(data, dict):
            step["passed"] = data.get("passed", True)
            step["labels"] = data.get("attack_labels", [])

        elif step_type in ("login", "registration_complete") and isinstance(data, dict):
            step["user_id"] = data.get("user_id", "unknown")

        elif step_type == "checkpoint" and isinstance(data, dict):
            step["id"]   = data.get("checkpoint_id", "")
            step["data"] = data

        elif step_type == "error":
            step["message"] = str(data)

        else:
            step["data"] = str(data)[:200]

        push_step(step)

    orch.logger.log_step = patched_log

    # Also patch guardrail to emit its result as a step
    orig_process = orch.process

    def patched_process(user_query: str) -> str:
        # Clear trace for new turn
        with _step_lock:
            _trace_steps.clear()

        # Emit guardrail step eagerly
        gr = orch.guardrail_agent.process(
            query=user_query,
            session_id=orch.current_session_id,
            user_id=orch.current_user_id,
        )
        push_step({
            "type":   "guardrail",
            "passed": gr["passed"],
            "labels": gr.get("attack_labels", []),
        })

        return orig_process(user_query)

    # Don't double-apply
    if not getattr(orch, "_patched", False):
        orch.process = patched_process
        orch._patched = True


def get_orchestrator() -> "CoFinaOrchestrator":
    global _orchestrator
    if _orchestrator is None:
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        _orchestrator = CoFinaOrchestrator(API_KEY)
        _orchestrator.turn_count = 0
        _patch_orchestrator(_orchestrator)
    return _orchestrator


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory("ui", "chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data       = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()
    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        with _lock:
            orch = get_orchestrator()
            response = orch.process(user_input)
            orch.turn_count += 1

            # ── Auto-save checkpoint after every turn ──────────────
            ckpt_data = None
            try:
                ckpt_id = orch.checkpoint_manager.create_checkpoint(
                    session_id=orch.current_session_id,
                    user_id=orch.current_user_id,
                    state={
                        "conversation": [
                            m.content for m in orch.conversation_history[-10:]
                        ],
                        "task": {"turn": orch.turn_count},
                    },
                    reason="auto_turn",
                )
                ckpt_data = {
                    "checkpoint_id": ckpt_id,
                    "user_id":       orch.current_user_id,
                    "turn":          orch.turn_count,
                    "timestamp":     datetime.now().isoformat(),
                }
                push_step({"type": "checkpoint", "id": ckpt_id, "data": ckpt_data})
            except Exception as ce:
                _logger.log_step("checkpoint_error", str(ce))

            # ── Extract verify_score from last verification step ───
            verify_score = None
            with _step_lock:
                for s in reversed(list(_trace_steps)):
                    if s.get("type") == "verification":
                        verify_score = s.get("score")
                        break

            return jsonify({
                "response":     response,
                "user_id":      orch.current_user_id,
                "session_id":   orch.current_session_id,
                "turn":         orch.turn_count,
                "verify_score": verify_score,
                "checkpoint":   ckpt_data,
            })

    except Exception as e:
        _logger.log_step("error", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    try:
        orch = get_orchestrator()
        # Fetch recent checkpoints
        ckpts = []
        try:
            raw = orch.checkpoint_manager.list_checkpoints(
                session_id=orch.current_session_id, limit=5
            )
            ckpts = raw or []
        except Exception:
            pass

        return jsonify({
            "user_id":             orch.current_user_id,
            "session_id":          orch.current_session_id,
            "turn_count":          getattr(orch, "turn_count", 0),
            "registration_active": orch.registration_agent.is_active(),
            "checkpoints":         ckpts,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/trace/latest")
def trace_latest():
    """Return current turn's agent steps for UI polling."""
    with _step_lock:
        steps = list(_trace_steps)
    return jsonify({"steps": steps, "count": len(steps)})


@app.route("/checkpoint/save", methods=["POST"])
def checkpoint_save():
    """Manually save a checkpoint."""
    try:
        with _lock:
            orch = get_orchestrator()
            ckpt_id = orch.checkpoint_manager.create_checkpoint(
                session_id=orch.current_session_id,
                user_id=orch.current_user_id,
                state={
                    "conversation": [m.content for m in orch.conversation_history[-10:]],
                    "task": {"turn": getattr(orch, "turn_count", 0)},
                },
                reason="manual",
            )
            ckpt_data = {
                "checkpoint_id": ckpt_id,
                "user_id":       orch.current_user_id,
                "turn":          getattr(orch, "turn_count", 0),
                "timestamp":     datetime.now().isoformat(),
            }
        return jsonify({"success": True, "checkpoint": ckpt_data})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/checkpoint/restore", methods=["POST"])
def checkpoint_restore():
    """Restore session state from a checkpoint."""
    data = request.get_json(force=True)
    ckpt_id = (data or {}).get("checkpoint_id", "")
    if not ckpt_id:
        return jsonify({"success": False, "error": "No checkpoint_id provided"}), 400
    try:
        with _lock:
            orch = get_orchestrator()
            snap = orch.checkpoint_manager.restore_checkpoint(ckpt_id)
            if snap is None:
                return jsonify({"success": False, "error": "Checkpoint not found"}), 404

            restored_user = snap.get("user_id", "guest")
            orch.current_user_id = restored_user
            orch.conversation_history = []  # Reset conversation after restore
            _logger.log_step("checkpoint_restored", {"id": ckpt_id, "user_id": restored_user})

        return jsonify({"success": True, "checkpoint_id": ckpt_id, "user_id": restored_user})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/checkpoint/list")
def checkpoint_list():
    """List recent checkpoints."""
    try:
        orch = get_orchestrator()
        ckpts = orch.checkpoint_manager.list_checkpoints(
            session_id=orch.current_session_id, limit=20
        ) or []
        return jsonify({"checkpoints": ckpts})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logout", methods=["POST"])
def logout():
    try:
        with _lock:
            orch = get_orchestrator()
            orch.logout()
        return jsonify({"message": "Logged out successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    try:
        get_async_processor().shutdown()
        return jsonify({"message": "Async processor shutdown complete"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Main ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  CoFina v2.0 — Agent Transparency Edition")
    print("  Intelligent Financial Assistant for Young Professionals")
    print("  Open: http://localhost:5013")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5013, use_reloader=False)
