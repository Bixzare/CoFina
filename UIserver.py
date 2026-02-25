"""
CoFina Web Server — Professional Financial Assistant for Young Professionals
Flask wrapper around CoFinaOrchestrator with ORDAEU execution engine.

Usage:
    pip install flask flask-cors python-dotenv
    python cofina_server.py
Then open http://localhost:5011 in your browser.
"""

import os
import sys
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv


# ------------------------------------------------------------------
# Setup Paths (CRITICAL FIX)
# ------------------------------------------------------------------

BASE_DIR = Path(__file__).parent.resolve()
SRC_DIR = BASE_DIR / "src"

# Add src to Python path
sys.path.insert(0, str(SRC_DIR))


# ------------------------------------------------------------------
# Load Environment
# ------------------------------------------------------------------

load_dotenv(dotenv_path=BASE_DIR / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")


# ------------------------------------------------------------------
# Initialise RAG on startup
# ------------------------------------------------------------------

print("🔍 Initialising RAG knowledge base...")
try:
    from RAG.index import ensure_index

    vector_store = ensure_index(API_KEY, force_reindex=False)
    print(" ... RAG ready")

except Exception as e:
    print(f" ... RAG skipped: {e}")


# ------------------------------------------------------------------
# Import CoFina Core (NOW CORRECT)
# ------------------------------------------------------------------

from agents.orchestrator import CoFinaOrchestrator
from utils.logger import AgentLogger
from utils.async_processor import get_async_processor


# ------------------------------------------------------------------
# Flask App
# ------------------------------------------------------------------

app = Flask(__name__, static_folder="ui", static_url_path="")
CORS(app)

_orchestrator = None
_logger = AgentLogger()


def get_orchestrator():
    """
    Lazy-load orchestrator so server starts even if something fails.
    """
    global _orchestrator

    if _orchestrator is None:
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")

        _orchestrator = CoFinaOrchestrator(API_KEY)
        _orchestrator.turn_count = 0

    return _orchestrator


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.route("/")
def index():
    """
    Serve chat UI
    """
    return send_from_directory("ui", "chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    """
    Main chat endpoint.
    """
    data = request.get_json(force=True)
    user_input = (data.get("message") or "").strip()

    if not user_input:
        return jsonify({"error": "Empty message"}), 400

    try:
        orchestrator = get_orchestrator()

        response = orchestrator.process(user_input)
        orchestrator.turn_count += 1

        return jsonify({
            "response": response,
            "user_id": orchestrator.current_user_id,
            "session_id": orchestrator.current_session_id,
            "turn": orchestrator.turn_count,
        })

    except Exception as e:
        _logger.log_step("error", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/status")
def status():
    """
    Returns session metadata.
    """
    try:
        orchestrator = get_orchestrator()

        return jsonify({
            "user_id": orchestrator.current_user_id,
            "session_id": orchestrator.current_session_id,
            "turn_count": getattr(orchestrator, "turn_count", 0),
            "registration_active": orchestrator.registration_agent.is_active(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/logout", methods=["POST"])
def logout():
    """
    Log out current user.
    """
    try:
        orchestrator = get_orchestrator()
        orchestrator.logout()

        return jsonify({
            "message": "Logged out successfully"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/shutdown", methods=["POST"])
def shutdown():
    """
    Optional graceful shutdown endpoint.
    """
    try:
        get_async_processor().shutdown()
        return jsonify({
            "message": "Async processor shutdown complete"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  CoFina Web Interface")
    print("  Intelligent Financial Assistant for Young Professionals")
    print("  Open: http://localhost:5013")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=5013, use_reloader=False)