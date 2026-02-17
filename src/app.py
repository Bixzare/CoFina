"""
CoFina - Professional Financial Assistant for Young Professionals
Main Application Entry Point
"""

import os
import platform
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent))

from agents.orchestrator import CoFinaOrchestrator
from utils.logger import AgentLogger
from utils.cache import get_cache
from utils.async_processor import get_async_processor

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialise RAG knowledge base on startup
print("🔍 Initialising RAG knowledge base...")
try:
    from RAG.index import ensure_index
    vector_store = ensure_index(API_KEY, force_reindex=False)
    print("✅ RAG ready")
except Exception as _e:
    print(f"⚠️  RAG skipped: {_e}")


# ============================================================================
# Terminal UI
# ============================================================================

class CoFinaInterface:
    """Professional ANSI terminal interface."""

    COLORS = {
        "reset":          "\033[0m",
        "bold":           "\033[1m",
        "dim":            "\033[2m",
        "bright_black":   "\033[90m",
        "bright_red":     "\033[91m",
        "bright_green":   "\033[92m",
        "bright_yellow":  "\033[93m",
        "bright_blue":    "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan":    "\033[96m",
        "bright_white":   "\033[97m",
        "bg_blue":        "\033[44m",
    }

    def __init__(self):
        self.has_colors = platform.system() != "Windows" or os.getenv("WT_SESSION")

    def c(self, text: str, *styles: str) -> str:
        if not self.has_colors:
            return text
        codes = "".join(self.COLORS.get(s, "") for s in styles)
        return f"{codes}{text}{self.COLORS['reset']}"

    def clear_screen(self):
        os.system("clear" if os.name == "posix" else "cls")

    def print_banner(self):
        w = 80
        print(f"\n{self.c('─' * w, 'bright_blue')}")
        print(self.c("  CoFina  │  Intelligent Financial Assistant for Young Professionals",
                      "bright_white", "bold"))
        print(f"{self.c('─' * w, 'bright_blue')}")
        print(
            f"  {self.c('⚡ Status:', 'bright_yellow')} {self.c('Active', 'bright_green')}  "
            f"│  {self.c('🕒', 'bright_cyan')} "
            f"{self.c(datetime.now().strftime('%Y-%m-%d %H:%M'), 'bright_black')}"
        )
        print(f"{self.c('─' * w, 'bright_black')}\n")

    def print_help(self):
        commands = [
            ("exit / quit", "Exit CoFina"),
            ("logout",       "Log out current user"),
            ("status",       "Show session info"),
            ("cache",        "Show cache statistics"),
            ("clear",        "Clear all caches"),
            ("help",         "Show this list"),
        ]
        print(f"\n{self.c('Available commands:', 'bright_white', 'bold')}")
        for cmd, desc in commands:
            print(f"  {self.c(cmd.ljust(14), 'bright_yellow')} {self.c(desc, 'dim')}")
        print()

    def print_status(self, orchestrator):
        u = orchestrator.current_user_id
        color = "bright_green" if u != "guest" else "bright_yellow"
        reg = "🟢 Active" if orchestrator.registration_agent.is_active() else "⚫ None"
        print(f"\n{self.c('── Session Status ──────────────────────', 'bright_blue')}")
        print(f"  User     : {self.c(u, color)}")
        print(f"  Session  : {self.c(orchestrator.current_session_id, 'bright_black')}")
        print(f"  Reg flow : {reg}")
        try:
            turn = orchestrator.state_manager.current_state["conversation"]["turn_count"]
            print(f"  Turns    : {self.c(str(turn), 'bright_white')}")
        except Exception:
            pass
        print(f"{self.c('────────────────────────────────────────', 'bright_blue')}\n")

    def print_cache_stats(self):
        cache = get_cache()
        stats = cache.get_stats()
        print(f"\n{self.c('── Cache ──────────────────', 'bright_blue')}")
        print(f"  Entries : {stats['entry_count']}")
        print(f"  Size    : {stats['total_size_mb']} MB")
        print(f"{self.c('───────────────────────────', 'bright_blue')}\n")

    def loading_animation(self, message: str = "Processing"):
        """Return a context-manager spinner."""
        ui = self

        class _Spinner:
            def __enter__(self):
                self._running = True
                self._t = threading.Thread(target=self._run, daemon=True)
                self._t.start()
                return self

            def _run(self):
                chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while self._running:
                    sys.stdout.write(f"\r{ui.c(chars[i % len(chars)], 'bright_cyan')} "
                                     f"{ui.c(message, 'dim')}...")
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1

            def __exit__(self, *_):
                self._running = False
                self._t.join(timeout=0.5)
                sys.stdout.write("\r" + " " * 60 + "\r")
                sys.stdout.flush()

        return _Spinner()

    def prompt_user(self) -> str:
        return input(
            f"\n{self.c('┌', 'bright_blue')} {self.c('You:', 'bright_white', 'bold')}\n"
            f"{self.c('└→', 'bright_blue')} "
        ).strip()

    def show_response(self, response: str):
        """
        Render CoFina's response.  Multi-line responses (registration prompts,
        step-by-step questions) are displayed line-by-line with a left margin.
        """
        print(f"\n{self.c('┌', 'bright_magenta')} {self.c('CoFina:', 'bright_white', 'bold')}")
        for line in response.splitlines():
            # Wrap long single lines at 74 chars
            if len(line) > 74:
                words = line.split()
                current = []
                length = 0
                for word in words:
                    if length + len(word) + 1 <= 74:
                        current.append(word)
                        length += len(word) + 1
                    else:
                        print(f"{self.c('│', 'bright_magenta')}  {self.c(' '.join(current), 'bright_white')}")
                        current = [word]
                        length = len(word)
                if current:
                    print(f"{self.c('│', 'bright_magenta')}  {self.c(' '.join(current), 'bright_white')}")
            else:
                print(f"{self.c('│', 'bright_magenta')}  {self.c(line, 'bright_white')}")
        print(f"{self.c('└', 'bright_magenta')}{self.c('─' * 42, 'bright_magenta', 'dim')}\n")

    def show_welcome(self):
        register_word = '"register"'
        print(
            f"{self.c('✨ Welcome to CoFina!', 'bright_white', 'bold')}\n\n"
            f"  I can help you with:\n"
            f"  {self.c('•', 'bright_green')} Register & build your complete financial profile\n"
            f"  {self.c('•', 'bright_green')} Create personalised financial plans\n"
            f"  {self.c('•', 'bright_green')} Track savings goals and debt repayment\n"
            f"  {self.c('•', 'bright_green')} Research products and check affordability\n"
            f"  {self.c('•', 'bright_green')} Monitor alerts and financial progress\n\n"
            f"  {self.c('Say', 'dim')} {self.c(register_word, 'bright_yellow')} "
            f"{self.c('to create your account, or ask anything!', 'dim')}\n"
        )


# ============================================================================
# Helpers
# ============================================================================

def _needs_rag(query: str) -> bool:
    """Heuristic: long or conceptual queries benefit from a loading spinner."""
    triggers = [
        "what is", "explain", "how does", "define", "difference between",
        "compare", "versus", " vs ", "example of", "tell me about",
    ]
    q = query.lower()
    return len(query.split()) > 10 or any(t in q for t in triggers)


# ============================================================================
# Main
# ============================================================================

def main():
    if not API_KEY:
        print("❌  OPENAI_API_KEY not set in .env")
        return

    ui = CoFinaInterface()
    ui.clear_screen()
    ui.print_banner()
    ui.show_welcome()
    ui.print_help()

    print(f"{ui.c('🔄 Initialising CoFina...', 'dim')}")
    orchestrator = CoFinaOrchestrator(API_KEY)
    logger = AgentLogger()
    orchestrator.turn_count = 0
    print(f"{ui.c('✅ Ready!', 'bright_green', 'bold')} {ui.c('Ask me anything.', 'dim')}\n")

    while True:
        try:
            user_input = ui.prompt_user()

            # ── Built-in commands ─────────────────────────────────────────
            if not user_input:
                continue

            lc = user_input.lower()

            if lc in ("exit", "quit"):
                print(f"\n{ui.c('👋 Thank you for using CoFina. Goodbye!', 'bright_white')}\n")
                try:
                    get_async_processor().shutdown()
                except Exception:
                    pass
                break

            if lc == "logout":
                orchestrator.logout()
                print(f"\n{ui.c('✅ Logged out successfully.', 'bright_green')}\n")
                continue

            if lc == "status":
                ui.print_status(orchestrator)
                continue

            if lc in ("cache", "cache stats"):
                ui.print_cache_stats()
                continue

            if lc == "clear":
                cache = get_cache()
                for sub in (None, "embeddings", "rag", "responses", "advice"):
                    try:
                        if sub:
                            cache.clear_all(subdir=sub)
                        else:
                            cache.clear_all()
                    except Exception:
                        pass
                print(f"\n{ui.c('🧹 All caches cleared.', 'bright_cyan')}\n")
                continue

            if lc == "help":
                ui.print_help()
                continue

            # ── Process query ─────────────────────────────────────────────
            # Show spinner for heavy queries; registration steps respond instantly
            show_spinner = (
                _needs_rag(user_input)
                and not orchestrator.registration_agent.is_active()
            )

            if show_spinner:
                with ui.loading_animation("Thinking"):
                    response = orchestrator.process(user_input)
            else:
                response = orchestrator.process(user_input)

            ui.show_response(response)

            # Periodic checkpoint every 5 turns
            orchestrator.turn_count += 1
            if orchestrator.turn_count % 5 == 0:
                try:
                    ckpt = orchestrator.checkpoint_manager.create_checkpoint(
                        session_id=orchestrator.current_session_id,
                        user_id=orchestrator.current_user_id,
                        state=orchestrator.state_manager.current_state,
                        reason="periodic",
                    )
                    logger.log_step("checkpoint", {"id": ckpt})
                except Exception:
                    pass

        except KeyboardInterrupt:
            print(f"\n\n{ui.c('👋 Goodbye!', 'bright_white')}\n")
            try:
                get_async_processor().shutdown()
            except Exception:
                pass
            break

        except Exception as exc:
            print(f"\n{ui.c(f'❌ Error: {exc}', 'bright_red')}")
            logger.log_step("error", str(exc))
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
