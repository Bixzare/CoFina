"""
CoFina - Professional Financial Assistant for Young Professionals
Main Application Entry Point — with ORDAEU execution transparency.
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
print(" ... Initialising RAG knowledge base...")
try:
    from RAG.index import ensure_index
    vector_store = ensure_index(API_KEY, force_reindex=False)
    print(" ... RAG ready")
except Exception as _e:
    print(f" ... RAG skipped: {_e}")


# ─────────────────────────────────────────────────────────────────────────────
# ORDAEU Stage Map  (tool name → (stage_label, agent_label))
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_STAGE_MAP = {
    # OBSERVE
    "get_user_status":            ("OBSERVE",  "Intent & State Agent"),
    "check_user_exists":          ("OBSERVE",  "Intent & State Agent"),
    "get_my_profile":             ("OBSERVE",  "Intent & State Agent"),
    "search_financial_documents": ("OBSERVE",  "Intent & State Agent"),

    # REASON
    "get_current_time":           ("REASON",   "Financial Analyst Agent"),
    "get_date_difference":        ("REASON",   "Financial Analyst Agent"),
    "add_to_date":                ("REASON",   "Financial Analyst Agent"),
    "get_day_info":               ("REASON",   "Financial Analyst Agent"),
    "calculate_age":              ("REASON",   "Financial Analyst Agent"),
    "get_financial_dates":        ("REASON",   "Financial Analyst Agent"),
    "calculate_compounding":      ("REASON",   "Financial Analyst Agent"),
    "monitoring_flow":            ("REASON",   "Financial Analyst Agent"),

    # DECIDE
    "financial_planning_flow":    ("DECIDE",   "Strategic Planning Agent"),

    # ACT
    "login_flow":                 ("ACT",      "Execution Agent"),
    "registration_flow":          ("ACT",      "Execution Agent"),
    "market_research_flow":       ("ACT",      "Execution Agent"),

    # EVALUATE
    "verify_response":            ("EVALUATE", "Risk & Monitoring Agent"),

    # UPDATE
    "save_user_preference":       ("UPDATE",   "Memory & State Agent"),
    "create_financial_plan_tool": ("UPDATE",   "Memory & State Agent"),
    "authenticate_user":          ("UPDATE",   "Memory & State Agent"),
    "register_new_user":          ("UPDATE",   "Memory & State Agent"),
}

_STAGE_META = {
    #           (icon, color_code)
    "OBSERVE":  ("🟢", "\033[92m"),
    "REASON":   ("🟡", "\033[93m"),
    "DECIDE":   ("🔵", "\033[94m"),
    "ACT":      ("🟣", "\033[95m"),
    "EVALUATE": ("🔴", "\033[91m"),
    "UPDATE":   ("⚪", "\033[96m"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Terminal UI
# ─────────────────────────────────────────────────────────────────────────────

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

    def print_ordaeu_legend(self):
        """Print ORDAEU stage key at startup."""
        w = 80
        print(f"{self.c('─' * w, 'bright_black')}")
        print(self.c("  ORDAEU Execution Stages", "bright_white", "bold"))
        print(f"{self.c('─' * w, 'bright_black')}")
        stages = [
            ("🟢 OBSERVE",  "Intent & state detection"),
            ("🟡 REASON",   "Financial diagnosis & analysis"),
            ("🔵 DECIDE",   "Strategic planning & goal optimisation"),
            ("🟣 ACT",      "Tool execution & plan generation"),
            ("🔴 EVALUATE", "Risk assessment & verification"),
            ("⚪ UPDATE",   "State synchronisation & memory write"),
        ]
        for icon_label, desc in stages:
            print(f"  {icon_label:<18}  {self.c(desc, 'dim')}")
        print(f"{self.c('─' * w, 'bright_black')}\n")

    def print_ordaeu_stage(self, stage: str, agent: str):
        """Print the ORDAEU stage header block."""
        icon, color = _STAGE_META.get(stage, ("⚙️", "\033[97m"))
        reset = self.COLORS["reset"]
        dim   = self.COLORS["dim"]
        bold  = self.COLORS["bold"]
        print(
            f"\n{dim}┌{'─' * 57}{reset}\n"
            f"{dim}│{reset} {icon} {color}{bold}Stage: {stage}{reset}\n"
            f"{dim}│{reset} 🧠 {self.c('Agent:', 'bright_white')} {self.c(agent, 'dim')}"
        )

    def print_tool_call(self, name: str, args: dict):
        """Print the tool invocation line."""
        dim   = self.COLORS["dim"]
        reset = self.COLORS["reset"]
        cyan  = self.COLORS["bright_cyan"]
        args_str = str(args)
        if len(args_str) > 120:
            args_str = args_str[:117] + "..."
        print(f"{dim}│{reset} 🛠️  {cyan}{name}{reset}({self.c(args_str, 'dim')})")

    def print_tool_result(self, result: str, elapsed_ms: int):
        """Print the tool output and timing."""
        dim   = self.COLORS["dim"]
        reset = self.COLORS["reset"]
        green = self.COLORS["bright_green"]
        result_str = result[:160] + "..." if len(result) > 160 else result
        print(f"{dim}│{reset} 📦 {self.c(result_str, 'dim')}")
        print(f"{dim}│{reset} ⏱  {green}{elapsed_ms} ms{reset}")
        print(f"{dim}└{'─' * 57}{reset}\n")

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
            f"\n{self.c('┌', 'bright_blue')} {self.c(' 👤 You:', 'bright_white', 'bold')}\n"
            f"{self.c('|> ', 'bright_blue')} "
        ).strip()

    def show_response(self, response: str):
        print(f"\n{self.c('┌', 'bright_magenta')} {self.c('🤖 CoFina:', 'bright_white', 'bold')}")
        for line in response.splitlines():
            if len(line) > 74:
                words = line.split()
                current, length = [], 0
                for word in words:
                    if length + len(word) + 1 <= 74:
                        current.append(word)
                        length += len(word) + 1
                    else:
                        print(f"{self.c('│', 'bright_magenta')}  {self.c(' '.join(current), 'bright_white')}")
                        current, length = [word], len(word)
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


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented Orchestrator — injects ORDAEU console display into tool loop
# ─────────────────────────────────────────────────────────────────────────────

class InstrumentedOrchestrator(CoFinaOrchestrator):
    """
    Thin subclass that overrides _tool_loop to print ORDAEU stage blocks
    for every tool call without touching any core orchestrator logic.
    """

    def __init__(self, api_key: str, ui: CoFinaInterface) -> None:
        super().__init__(api_key)
        self._ui = ui

    def _tool_loop(self, messages: list, user_query: str) -> str:
        import json, traceback
        from langchain_core.messages import ToolMessage

        tool_map  = {t.name: t for t in self.tools}
        max_turns = 4

        for _ in range(max_turns):
            try:
                llm_response = self.llm_with_tools.invoke(messages)
                messages.append(llm_response)
                tool_calls = getattr(llm_response, "tool_calls", None) or []

                if not tool_calls:
                    final = llm_response.content
                    self._append_history(user_query, final)
                    return final

                for call in tool_calls:
                    name    = call.get("name")
                    args    = call.get("args", {})
                    call_id = call.get("id")

                    # ── ORDAEU stage display ──────────────────────────────
                    stage, agent = _TOOL_STAGE_MAP.get(
                        name, ("ACT", "Execution Agent")
                    )
                    self._ui.print_ordaeu_stage(stage, agent)
                    self._ui.print_tool_call(name, args)
                    # ─────────────────────────────────────────────────────

                    self.logger.log_step("tool_call", {"tool": name, "args": args})

                    fn = tool_map.get(name)
                    t0 = time.perf_counter()
                    if fn:
                        try:
                            result = fn.invoke(args)
                            success = True
                            try:
                                parsed  = json.loads(result) if isinstance(result, str) else result
                                success = not ("error" in parsed or parsed.get("success") is False)
                                self.evaluator.log_tool_call(self.current_session_id, name, success)
                            except Exception:
                                pass
                        except Exception as exc:
                            result  = json.dumps({"error": str(exc)})
                            success = False
                            self.evaluator.log_tool_call(self.current_session_id, name, False)
                    else:
                        result  = json.dumps({"error": f"Unknown tool: {name}"})
                        success = False
                        self.evaluator.log_tool_call(self.current_session_id, name, False)

                    elapsed_ms = round((time.perf_counter() - t0) * 1000)

                    # ── Print result + timing ─────────────────────────────
                    self._ui.print_tool_result(str(result), elapsed_ms)
                    # ─────────────────────────────────────────────────────

                    messages.append(ToolMessage(content=str(result), tool_call_id=call_id))

                    try:
                        parsed = json.loads(result)
                        self._handle_registration_result(parsed)
                    except Exception:
                        pass

            except Exception as exc:
                error_msg = f"Processing error: {exc}"
                print(traceback.format_exc())
                self.logger.log_step("error", error_msg)
                return f"I encountered an error: {error_msg}"

        last_content = getattr(messages[-1], "content", "I need more information to help you.")
        self._append_history(user_query, last_content)
        return last_content


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _needs_rag(query: str) -> bool:
    triggers = [
        "what is", "explain", "how does", "define", "difference between",
        "compare", "versus", " vs ", "example of", "tell me about",
    ]
    q = query.lower()
    return len(query.split()) > 10 or any(t in q for t in triggers)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    if not API_KEY:
        print(" OPENAI_API_KEY not set in .env")
        return

    ui = CoFinaInterface()
    ui.clear_screen()
    ui.print_banner()
    ui.show_welcome()
    ui.print_ordaeu_legend()
    ui.print_help()

    print(f"{ui.c('... Initialising CoFina...', 'dim')}")
    orchestrator = InstrumentedOrchestrator(API_KEY, ui)
    logger = AgentLogger()
    orchestrator.turn_count = 0
    print(f"{ui.c('... Ready!', 'bright_green', 'bold')} {ui.c('Ask me anything.', 'dim')}\n")

    while True:
        try:
            user_input = ui.prompt_user()

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
                print(f"\n{ui.c('...Logged out successfully.', 'bright_green')}\n")
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
            print(f"\n\n{ui.c('... Goodbye!', 'bright_white')}\n")
            try:
                get_async_processor().shutdown()
            except Exception:
                pass
            break

        except Exception as exc:
            print(f"\n{ui.c(f' Error: {exc}', 'bright_red')}")
            logger.log_step("error", str(exc))
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()