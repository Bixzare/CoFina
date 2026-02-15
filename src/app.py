"""
CoFina - Professional Financial Assistant for Young Professionals
Main Application Entry Point with Professional Interface
"""

import os
import sys
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import platform

# Add src to path
sys.path.append(str(Path(__file__).parent))

from agents.orchestrator import CoFinaOrchestrator
from utils.logger import AgentLogger
from utils.cache import get_cache
from utils.async_processor import get_async_processor

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize RAG on startup
print("🔍 Initializing RAG knowledge base...")
from RAG.index import ensure_index
vector_store = ensure_index(API_KEY, force_reindex=False)

# ============================================================================
# PROFESSIONAL UI COMPONENTS
# ============================================================================

class CoFinaInterface:
    """Professional terminal interface for CoFina"""
    
    # Colors (ANSI)
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',
        'italic': '\033[3m',
        'underline': '\033[4m',
        'blink': '\033[5m',
        
        # Foreground
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        
        # Bright
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        
        # Background
        'bg_black': '\033[40m',
        'bg_red': '\033[41m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_blue': '\033[44m',
        'bg_magenta': '\033[45m',
        'bg_cyan': '\033[46m',
        'bg_white': '\033[47m',
    }
    
    # Symbols
    SYMBOLS = {
        'logo': '▓',
        'success': '✓',
        'error': '✗',
        'warning': '⚠',
        'info': 'ℹ',
        'arrow': '→',
        'bullet': '•',
        'diamond': '◆',
        'star': '★',
        'clock': '🕒',
        'user': '👤',
        'bot': '🤖',
        'brain': '🧠',
        'database': '🗄️',
        'cache': '⚡',
        'logout': '🚪',
        'exit': '👋',
        'thinking': '⏳'
    }
    
    def __init__(self):
        self.loader = None
        self._check_terminal_support()
    
    def _check_terminal_support(self):
        """Check if terminal supports colors"""
        self.has_colors = platform.system() != 'Windows' or os.getenv('WT_SESSION')
    
    def c(self, text, *styles):
        """Apply color/styles to text"""
        if not self.has_colors:
            return text
        
        codes = []
        for style in styles:
            if style in self.COLORS:
                codes.append(self.COLORS[style])
        
        return f"{''.join(codes)}{text}{self.COLORS['reset']}"
    
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def print_banner(self):
        """Print professional banner"""
        banner = f"""
{self.c('┌', 'bright_blue')}{self.c('─' * 78, 'bright_blue')}{self.c('┐', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('██████╗ ██████╗ ███████╗██╗███╗   ██╗ █████╗ ', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('██╔══██╗██╔══██╗██╔════╝██║████╗  ██║██╔══██╗', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('██║  ██║██████╔╝█████╗  ██║██╔██╗ ██║███████║', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('██║  ██║██╔══██╗██╔══╝  ██║██║╚██╗██║██╔══██║', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('██████╔╝██║  ██║██║     ██║██║ ╚████║██║  ██║', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c('╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝', 'bright_white', 'bold')}        {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c(' ' * 78, 'dim')}  {self.c('│', 'bright_blue')}
{self.c('│', 'bright_blue')}  {self.c(' INTELLIGENT FINANCIAL ASSISTANT FOR YOUNG PROFESSIONALS', 'bright_cyan')}  {self.c('│', 'bright_blue')}
{self.c('└', 'bright_blue')}{self.c('─' * 78, 'bright_blue')}{self.c('┘', 'bright_blue')}
        """
        print(banner)
        
        # System info
        print(f"\n{self.c('⚡', 'bright_yellow')} {self.c('System Status:', 'bright_white')} {self.c('✓ Active', 'bright_green')}  |  {self.c('🕒', 'bright_cyan')} {self.c(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'bright_black')}")
        print(f"{self.c('─' * 80, 'bright_black')}\n")
    
    def print_commands(self):
        """Print available commands"""
        commands = [
            ("exit", "Exit the application"),
            ("logout", "Log out current user"),
            ("status", "Show session status"),
            ("cache", "Show cache statistics"),
            ("clear", "Clear all caches"),
            ("help", "Show this help")
        ]
        
        print(f"\n{self.c('📋', 'bright_magenta')} {self.c('Available Commands:', 'bright_white', 'bold')}")
        for cmd, desc in commands:
            print(f"  {self.c('•', 'bright_blue')} {self.c(cmd.ljust(10), 'bright_yellow')} {self.c(desc, 'dim')}")
        print()
    
    def print_status(self, orchestrator):
        """Print detailed session status"""
        print(f"\n{self.c('┌', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┐', 'bright_blue')}")
        print(f"{self.c('│', 'bright_blue')}  {self.c('📊 SESSION STATUS', 'bright_white', 'bold')} {' ' * 17}{self.c('│', 'bright_blue')}")
        print(f"{self.c('├', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┤', 'bright_blue')}")
        
        # Session info
        print(f"{self.c('│', 'bright_blue')}  {self.c('🆔', 'bright_cyan')} Session:  {self.c(orchestrator.current_session_id, 'bright_white')}")
        print(f"{self.c('│', 'bright_blue')}  {self.c('👤', 'bright_cyan')} User:     {self.c(orchestrator.current_user_id, 'bright_green' if orchestrator.current_user_id != 'guest' else 'bright_yellow')}")
        
        # Turn count
        try:
            turn = orchestrator.state_manager.current_state['conversation']['turn_count']
            phase = orchestrator.state_manager.current_state['conversation']['current_phase']
            print(f"{self.c('│', 'bright_blue')}  {self.c('🔄', 'bright_cyan')} Turns:    {self.c(str(turn), 'bright_white')}")
            print(f"{self.c('│', 'bright_blue')}  {self.c('📌', 'bright_cyan')} Phase:    {self.c(phase, 'bright_magenta')}")
        except:
            print(f"{self.c('│', 'bright_blue')}  {self.c('🔄', 'bright_cyan')} Turns:    {self.c('0', 'dim')}")
        
        print(f"{self.c('└', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┘', 'bright_blue')}\n")
    
    def print_cache_stats(self):
        """Print cache statistics"""
        cache = get_cache()
        
        print(f"\n{self.c('┌', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┐', 'bright_blue')}")
        print(f"{self.c('│', 'bright_blue')}  {self.c('⚡ CACHE STATISTICS', 'bright_white', 'bold')} {' ' * 14}{self.c('│', 'bright_blue')}")
        print(f"{self.c('├', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┤', 'bright_blue')}")
        
        # Main cache
        stats = cache.get_stats()
        print(f"{self.c('│', 'bright_blue')}  {self.c('📦', 'bright_cyan')} Main Cache:")
        print(f"{self.c('│', 'bright_blue')}     {self.c('•', 'bright_blue')} Entries: {self.c(str(stats['entry_count']), 'bright_white')}")
        size_mb = stats['total_size_mb']
        print(f"{self.c('│', 'bright_blue')}     {self.c('•', 'bright_blue')} Size:    {self.c(f'{size_mb} MB', 'bright_white')}")
        
        # Subdirectories
        for subdir in ["embeddings", "rag", "responses", "advice"]:
            substats = cache.get_stats(subdir=subdir)
            if substats['entry_count'] > 0:
                print(f"{self.c('│', 'bright_blue')}  {self.c('📁', 'bright_cyan')} {subdir.capitalize()}:")
                print(f"{self.c('│', 'bright_blue')}     {self.c('•', 'bright_blue')} Entries: {self.c(str(substats['entry_count']), 'bright_white')}")
                sub_size = substats['total_size_mb']
                print(f"{self.c('│', 'bright_blue')}     {self.c('•', 'bright_blue')} Size:    {self.c(f'{sub_size} MB', 'bright_white')}")
        
        print(f"{self.c('└', 'bright_blue')}{self.c('─' * 38, 'bright_blue')}{self.c('┘', 'bright_blue')}\n")
    
    def loading_animation(self, message="Processing"):
        """Context manager for loading animation"""
        class Loader:
            def __init__(self, interface, message):
                self.interface = interface
                self.message = message
                self.running = False
                self.thread = None
            
            def __enter__(self):
                self.running = True
                self.thread = threading.Thread(target=self._animate)
                self.thread.daemon = True
                self.thread.start()
                return self
            
            def _animate(self):
                chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                i = 0
                while self.running:
                    sys.stdout.write(f'\r{self.interface.c(chars[i % len(chars)], "bright_cyan")} {self.interface.c(self.message, "dim")}...')
                    sys.stdout.flush()
                    time.sleep(0.1)
                    i += 1
            
            def __exit__(self, *args):
                self.running = False
                if self.thread:
                    self.thread.join(timeout=0.5)
                sys.stdout.write('\r' + ' ' * 50 + '\r')
                sys.stdout.flush()
        
        return Loader(self, message)
    
    def prompt_user(self):
        """Get user input with styled prompt"""
        return input(f"\n{self.c('┌', 'bright_blue')} {self.c('👤', 'bright_green')} {self.c('You:', 'bright_white', 'bold')}\n{self.c('└', 'bright_blue')} {self.c('→', 'bright_cyan')} ").strip()
    
    def show_response(self, response):
        """Display agent response with styling"""
        print(f"\n{self.c('┌', 'bright_magenta')} {self.c('🤖', 'bright_magenta')} {self.c('CoFina:', 'bright_white', 'bold')}")
        
        # Format response with proper wrapping
        words = response.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= 70:
                current_line.append(word)
                current_length += len(word) + 1
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
        
        if current_line:
            lines.append(' '.join(current_line))
        
        for line in lines:
            print(f"{self.c('│', 'bright_magenta')}   {self.c(line, 'bright_white')}")
        
        print(f"{self.c('└', 'bright_magenta')}{self.c('─' * 40, 'bright_magenta', 'dim')}\n")
    
    def show_welcome(self):
        """Show welcome message"""
        welcome = f"""
{self.c('✨', 'bright_yellow')} {self.c('Welcome to CoFina!', 'bright_white', 'bold')} {self.c('✨', 'bright_yellow')}

{self.c('I can help you with:', 'bright_cyan')}
  {self.c('•', 'bright_green')} Creating personalized financial plans
  {self.c('•', 'bright_green')} Tracking savings goals
  {self.c('•', 'bright_green')} Researching product prices
  {self.c('•', 'bright_green')} Understanding financial concepts
  {self.c('•', 'bright_green')} Managing budgets and spending

{self.c('Type', 'dim')} {self.c('help', 'bright_yellow')} {self.c('to see available commands.', 'dim')}
        """
        print(welcome)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def needs_long_processing(query: str) -> bool:
    """Check if query might need long processing"""
    query_lower = query.lower()
    
    rag_keywords = [
        "what is", "explain", "tell me about", "how does", 
        "define", "meaning", "concept", "difference between",
        "compare", "versus", "vs", "example of"
    ]
    
    for kw in rag_keywords:
        if kw in query_lower:
            return True
    
    return len(query.split()) > 10


def main():
    """Main application entry point"""
    
    # Check API key
    if not API_KEY:
        print("❌ Error: OPENAI_API_KEY not found in .env file")
        return
    
    # Initialize interface
    ui = CoFinaInterface()
    ui.clear_screen()
    ui.print_banner()
    ui.show_welcome()
    ui.print_commands()
    
    # Initialize orchestrator
    print(f"{ui.c('🔄', 'bright_cyan')} {ui.c('Initializing CoFina...', 'dim')}")
    orchestrator = CoFinaOrchestrator(API_KEY)
    logger = AgentLogger()
    
    # Show initialization complete
    print(f"{ui.c('✅', 'bright_green')} {ui.c('Ready!', 'bright_green', 'bold')} {ui.c('Ask me anything about your finances.', 'dim')}\n")
    
    # Initialize turn counter
    orchestrator.turn_count = 0
    
    # Main loop
    while True:
        try:
            # Get user input with professional prompt
            user_input = ui.prompt_user()
            
            # Handle commands
            if user_input.lower() in ['exit', 'quit']:
                print(f"\n{ui.c('👋', 'bright_yellow')} {ui.c('Thank you for using CoFina. Have a great day!', 'bright_white')}\n")
                # Cleanup
                try:
                    get_async_processor().shutdown()
                except:
                    pass
                break
            
            elif user_input.lower() == 'logout':
                orchestrator.logout()
                print(f"\n{ui.c('✅', 'bright_green')} {ui.c('Logged out successfully', 'bright_white')}\n")
                continue
            
            elif user_input.lower() == 'status':
                ui.print_status(orchestrator)
                continue
            
            elif user_input.lower() in ['cache', 'cache stats']:
                ui.print_cache_stats()
                continue
            
            elif user_input.lower() == 'clear':
                cache = get_cache()
                cache.clear_all()
                cache.clear_all(subdir="embeddings")
                cache.clear_all(subdir="rag")
                cache.clear_all(subdir="responses")
                cache.clear_all(subdir="advice")
                print(f"\n{ui.c('🧹', 'bright_cyan')} {ui.c('All caches cleared', 'bright_white')}\n")
                continue
            
            elif user_input.lower() == 'help':
                ui.print_commands()
                continue
            
            elif not user_input:
                continue
            
            # Check if this needs long processing
            show_loading = needs_long_processing(user_input)
            
            # Process with optional loading animation
            if show_loading:
                with ui.loading_animation("Searching knowledge base"):
                    response = orchestrator.process(user_input)
            else:
                response = orchestrator.process(user_input)
            
            # Create checkpoint every 5 turns
            orchestrator.turn_count += 1
            if orchestrator.turn_count % 5 == 0:
                checkpoint_id = orchestrator.checkpoint_manager.create_checkpoint(
                    session_id=orchestrator.current_session_id,
                    user_id=orchestrator.current_user_id,
                    state=orchestrator.state_manager.current_state,
                    reason="periodic"
                )
                logger.log_step("checkpoint", {"id": checkpoint_id})
            
            # Display response professionally
            ui.show_response(response)
            
        except KeyboardInterrupt:
            print(f"\n\n{ui.c('👋', 'bright_yellow')} {ui.c('Goodbye!', 'bright_white')}\n")
            try:
                get_async_processor().shutdown()
            except:
                pass
            break
            
        except Exception as e:
            print(f"\n{ui.c('❌', 'bright_red')} {ui.c(f'Error: {e}', 'bright_red')}")
            logger.log_step("error", str(e))
            
            if os.getenv("DEBUG"):
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()