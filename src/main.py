"""
Main entry point for YouTube RAG application.
Runs both the CLI and Gradio UI interfaces.
"""

import os
import logging
import argparse
import threading
import webbrowser
from dotenv import load_dotenv
import sys
from pathlib import Path

# Add the parent directory to sys.path if running the script directly
if __name__ == "__main__":
    parent_dir = str(Path(__file__).parent.parent)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)

# Import local modules
from src.cli_app import run_cli
from src.ui_app import run_ui

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def check_api_keys():
    """Check if necessary API keys are set and warn if not."""
    openai_key = os.getenv("OPENAI_API_KEY")
    youtube_key = os.getenv("YOUTUBE_API_KEY")
    
    if not openai_key:
        logger.warning("⚠️ OPENAI_API_KEY not found. Please set it in your .env file or environment variables.")
    
    if not youtube_key:
        logger.warning("⚠️ YOUTUBE_API_KEY not found. This is needed for downloading from playlists.")
        
def open_browser(port, path="/"):
    """Open the browser after a delay."""
    import time
    time.sleep(2)  # Give the server time to start
    url = f"http://localhost:{port}{path}"
    logger.info(f"Opening browser at: {url}")
    webbrowser.open(url)

def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description='YouTube Transcript RAG Application')
    parser.add_argument('--mode', type=str, choices=['cli', 'ui', 'both'], default='both',
                        help='Run mode: "cli" for Panel CLI, "ui" for Gradio UI, or "both" (default)')
    parser.add_argument('--cli-port', type=int, default=5006,
                        help='Port for the CLI interface (default: 5006)')
    parser.add_argument('--ui-port', type=int, default=7860,
                        help='Port for the UI interface (default: 7860)')
    parser.add_argument('--no-browser', action='store_true',
                        help='Do not automatically open browser')
    
    args = parser.parse_args()
    
    # Check for API keys
    check_api_keys()
    
    # Create a slightly nicer console header
    print("\n" + "=" * 60)
    print(" YouTube Transcript RAG Application ".center(60, "="))
    print("=" * 60)
    print(f" Mode: {args.mode.upper()} ".center(60, "-"))
    print("=" * 60 + "\n")
    
    if args.mode == 'cli' or args.mode == 'both':
        logger.info(f"Starting Panel CLI on port {args.cli_port}...")
        panel_app = run_cli()
        
        if args.mode == 'cli':
            # If only running CLI, start it in the main thread
            if not args.no_browser:
                threading.Thread(target=open_browser, args=(args.cli_port,)).start()
            panel_app.show(port=args.cli_port)
        else:
            # If running both, start CLI in a separate thread
            import panel as pn
            panel_thread = threading.Thread(
                target=lambda: pn.serve(panel_app, port=args.cli_port, show=False)
            )
            panel_thread.daemon = True
            panel_thread.start()
            if not args.no_browser:
                threading.Thread(target=open_browser, args=(args.cli_port,)).start()
    
    if args.mode == 'ui' or args.mode == 'both':
        logger.info(f"Starting Gradio UI on port {args.ui_port}...")
        gradio_app = run_ui()
        
        if not args.no_browser and args.mode == 'ui':
            # Only auto-open browser for Gradio if not already opened for Panel
            threading.Thread(target=open_browser, args=(args.ui_port,)).start()
            
        # Start Gradio in the main thread
        gradio_app.launch(
            server_port=args.ui_port,
            share=False,
            inbrowser=False,  # We handle browser opening ourselves
            server_name="0.0.0.0"
        )

if __name__ == "__main__":
    main()