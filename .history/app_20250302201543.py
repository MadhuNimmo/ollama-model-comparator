#!/usr/bin/env python3
"""
Ollama Model Comparator - Compare responses from different Ollama models
Main application entry point
"""

import os
import argparse
from src.comparator import OllamaModelComparator
from src.interface import create_interface

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Ollama Model Comparator - Compare different LLMs side by side"
    )
    parser.add_argument(
        "--host", 
        type=str, 
        default="localhost",
        help="Host to run the Gradio interface on (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=7860,
        help="Port to run the Gradio interface on (default: 7860)"
    )
    parser.add_argument(
        "--ollama-url", 
        type=str, 
        default="http://localhost:11434",
        help="Ollama API URL (default: http://localhost:11434)"
    )
    parser.add_argument(
        "--share", 
        action="store_true",
        help="Create a shareable link for the interface"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    # Create Ollama client
    ollama = OllamaModelComparator(ollama_base_url=args.ollama_url)
    
    # Create and launch the interface
    app = create_interface(ollama)
    
    # Launch the app
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )

if __name__ == "__main__":
    main()
