"""
run_fastvlm.py
Main entry point for the FastVLM Vision-Language Chatbot.
Supports both text-only and image+text conversations.
"""

import sys
import argparse
from vlm_interface import VLMChatInterface


def parse_arguments():
    """
    Parse command-line arguments for FastVLM chatbot configuration.

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="FastVLM Vision-Language Chatbot using Apple's FastVLM-0.5B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default settings (GPU auto-detected)
  python run_fastvlm.py

  # Use custom model path (if model is in different location)
  python run_fastvlm.py --model apple/FastVLM-0.5B

  # Change memory window size
  python run_fastvlm.py --max-turns 3

  # Force CPU usage (disable GPU)
  python run_fastvlm.py --no-gpu

  # Combine options
  python run_fastvlm.py --max-turns 7 --no-gpu

Features:
  • Text conversations with context memory
  • Image understanding and analysis
  • Multi-modal conversations (text + images)
  • Special commands (/help, /image, /clear, etc.)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="apple/FastVLM-0.5B",
        help="Hugging Face model name or local path (default: apple/FastVLM-0.5B)",
    )

    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns to remember (default: 10)",
    )

    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU usage, run on CPU only",
    )

    return parser.parse_args()


def print_banner():
    """
    Print startup banner with model information.
    """
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "  🤖 FASTVLM VISION-LANGUAGE CHATBOT".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("║" + "  Powered by Apple FastVLM-0.5B".center(78) + "║")
    print("║" + "  Supports Text + Image Understanding".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()


def main():
    """
    Main function to initialize and run the FastVLM chatbot.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Print banner
    print_banner()

    # Display configuration
    print("Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Memory Window: {args.max_turns} turns")
    print(
        f"  GPU: {'Disabled (CPU only)' if args.no_gpu else 'Auto-detect (GPU if available)'}"
    )
    print()

    try:
        # Initialize chatbot interface
        chat = VLMChatInterface(
            model_name=args.model,
            max_turns=args.max_turns,
            use_gpu=not args.no_gpu,
        )

        # Run the interactive chat loop
        chat.run()

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print("  Chatbot interrupted by user (Ctrl+C)")
        print("  Goodbye! 👋")
        print("=" * 80)
        print()
        sys.exit(0)

    except ImportError as e:
        print("\n" + "=" * 80)
        print("  ✗ IMPORT ERROR")
        print("=" * 80)
        print(f"  {e}")
        print()
        print("  Make sure all required dependencies are installed:")
        print("    pip install -r requirements.txt")
        print()
        print("  For GPU support, install PyTorch with CUDA:")
        print(
            "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
        )
        print("=" * 80)
        print()
        sys.exit(1)

    except FileNotFoundError as e:
        print("\n" + "=" * 80)
        print("  ✗ FILE NOT FOUND ERROR")
        print("=" * 80)
        print(f"  {e}")
        print()
        print("  Possible causes:")
        print(
            "    • Model not downloaded yet (will download automatically on first run)"
        )
        print("    • Image file path is incorrect (check /image command)")
        print("    • Missing configuration files")
        print("=" * 80)
        print()
        sys.exit(1)

    except Exception as e:
        print("\n" + "=" * 80)
        print("  ✗ FATAL ERROR")
        print("=" * 80)
        print(f"  {e}")
        print()
        print("  Troubleshooting tips:")
        print("    • Check your internet connection (for model download)")
        print("    • Verify you have enough disk space (~2GB for model)")
        print("    • Try running with --no-gpu flag")
        print("    • Check requirements.txt dependencies are installed")
        print()
        print("  For GPU-related errors:")
        print("    • Verify CUDA is installed (if using GPU)")
        print("    • Update GPU drivers")
        print("    • Try: python run_fastvlm.py --no-gpu")
        print("=" * 80)
        print()
        sys.exit(1)


if __name__ == "__main__":
    main()
