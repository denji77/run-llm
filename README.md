# Local Command-Line Chatbot using Hugging Face

**Machine Learning Intern Technical Assignment**

## Overview

A command-line chatbot implementation using Apple FastVLM-0.5B from Hugging Face, featuring sliding window memory management for multi-turn conversations.

## Project Structure

```
assign/
├── vlm_loader.py           # Model and tokenizer loading
├── chat_memory.py          # Sliding window memory buffer
├── vlm_interface.py        # CLI loop and integration
├── run_fastvlm.py          # Main entry point
├── run_fastvlm.bat         # Windows launcher
├── requirements.txt        # Dependencies
└── README.md              # Documentation
```

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB recommended)
- ~2GB storage for model cache
- GPU optional (CUDA-compatible, auto-detected)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: GPU support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

```bash
# Run chatbot
python run_fastvlm.py

# Windows shortcut
run_fastvlm.bat

# Options
python run_fastvlm.py --no-gpu          # Force CPU
python run_fastvlm.py --max-turns 4     # Set memory window (default)
python run_fastvlm.py --help            # Show all options
```

## Example Interaction

```
You: What is the capital of France?
Bot: Paris

You: And what about Italy?
Bot: The capital of Italy is Rome.

You: /exit
Exiting chatbot. Goodbye!
```

## Available Commands

- `/exit` - Exit the chatbot
- `/clear` - Clear conversation history
- `/info` - Display model and memory information
- `/help` - Show available commands

## Technical Details

### Model
- **Name**: Apple FastVLM-0.5B
- **Architecture**: LlavaQwen2ForCausalLM
- **Size**: ~1.4GB
- **Inference**: Local (no API calls)

### Memory Management
- **Type**: Sliding window buffer using Python deque
- **Default**: 4 turns (8 messages)
- **Behavior**: Automatic FIFO removal when full
- **Purpose**: Maintains conversation context while managing token limits

### Text Generation
- **Max tokens**: 80 per response
- **Temperature**: 0.8
- **Top-p**: 0.95
- **Repetition penalty**: 1.3
- **No-repeat n-gram**: 3

## Module Descriptions

### vlm_loader.py
Handles model initialization and text generation. Key methods:
- `load_model()` - Loads model and tokenizer from Hugging Face
- `generate_response()` - Generates text with optimized parameters
- `chat()` - Manages conversational interface with history

### chat_memory.py
Implements sliding window conversation buffer. Key methods:
- `add_user_message()` - Adds user input to history
- `add_bot_message()` - Adds bot response to history
- `get_messages()` - Retrieves conversation history
- `clear_history()` - Resets conversation state

### vlm_interface.py
Manages CLI interaction loop. Key methods:
- `run()` - Main conversation loop
- `handle_command()` - Processes special commands
- `generate_response()` - Integrates model and memory for response generation

### run_fastvlm.py
Entry point with argument parsing. Initializes chatbot with command-line options and handles graceful shutdown.
