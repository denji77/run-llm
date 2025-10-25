"""
vlm_interface.py
Interactive command-line interface for Apple FastVLM vision-language chatbot.
Supports both text-only and image+text conversations.
"""

import os
from pathlib import Path
from chat_memory import ChatMemory
from vlm_loader import VLMLoader


class VLMChatInterface:
    """
    Command-line interface for FastVLM chatbot.
    Handles user input, commands, and conversation flow with vision support.
    """

    def __init__(self, model_name=None, max_turns=5, use_gpu=True):
        """
        Initialize the chat interface.

        Args:
            model_name (str): Hugging Face model name or path
            max_turns (int): Maximum conversation turns to remember
            use_gpu (bool): Whether to use GPU if available
        """
        self.model_name = model_name or "apple/FastVLM-0.5B"
        self.max_turns = max_turns
        self.use_gpu = use_gpu
        self.loader = None
        self.memory = None
        self.current_image = None

    def initialize(self):
        """
        Initialize the model and memory components.
        """
        print("\n" + "=" * 70)
        print("  🤖 FASTVLM VISION-LANGUAGE CHATBOT")
        print("=" * 70)
        print()

        # Initialize model loader
        self.loader = VLMLoader(model_name=self.model_name, use_gpu=self.use_gpu)
        self.loader.load_model()

        # Initialize conversation memory
        self.memory = ChatMemory(max_turns=self.max_turns)

        print()
        print("=" * 70)
        print("  ✓ Chatbot ready! Type your message or a command.")
        print("  📷 This model can understand both text and images!")
        print("  Type '/help' for available commands.")
        print("=" * 70)
        print()

    def display_welcome(self):
        """
        Display welcome banner with instructions.
        """
        print()
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 68 + "║")
        print("║" + "  WELCOME TO FASTVLM CHATBOT".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("║" + "  Powered by Apple FastVLM-0.5B (LlavaQwen2)".center(68) + "║")
        print("║" + " " * 68 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        print("📝 Features:")
        print("  • Natural text conversations")
        print("  • Conversation memory (sliding window)")
        print("  • Special commands (type /help to see them)")
        print()
        print("💡 Examples:")
        print("  • 'Hello! How are you?'")
        print("  • 'Tell me about machine learning'")
        print("  • 'What is the capital of France?'")
        print()
        print("⚠️  Note: Image support is currently limited for this model.")
        print()

    def generate_response(self, user_input):
        """
        Generate a response from the model based on user input and history.

        Args:
            user_input (str): User's message

        Returns:
            str: Generated response from the model
        """
        try:
            # Get conversation history
            history = self.memory.get_messages()

            # Format conversation history for the model
            conversation_list = []
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                conversation_list.append({"role": role, "content": content})

            # Generate response using the VLM
            response = self.loader.chat(
                message=user_input,
                image_path=self.current_image,
                conversation_history=conversation_list,
            )

            return response.strip()

        except Exception as e:
            error_msg = f"Generation error: {str(e)}"
            print(f"\n⚠️  {error_msg}")
            return "I apologize, but I encountered an error generating a response. Please try rephrasing your message."

    def handle_command(self, command):
        """
        Handle special commands from the user.

        Args:
            command (str): Command string (e.g., '/help', '/exit')

        Returns:
            bool: True if chat should continue, False to exit
        """
        command = command.lower().strip()

        if command == "/exit" or command == "/quit":
            print("\nExiting chatbot. Goodbye! 👋")
            return False

        elif command == "/help":
            self.display_help()

        elif command == "/clear":
            self.memory.clear_history()
            self.current_image = None
            print("\n✓ Conversation history cleared!")
            print("✓ Image context cleared!")

        elif command == "/info":
            self.display_info()

        elif command.startswith("/image "):
            image_path = command[7:].strip()
            self.set_image(image_path)

        elif command == "/clearimage":
            self.current_image = None
            print("\n✓ Image context cleared!")

        else:
            print(f"\n⚠ Unknown command: {command}")
            print("Type '/help' to see available commands.")

        return True

    def set_image(self, image_path):
        """
        Set the current image for the conversation.

        Args:
            image_path (str): Path to the image file
        """
        # Remove quotes if present
        image_path = image_path.strip('"').strip("'")

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"\n✗ Error: Image file not found: {image_path}")
            print("Please provide a valid path to an image file.")
            return

        # Check if it's a valid image format
        valid_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"]
        file_ext = Path(image_path).suffix.lower()

        if file_ext not in valid_extensions:
            print(f"\n⚠ Warning: '{file_ext}' may not be a supported image format")
            print(f"Supported formats: {', '.join(valid_extensions)}")

        self.current_image = image_path
        print(f"\n✓ Image loaded: {image_path}")
        print("Now you can ask questions about this image!")

    def display_help(self):
        """
        Display available commands.
        """
        print()
        print("=" * 70)
        print("  AVAILABLE COMMANDS")
        print("=" * 70)
        print()
        print("  TEXT COMMANDS:")
        print("    /help              - Show this help message")
        print("    /info              - Show chatbot and memory information")
        print("    /clear             - Clear conversation history and image")
        print("    /exit or /quit     - Exit the chatbot")
        print()
        print("  IMAGE COMMANDS:")
        print("    /image <path>      - Load an image (limited support)")
        print("    /clearimage        - Remove current image from context")
        print()
        print("  ⚠️  Note: Image features are currently limited for this model.")
        print()
        print("=" * 70)
        print()

    def display_info(self):
        """
        Display chatbot and memory information.
        """
        model_info = self.loader.get_model_info()
        memory_info = self.memory.get_memory_info()

        print()
        print("=" * 70)
        print("  CHATBOT INFORMATION")
        print("=" * 70)
        print()
        print(f"  MODEL:")
        print(f"    Name: {model_info['model_name']}")
        print(f"    Type: {model_info['model_type']}")
        print(f"    Device: {model_info['device']}")
        print(f"    Size: {model_info['size_mb']:.2f} MB")
        print(f"    Capabilities: {', '.join(model_info['capabilities'])}")
        print()
        print(f"  CONVERSATION:")
        print(f"    Turns: {memory_info['conversation_turns']}")
        print(
            f"    Messages: {memory_info['total_messages']}/{memory_info['max_messages']}"
        )
        print(f"    Memory Window: {memory_info['max_turns']} turns")
        print()
        print(f"  IMAGE CONTEXT:")
        if self.current_image:
            print(f"    Current Image: {self.current_image}")
        else:
            print(f"    Current Image: None (text-only mode)")
        print()
        print("=" * 70)
        print()

    def run(self):
        """
        Main conversation loop.
        """
        # Initialize components
        self.display_welcome()
        self.initialize()

        # Main loop
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Skip empty input
                if not user_input:
                    continue

                # Handle commands
                if user_input.startswith("/"):
                    should_continue = self.handle_command(user_input)
                    if not should_continue:
                        break
                    continue

                # Add user message to memory
                self.memory.add_user_message(user_input)

                # Generate response
                print("Bot: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

                # Add bot response to memory
                self.memory.add_bot_message(response)

                print()  # Add spacing

            except KeyboardInterrupt:
                print("\n\nChatbot interrupted. Goodbye! 👋")
                break

            except Exception as e:
                print(f"\n✗ Error: {e}")
                print("Please try again or type '/help' for assistance.")


def main():
    """
    Entry point for running the VLM chatbot interface.
    """
    chat = VLMChatInterface(model_name="apple/FastVLM-0.5B", max_turns=5, use_gpu=True)
    chat.run()


if __name__ == "__main__":
    main()
