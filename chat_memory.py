"""
chat_memory.py
Module for managing conversation history with sliding window buffer.
"""

from collections import deque
from typing import List, Dict, Optional


class ChatMemory:
    """
    Manages conversation history using a sliding window buffer.
    Maintains only the most recent N turns to keep context relevant.
    """

    def __init__(self, max_turns=5):
        """
        Initialize the ChatMemory with a sliding window buffer.

        Args:
            max_turns (int): Maximum number of conversation turns to maintain.
                           Default is 5 (5 user messages + 5 bot responses = 10 messages).
        """
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns * 2)
        self.turn_count = 0

    def add_user_message(self, message: str):
        """
        Add a user message to the conversation history.

        Args:
            message (str): The user's input message
        """
        self.history.append({"role": "user", "content": message})

    def add_bot_message(self, message: str):
        """
        Add a bot response to the conversation history.

        Args:
            message (str): The bot's response message
        """
        self.history.append({"role": "bot", "content": message})
        self.turn_count += 1

    def add_exchange(self, user_message: str, bot_message: str):
        """
        Add a complete conversation exchange (user + bot).

        Args:
            user_message (str): The user's input
            bot_message (str): The bot's response
        """
        self.add_user_message(user_message)
        self.add_bot_message(bot_message)

    def get_history(self) -> List[Dict[str, str]]:
        """
        Get the entire conversation history as a list.

        Returns:
            list: List of message dictionaries with 'role' and 'content' keys
        """
        return list(self.history)

    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the conversation history.
        Alias for get_history() for compatibility.

        Returns:
            list: List of message dictionaries with 'role' and 'content' keys
        """
        return list(self.history)

    def get_formatted_history(self, separator="\n") -> str:
        """
        Get conversation history formatted as a single string.

        Args:
            separator (str): String to use between messages

        Returns:
            str: Formatted conversation history
        """
        formatted_lines = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Bot"
            formatted_lines.append(f"{role}: {msg['content']}")
        return separator.join(formatted_lines)

    def get_context_for_model(self, current_input: str) -> str:
        """
        Build context string for the model including history and current input.

        Args:
            current_input (str): The current user input

        Returns:
            str: Complete context string for model input
        """
        context_parts = []

        for msg in self.history:
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            else:
                context_parts.append(f"Bot: {msg['content']}")

        context_parts.append(f"User: {current_input}")
        context_parts.append("Bot:")

        return "\n".join(context_parts)

    def get_last_n_turns(self, n: int) -> List[Dict[str, str]]:
        """
        Get the last N conversation turns.

        Args:
            n (int): Number of recent turns to retrieve

        Returns:
            list: Last N turns from the conversation
        """
        return list(self.history)[-(n * 2) :] if len(self.history) > 0 else []

    def clear_history(self):
        """
        Clear all conversation history and reset turn count.
        """
        self.history.clear()
        self.turn_count = 0

    def is_empty(self) -> bool:
        """
        Check if the conversation history is empty.

        Returns:
            bool: True if no messages in history, False otherwise
        """
        return len(self.history) == 0

    def get_turn_count(self) -> int:
        """
        Get the total number of conversation turns (exchanges).

        Returns:
            int: Number of complete conversation turns
        """
        return self.turn_count

    def get_memory_size(self) -> int:
        """
        Get the current number of messages in memory.

        Returns:
            int: Current number of messages stored
        """
        return len(self.history)

    def get_memory_info(self) -> Dict[str, int]:
        """
        Get detailed information about the memory state.

        Returns:
            dict: Memory statistics including size, turns, and capacity
        """
        return {
            "current_messages": len(self.history),
            "total_messages": len(self.history),
            "max_messages": self.max_turns * 2,
            "conversation_turns": self.turn_count,
            "max_turns": self.max_turns,
            "is_full": len(self.history) >= self.max_turns * 2,
        }

    def __str__(self) -> str:
        """
        String representation of the ChatMemory.

        Returns:
            str: Summary of memory state
        """
        return (
            f"ChatMemory(turns={self.turn_count}, "
            f"messages={len(self.history)}/{self.max_turns * 2})"
        )

    def __repr__(self) -> str:
        """
        Detailed representation of the ChatMemory.

        Returns:
            str: Detailed memory representation
        """
        return f"ChatMemory(max_turns={self.max_turns}, current_messages={len(self.history)})"


if __name__ == "__main__":
    print("Testing ChatMemory...")

    memory = ChatMemory(max_turns=3)

    memory.add_exchange("Hello!", "Hi there! How can I help you?")
    memory.add_exchange("What is Python?", "Python is a programming language.")
    memory.add_exchange("Tell me more", "Python is known for its simplicity.")

    print(f"\n{memory}")
    print("\nConversation History:")
    print(memory.get_formatted_history())

    print("\nMemory Info:")
    info = memory.get_memory_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    print("\n--- Adding new exchange (should slide window) ---")
    memory.add_exchange("What about Java?", "Java is another popular language.")

    print(f"\n{memory}")
    print("\nUpdated History:")
    print(memory.get_formatted_history())

    print("\n✓ ChatMemory test completed successfully!")
