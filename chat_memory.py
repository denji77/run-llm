"""
chat_memory.py
Module for managing conversation history with sliding window buffer.
Enhanced with persistent key facts storage for better context awareness.
"""

from collections import deque
from typing import List, Dict, Optional
import re


class ChatMemory:
    """
    Manages conversation history using a sliding window buffer.
    Maintains only the most recent N turns to keep context relevant.
    Enhanced with persistent key facts storage for important information.
    """

    def __init__(self, max_turns=10):
        """
        Initialize the ChatMemory with a sliding window buffer.

        Args:
            max_turns (int): Maximum number of conversation turns to maintain.
                           Default is 4 (4 user messages + 4 bot responses = 8 messages).
        """
        self.max_turns = max_turns
        self.history = deque(maxlen=max_turns * 2)
        self.turn_count = 0
        self.key_facts = {}  # Store important facts that persist beyond sliding window
        self.conversation_summary = []  # Store summaries of older conversations

    def add_user_message(self, message: str):
        """
        Add a user message to the conversation history.
        Also extracts and stores key facts from the message.

        Args:
            message (str): The user's input message
        """
        self.history.append({"role": "user", "content": message})
        self._extract_key_facts(message, "user")

    def add_bot_message(self, message: str):
        """
        Add a bot response to the conversation history.
        Also extracts and stores key facts from the message.

        Args:
            message (str): The bot's response message
        """
        self.history.append({"role": "assistant", "content": message})
        self.turn_count += 1
        self._extract_key_facts(message, "assistant")

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
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_lines.append(f"{role}: {msg['content']}")
        return separator.join(formatted_lines)

    def get_context_for_model(self, current_input: str) -> str:
        """
        Build context string for the model including history and current input.
        Includes key facts for better context awareness.

        Args:
            current_input (str): The current user input

        Returns:
            str: Complete context string for model input
        """
        context_parts = []

        # Add key facts if they exist
        if self.key_facts:
            facts_str = "Important context: " + ", ".join(
                [f"{k}: {v}" for k, v in self.key_facts.items()]
            )
            context_parts.append(facts_str)
            context_parts.append("")

        for msg in self.history:
            if msg["role"] == "user":
                context_parts.append(f"User: {msg['content']}")
            elif msg["role"] == "assistant":
                context_parts.append(f"Assistant: {msg['content']}")

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
        Optionally preserves key facts.
        """
        self.history.clear()
        self.turn_count = 0
        # Note: key_facts are preserved by default. Use clear_all() to clear everything.

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
        return f"ChatMemory(max_turns={self.max_turns}, current_messages={len(self.history)}, key_facts={len(self.key_facts)})"

    def _extract_key_facts(self, message: str, role: str):
        """
        Extract and store key facts from messages.
        Identifies user names, preferences, and important context.

        Args:
            message (str): The message to extract facts from
            role (str): The role (user or assistant)
        """
        if role == "user":
            # Extract user's name
            name_patterns = [
                r"my name is (\w+)",
                r"i'm (\w+)",
                r"i am (\w+)",
                r"call me (\w+)",
                r"this is (\w+)",
            ]
            for pattern in name_patterns:
                match = re.search(pattern, message.lower())
                if match:
                    name = match.group(1).capitalize()
                    self.key_facts["user_name"] = name
                    break

            # Track first question asked
            if "first_question" not in self.key_facts and len(self.history) <= 2:
                # This might be the first question
                question_indicators = [
                    "?",
                    "what",
                    "how",
                    "why",
                    "when",
                    "where",
                    "who",
                ]
                if any(
                    indicator in message.lower() for indicator in question_indicators
                ):
                    self.key_facts["first_question"] = message

        elif role == "assistant":
            # Could extract other facts from bot responses if needed
            pass

    def get_key_facts(self) -> Dict[str, str]:
        """
        Get all stored key facts.

        Returns:
            dict: Dictionary of key facts
        """
        return self.key_facts.copy()

    def add_key_fact(self, key: str, value: str):
        """
        Manually add a key fact.

        Args:
            key (str): Fact key/name
            value (str): Fact value
        """
        self.key_facts[key] = value

    def remove_key_fact(self, key: str):
        """
        Remove a specific key fact.

        Args:
            key (str): Fact key to remove
        """
        if key in self.key_facts:
            del self.key_facts[key]

    def clear_all(self):
        """
        Clear everything including history and key facts.
        """
        self.history.clear()
        self.turn_count = 0
        self.key_facts.clear()
        self.conversation_summary.clear()

    def get_enriched_history(self) -> List[Dict[str, str]]:
        """
        Get conversation history enriched with key facts context.

        Returns:
            list: Enriched message history
        """
        messages = list(self.history)

        # If we have key facts and messages, prepend context as a system-like message
        if self.key_facts and messages:
            facts_content = "Context: " + ", ".join(
                [f"{k.replace('_', ' ')}: {v}" for k, v in self.key_facts.items()]
            )
            # Insert at beginning (but this is informational, not sent to model directly)
            return [{"role": "system", "content": facts_content}] + messages

        return messages

    def answer_contextual_question(self, question: str) -> Optional[str]:
        """
        Try to answer contextual questions using stored facts and history.

        Args:
            question (str): The question to answer

        Returns:
            str or None: Answer if found, None otherwise
        """
        question_lower = question.lower().strip()

        # Check for name-related questions
        if any(
            phrase in question_lower
            for phrase in ["what is my name", "my name", "who am i"]
        ):
            if "user_name" in self.key_facts:
                return f"Your name is {self.key_facts['user_name']}."

        # Check for first question recall
        if any(
            phrase in question_lower
            for phrase in ["first question", "first thing i asked", "initially asked"]
        ):
            if "first_question" in self.key_facts:
                return (
                    f"Your first question was: \"{self.key_facts['first_question']}\""
                )

        # Check recent history for the answer
        if "first question" in question_lower or "asked" in question_lower:
            # Look through history
            for i, msg in enumerate(self.history):
                if msg["role"] == "user" and i == 0:
                    return f"Your first question was: \"{msg['content']}\""

        return None


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
