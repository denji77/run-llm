"""
vlm_loader.py
Module for loading and initializing Apple FastVLM (LlavaQwen2) vision-language model.
Supports text-only conversations with proper chat template usage.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from PIL import Image
from dotenv import load_dotenv

load_dotenv()


class VLMLoader:
    """
    Handles loading of Apple FastVLM (LlavaQwen2) vision-language model.
    Optimized for text-only conversations using chat templates.
    """

    def __init__(self, model_name=None, use_gpu=True):
        """
        Initialize the VLMLoader.

        Args:
            model_name (str): Name/path of the model to load.
                            If None, uses apple/FastVLM-0.5B
            use_gpu (bool): Whether to use GPU if available.
        """
        self.model_name = model_name or "apple/FastVLM-0.5B"
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN", None)
        self.model = None
        self.tokenizer = None

        print(f"Initializing VLMLoader with model: {self.model_name}")
        print(f"Device: {self.device}")

    def load_model(self):
        """
        Load the FastVLM (LlavaQwen2) model and tokenizer.

        Returns:
            tuple: (model, tokenizer) loaded from Hugging Face Hub or local cache
        """
        try:
            print(f"\nLoading FastVLM model '{self.model_name}'...")
            print("Loading from local cache or downloading if needed...")

            # Load tokenizer
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                use_fast=True,
            )

            # Ensure tokenizer has proper special tokens
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            print("Successfully loaded tokenizer")
            print(f"  Vocab size: {self.tokenizer.vocab_size}")
            print(f"  PAD token: {self.tokenizer.pad_token}")

            # Load the model with custom code support
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
            )

            self.model.to(self.device)
            self.model.eval()

            print(f"FastVLM model loaded successfully on {self.device}!")
            print(f"Model size: ~{self._get_model_size_mb():.2f} MB")
            print(f"Model type: {self.model.config.model_type}")

            return self.model, self.tokenizer

        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(
        self,
        messages,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    ):
        """
        Generate a response from the model using chat template.

        Args:
            messages (list): List of message dicts with 'role' and 'content'
            max_new_tokens (int): Maximum tokens to generate
            temperature (float): Sampling temperature
            top_p (float): Nucleus sampling parameter
            do_sample (bool): Whether to use sampling

        Returns:
            str: Generated response text
        """
        if self.model is None or self.tokenizer is None:
            self.load_model()

        try:
            # Apply chat template - this formats messages properly for the model
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Tokenize the formatted prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=False,  # Chat template already added them
            )

            # Move to device
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            # Generate response using the model's custom generate method
            # For text-only, we pass images=None
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs=input_ids,
                    attention_mask=attention_mask,
                    images=None,  # Text-only mode
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # The model returns the FULL sequence (input + generated tokens)
            # So we need to decode everything and then remove the prompt
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Decode just the input to see what prompt was
            prompt_decoded = self.tokenizer.decode(
                input_ids[0], skip_special_tokens=True
            )

            # Remove the prompt from the response
            if full_response.startswith(prompt_decoded):
                response = full_response[len(prompt_decoded) :].strip()
            else:
                response = full_response.strip()

            return response.strip()

        except Exception as e:
            error_msg = f"Error in generation: {str(e)}"
            print(f"Error: {error_msg}")
            return f"I apologize, but I encountered an error: {str(e)}"

    def chat(self, message, image_path=None, conversation_history=None):
        """
        Chat interface with conversation history support.

        Args:
            message (str): User message
            image_path (str, optional): Path to image (not currently supported)
            conversation_history (list, optional): List of previous messages

        Returns:
            str: Bot response
        """
        # Build messages list in the format expected by chat template
        messages = []

        # Add conversation history if available (but exclude the current message if it's already there)
        if conversation_history:
            # Only include recent history to avoid context overflow
            # Keep last 3 exchanges (6 messages)
            recent_history = conversation_history[-6:]
            for msg in recent_history:
                role = msg.get("role", "")
                content = msg.get("content", "")

                # Skip if this is the current message (avoid duplicates)
                if content == message and role == "user":
                    continue

                if role == "user":
                    messages.append({"role": "user", "content": content})
                elif role in ["assistant", "bot"]:
                    messages.append({"role": "assistant", "content": content})

        # Add current user message (only if not already in messages)
        if not messages or messages[-1].get("content") != message:
            messages.append({"role": "user", "content": message})

        # Generate response
        response = self.generate_response(
            messages=messages,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

        # Clean up response
        response = response.strip()

        # Remove common artifacts that might appear
        if response.startswith("Assistant:"):
            response = response[10:].strip()
        if response.startswith("assistant:"):
            response = response[10:].strip()

        # Stop at multiple answer indicators
        stop_markers = ["\n\nHuman:", "\n\nUser:", "\nA:", "\nB:", "\nC:"]
        for marker in stop_markers:
            if marker in response:
                response = response.split(marker)[0].strip()

        # If response is empty, provide fallback
        if not response or len(response) < 2:
            response = "I'm here to help! Could you please rephrase your question?"

        return response

    def _get_model_size_mb(self):
        """
        Calculate approximate model size in MB.

        Returns:
            float: Model size in megabytes
        """
        if self.model is None:
            return 0.0

        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        return size_mb

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            dict: Model information including name, device, and capabilities
        """
        return {
            "model_name": self.model_name,
            "device": self.device,
            "size_mb": self._get_model_size_mb() if self.model else 0,
            "loaded": self.model is not None,
            "capabilities": ["text"],
            "model_type": self.model.config.model_type if self.model else "unknown",
        }


def load_fastvlm():
    """
    Convenience function to load the FastVLM model.

    Returns:
        VLMLoader: Loaded VLM loader instance
    """
    loader = VLMLoader()
    loader.load_model()
    return loader


if __name__ == "__main__":
    print("Testing VLMLoader with Apple FastVLM...")
    print("=" * 70)

    loader = VLMLoader()
    model, tokenizer = loader.load_model()

    info = loader.get_model_info()
    print("\n" + "=" * 70)
    print("Model Information:")
    print("=" * 70)
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test text generation with chat template
    print("\n" + "=" * 70)
    print("Testing chat functionality:")
    print("=" * 70)

    test_message = "What is the capital of France?"
    print(f"User: {test_message}")
    response = loader.chat(test_message)
    print(f"Bot: {response}")

    print("\nVLM loader test completed successfully!")
    print("=" * 70)
