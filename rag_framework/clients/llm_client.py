"""LLM Client module for handling different LLM providers."""

from abc import ABC, abstractmethod
from typing import List, Optional
import os


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        pass

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts."""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key. If None, uses OPENAI_API_KEY env variable.
            model: Model name to use (default: gpt-3.5-turbo)
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package is required. Install with: pip install openai")
        
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY env var")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the API
            
        Returns:
            Generated text
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        return response.choices[0].message.content

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


class GeminiClient(BaseLLMClient):
    """Google Gemini client."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-pro"):
        """Initialize Gemini client.
        
        Args:
            api_key: Google API key. If None, uses GOOGLE_API_KEY env variable.
            model: Model name to use (default: gemini-pro)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "google-generativeai package is required. "
                "Install with: pip install google-generativeai"
            )
        
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API key must be provided or set in GOOGLE_API_KEY env var")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)

    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000, **kwargs) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for the API
            
        Returns:
            Generated text
        """
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        response = self.model.generate_content(prompt, generation_config=generation_config)
        return response.text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            **kwargs: Additional arguments for generate()
            
        Returns:
            List of generated texts
        """
        return [self.generate(prompt, **kwargs) for prompt in prompts]


def get_llm_client(provider: str = "openai", **kwargs) -> BaseLLMClient:
    """Factory function to get LLM client.
    
    Args:
        provider: Provider name ("openai" or "gemini")
        **kwargs: Arguments to pass to the client constructor
        
    Returns:
        LLM client instance
    """
    providers = {
        "openai": OpenAIClient,
        "gemini": GeminiClient,
    }
    
    if provider.lower() not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")
    
    return providers[provider.lower()](**kwargs)
