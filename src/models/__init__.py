# src/models package
from .base import LLMModel
from .gemini import GeminiModel
from .openai_model import OpenAIModel


def get_model(provider: str | None = None) -> LLMModel:
    """Return a configured LLMModel for the given provider (or the default from config.yaml)."""
    from src.config import CONFIG

    if provider is None:
        provider = CONFIG.get("model_provider", "gemini")

    provider = provider.lower()
    model_name = CONFIG.get("models", {}).get(provider)

    if provider == "gemini":
        return GeminiModel(model_name=model_name)
    elif provider == "openai":
        return OpenAIModel(model_name=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Choose 'gemini' or 'openai'.")
