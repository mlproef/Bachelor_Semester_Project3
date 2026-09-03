from src.configs import ModelConfig

from .model import BaseModel
from .open_ai import OpenAIGPT


def get_model(config: ModelConfig) -> BaseModel:
    """Condition C uses only the OpenAI-compatible provider."""
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    raise NotImplementedError(
        f"Provider {config.provider!r} is not included in this Condition C package."
    )
