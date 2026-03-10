"""Central config loader — reads config.yaml and .env."""

import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Project root is one level above this file (src/)
ROOT = Path(__file__).parent.parent

load_dotenv(ROOT / ".env")


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_api_key(provider: str) -> str:
    """Return the API key for the given provider, raising clearly if missing."""
    key_map = {
        "gemini": "GEMINI_API_KEY",
        "openai": "OPENAI_API_KEY",
    }
    env_var = key_map.get(provider)
    if env_var is None:
        raise ValueError(f"Unknown provider: {provider!r}")
    value = os.getenv(env_var)
    if not value:
        raise EnvironmentError(
            f"{env_var} is not set. Copy .env.example to .env and fill in your key."
        )
    return value


CONFIG = load_config()
