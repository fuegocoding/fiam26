"""Environment and LLM configuration shared by text scoring and council calls."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # Keep read-only config tests usable in minimal environments.
    def load_dotenv(*args, **kwargs):
        return False

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class MissingAPIKeyError(RuntimeError):
    pass


def _configured(name: str, fallback: str) -> str:
    value = os.getenv(name, "").strip()
    return value or fallback.strip()


def llm_config():
    """Return provider-neutral settings with legacy aliases still supported."""
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower().strip()
    legacy = {
        "openrouter": {
            "key": os.getenv("OPENROUTER_API_KEY", ""),
            "model": os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
            "base_url": os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        },
        "openai": {
            "key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        },
        "anthropic": {
            "key": os.getenv("ANTHROPIC_API_KEY", ""),
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            "base_url": os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        },
    }
    if provider not in legacy:
        raise ValueError(f"unsupported legacy provider alias: {provider!r}")
    fallback = legacy[provider]
    key = _configured("LLM_API_KEY", fallback["key"])
    model = _configured("LLM_MODEL", fallback["model"])
    base_url = _configured("LLM_BASE_URL", fallback["base_url"])
    if not key:
        raise MissingAPIKeyError(
            "no LLM API key set: fill LLM_API_KEY or the selected legacy provider key in .env"
        )
    return {
        "provider": provider,
        "api_key": key,
        "model": model,
        "base_url": base_url,
        "timeout_s": int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        "max_concurrency": int(os.getenv("LLM_MAX_CONCURRENCY", "8")),
        "temperature": float(os.getenv("LLM_TEMPERATURE", "0.0")),
        "max_output_tokens": int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "1600")),
        "response_format": os.getenv("LLM_RESPONSE_FORMAT", "none").strip().lower(),
    }
