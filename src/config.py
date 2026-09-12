"""Environment and LLM configuration loader for the text scoring engine."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


class MissingAPIKeyError(RuntimeError):
    pass


PROVIDERS = ("openrouter", "openai", "anthropic")


def llm_config():
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"unsupported LLM_PROVIDER: {provider!r}")
    if provider == "openrouter":
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini").strip()
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()
        base_url = None
    else:
        key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        model = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5").strip()
        base_url = None
    if not key:
        raise MissingAPIKeyError(
            f"no API key set for provider {provider!r}: fill OPENROUTER_API_KEY, "
            "OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"
        )
    return {
        "provider": provider,
        "api_key": key,
        "model": model,
        "base_url": base_url,
        "timeout_s": int(os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        "max_concurrency": int(os.getenv("LLM_MAX_CONCURRENCY", "8")),
    }
