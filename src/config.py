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


def jev_config():
    """Return Jev System One settings (OpenRouter Decisions route).

    Auth reuses the OpenRouter key already in ``.env``: ``JEV_API_KEY``
    overrides, otherwise ``LLM_API_KEY``. Model is pinned to ``jev-1.13`` —
    never the ``jev-latest`` alias — so tuned confidence thresholds cannot
    shift under a release.
    """
    key = os.getenv("JEV_API_KEY", "").strip() or os.getenv("LLM_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "no Jev API key set: fill JEV_API_KEY or LLM_API_KEY (OpenRouter key) in .env"
        )
    base_url = (
        os.getenv("JEV_BASE_URL", "").strip()
        or "https://openrouter.ai/api/v1/systemone"
    )
    return {
        "api_key": key,
        "model": os.getenv("JEV_MODEL", "").strip() or "jev-1.13",
        "base_url": base_url,
        "timeout_s": int(os.getenv("JEV_TIMEOUT_SECONDS", "").strip() or os.getenv("LLM_TIMEOUT_SECONDS", "120")),
        "max_concurrency": int(
            os.getenv("JEV_MAX_CONCURRENCY", "").strip() or os.getenv("LLM_MAX_CONCURRENCY", "8")
        ),
    }


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
        "tool_calling": os.getenv("LLM_TOOL_CALLING", "0").strip().lower()
        in {"1", "true", "yes", "on"},
    }
