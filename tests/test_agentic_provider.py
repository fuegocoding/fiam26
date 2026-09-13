import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.agentic.provider import CompatibleLLM, LLMConfig
from src.config import llm_config


class FakeCompletions:
    def __init__(self, content='{"ok": true}', fail_response_format=False):
        self.content = content
        self.fail_response_format = fail_response_format
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail_response_format and "response_format" in kwargs:
            raise TypeError("response_format is not supported")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
            _request_id="req-123",
        )


class FakeClient:
    def __init__(self, completions):
        self.chat = SimpleNamespace(completions=completions)


class ProviderTests(unittest.TestCase):
    def test_generic_environment_values_override_provider_aliases(self):
        env = {
            "LLM_BASE_URL": "https://local.example/v1",
            "LLM_API_KEY": "local-key",
            "LLM_MODEL": "local-model",
            "LLM_TIMEOUT_SECONDS": "31",
            "LLM_MAX_CONCURRENCY": "3",
            "LLM_TEMPERATURE": "0.2",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = llm_config()
        self.assertEqual(cfg["base_url"], "https://local.example/v1")
        self.assertEqual(cfg["api_key"], "local-key")
        self.assertEqual(cfg["model"], "local-model")
        self.assertEqual(cfg["timeout_s"], 31)
        self.assertEqual(cfg["max_concurrency"], 3)
        self.assertEqual(cfg["temperature"], 0.2)

    def test_client_uses_only_compatible_chat_arguments(self):
        completions = FakeCompletions()
        llm = CompatibleLLM(
            LLMConfig(
                base_url="https://local.example/v1",
                api_key="key",
                model="model",
                response_format="none",
            ),
            client=FakeClient(completions),
        )

        result = llm.complete_json("system", "user", request_id="trace-1")

        self.assertEqual(result.content, '{"ok": true}')
        call = completions.calls[0]
        self.assertEqual(call["model"], "model")
        self.assertNotIn("extra_body", call)
        self.assertNotIn("response_format", call)
        self.assertEqual(call["temperature"], 0.0)

    def test_json_mode_falls_back_when_endpoint_rejects_it(self):
        completions = FakeCompletions(fail_response_format=True)
        llm = CompatibleLLM(
            LLMConfig(
                base_url="https://local.example/v1",
                api_key="key",
                model="model",
                response_format="json_object",
            ),
            client=FakeClient(completions),
        )

        llm.complete_json("system", "user", request_id="trace-2")

        self.assertEqual(len(completions.calls), 2)
        self.assertIn("response_format", completions.calls[0])
        self.assertNotIn("response_format", completions.calls[1])


if __name__ == "__main__":
    unittest.main()
