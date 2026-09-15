import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from journalism_rag.config import Settings
from journalism_rag.llm import (
    GenerationOptions,
    LLMConfigurationError,
    LLMRequestError,
    ToolDefinition,
    options_from_settings,
)
from journalism_rag.providers.llm import DeepSeekChatModel, OllamaChatModel, create_chat_model


class FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body or {}
        self.text = text

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


class Phase2ProviderTests(unittest.TestCase):
    def test_deepseek_posts_openai_compatible_payload_and_usage(self):
        response = FakeResponse(
            body={
                "model": "deepseek-flash",
                "choices": [{"message": {"content": "Απάντηση"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500},
            }
        )
        session = FakeSession(response)
        settings = Settings(deepseek_api_key="secret")
        result = DeepSeekChatModel(settings, session=session).generate(
            [{"role": "user", "content": "Ερώτηση"}]
        )
        args, kwargs = session.calls[0]
        self.assertEqual(args[0], "https://api.deepseek.com/chat/completions")
        self.assertEqual(kwargs["json"]["model"], "deepseek-flash")
        self.assertEqual(kwargs["json"]["thinking"], {"type": "disabled"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer secret")
        self.assertEqual(result.text, "Απάντηση")
        self.assertAlmostEqual(result.estimated_cost_usd, 0.00045)

    def test_deepseek_thinking_can_be_enabled_per_request(self):
        session = FakeSession(
            FakeResponse(
                body={"choices": [{"message": {"content": "Απάντηση"}, "finish_reason": "stop"}]}
            )
        )
        DeepSeekChatModel(Settings(deepseek_api_key="secret"), session=session).generate(
            [{"role": "user", "content": "Ερώτηση"}],
            GenerationOptions(thinking="enabled"),
        )
        self.assertEqual(session.calls[0][1]["json"]["thinking"], {"type": "enabled"})

    def test_deepseek_thinking_environment_override_is_validated(self):
        with patch.dict("os.environ", {"RAG_DEEPSEEK_THINKING": "enabled"}, clear=False):
            self.assertEqual(Settings.from_env().deepseek_thinking, "enabled")

    def test_environment_provider_profiles_use_safe_model_defaults(self):
        with patch.dict("os.environ", {"RAG_LLM_PROVIDER": "ollama"}, clear=True):
            local = Settings.from_env()
        self.assertEqual(local.llm_model, "journalism-rag-qwen3.5:9b")
        self.assertEqual(local.llm_base_url, "http://localhost:11434/v1")
        self.assertEqual(local.llm_max_tokens, 8192)

        with patch.dict("os.environ", {"RAG_LLM_PROVIDER": "deepseek"}, clear=True):
            hosted = Settings.from_env()
        self.assertEqual(hosted.llm_model, "deepseek-flash")
        self.assertEqual(hosted.llm_base_url, "https://api.deepseek.com")
        self.assertEqual(hosted.llm_max_tokens, 8192)

    def test_zero_provider_retries_are_allowed(self):
        with patch.dict(
            "os.environ",
            {"RAG_LLM_PROVIDER": "ollama", "RAG_LLM_MAX_RETRIES": "0"},
            clear=True,
        ):
            self.assertEqual(Settings.from_env().llm_max_retries, 0)

    def test_logging_and_diagnostics_flags_follow_validated_environment(self):
        with patch.dict(
            "os.environ",
            {"RAG_ENABLE_LOGGING": "1", "RAG_SHOW_DIAGNOSTICS": "true"},
            clear=True,
        ):
            settings = Settings.from_env()

        self.assertTrue(settings.enable_logging)
        self.assertTrue(settings.show_diagnostics)

    def test_deepseek_requires_key(self):
        with self.assertRaises(LLMConfigurationError):
            create_chat_model(Settings())

    def test_empty_provider_content_is_a_typed_request_error(self):
        response = FakeResponse(
            body={"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}
        )
        with self.assertRaisesRegex(LLMRequestError, "finish_reason=length"):
            DeepSeekChatModel(Settings(deepseek_api_key="secret"), session=FakeSession(response)).generate(
                [{"role": "user", "content": "Ερώτηση"}]
            )

    def test_blank_tool_call_content_is_accepted_for_both_openai_shapes(self):
        tool = ToolDefinition(
            name="lookup",
            schema_version="test-v1",
            description="Look up one value.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
        )
        for content in (None, ""):
            response = FakeResponse(
                body={
                    "choices": [
                        {
                            "message": {
                                "content": content,
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query":"Tempi"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ]
                }
            )
            result = DeepSeekChatModel(
                Settings(deepseek_api_key="secret"),
                session=FakeSession(response),
            ).generate(
                [{"role": "user", "content": "Ερώτηση"}],
                tools=[tool],
            )
            self.assertEqual(result.text, "")
            self.assertEqual(len(result.tool_calls), 1)

    def test_ollama_is_explicit_and_uses_local_default(self):
        local = Settings(llm_provider="ollama", llm_model="deepseek-r1:1.5b", llm_base_url="http://localhost:11434/v1")
        model = OllamaChatModel(local, session=FakeSession(FakeResponse()))
        self.assertEqual(model.base_url, "http://localhost:11434/v1")
        self.assertEqual(model.provider, "ollama")

    def test_ollama_reasoning_is_disabled_by_default(self):
        local = Settings(
            llm_provider="ollama",
            llm_model="journalism-rag-qwen3.5:9b",
            llm_base_url="http://localhost:11434/v1",
        )
        self.assertEqual(options_from_settings(local).reasoning_effort, "none")

    def test_ollama_reasoning_environment_override_is_validated(self):
        with patch.dict(
            "os.environ",
            {
                "RAG_LLM_PROVIDER": "ollama",
                "RAG_OLLAMA_REASONING_EFFORT": "low",
            },
            clear=False,
        ):
            self.assertEqual(Settings.from_env().ollama_reasoning_effort, "low")

        with self.assertRaisesRegex(ValueError, "ollama_reasoning_effort"):
            Settings(ollama_reasoning_effort="unbounded").validate()

    def test_reranker_defaults_to_qwen_but_is_opt_in(self):
        settings = Settings()
        self.assertFalse(settings.reranker_enabled)
        self.assertEqual(settings.reranker_model, "Qwen/Qwen3-Reranker-0.6B")

        with patch.dict(
            "os.environ",
            {
                "RAG_RERANKER_ENABLED": "1",
                "RAG_RERANKER_DEVICE": "mps",
                "RAG_RERANKER_CANDIDATE_LIMIT": "20",
                "RAG_RERANKER_EVIDENCE_LIMIT": "5",
            },
            clear=False,
        ):
            configured = Settings.from_env()
        self.assertTrue(configured.reranker_enabled)
        self.assertEqual(configured.reranker_device, "mps")

    def test_reranker_limits_are_validated(self):
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            Settings(reranker_candidate_limit=5, reranker_evidence_limit=6).validate()
        with self.assertRaisesRegex(ValueError, "reranker_device"):
            Settings(reranker_device="cuda").validate()


if __name__ == "__main__":
    unittest.main()
