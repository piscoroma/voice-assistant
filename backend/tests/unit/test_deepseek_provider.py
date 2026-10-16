from types import SimpleNamespace
from unittest.mock import patch

import pytest

from voice_assistant.infrastructure.llm.deepseek import DeepSeekProvider


@pytest.fixture
def settings():
    return SimpleNamespace(
        deepseek_api_key="test-api-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-flash",
        deepseek_timeout=30.0,
        deepseek_max_retries=2,
    )


def make_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_initializes_client_from_settings(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        DeepSeekProvider(settings)

    openai_mock.assert_called_once_with(
        api_key="test-api-key",
        base_url="https://api.deepseek.com",
        timeout=30.0,
        max_retries=2,
    )


def test_generate_response_sends_model_history_and_user_message(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.return_value = make_response(
            "  Ciao, come posso aiutarti?  "
        )
        provider = DeepSeekProvider(settings)

        result = provider.generate_response(
            history=[{"role": "assistant", "content": "Ciao"}],
            user_message="Che tempo fa?",
        )

    assert result == "Ciao, come posso aiutarti?"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "deepseek-flash"
    assert kwargs["messages"][-1] == {
        "role": "user",
        "content": "Che tempo fa?",
    }
    assert kwargs["messages"][-2] == {
        "role": "assistant",
        "content": "Ciao",
    }
    assert kwargs["max_tokens"] == 300
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("content", [None, "", "   "])
def test_generate_response_rejects_empty_answer(settings, content):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.return_value = make_response(content)
        provider = DeepSeekProvider(settings)

        with pytest.raises(RuntimeError, match="empty response"):
            provider.generate_response(history=[], user_message="Ciao")


def test_generate_response_propagates_api_errors(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.side_effect = RuntimeError("API error")
        provider = DeepSeekProvider(settings)

        with pytest.raises(RuntimeError, match="API error"):
            provider.generate_response(history=[], user_message="Ciao")
