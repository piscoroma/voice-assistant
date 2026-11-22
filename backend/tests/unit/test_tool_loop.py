import logging
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from voice_assistant.infrastructure.llm.deepseek import DeepSeekProvider
from voice_assistant.tools.base import Tool

PARAMETERS = {"type": "object", "properties": {}, "additionalProperties": False}


@pytest.fixture
def settings():
    return SimpleNamespace(
        deepseek_api_key="test-api-key",
        deepseek_base_url="https://api.deepseek.com",
        deepseek_model="deepseek-flash",
        deepseek_timeout=30.0,
        deepseek_max_retries=2,
        assistant_city="Torino",
    )


def make_tool(handler):
    return Tool(
        name="web_search",
        description="Search the web",
        parameters=PARAMETERS,
        handler=handler,
    )


def tool_call_response(name="web_search", arguments='{"query": "notizie"}', call_id="call_1"):
    call = SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    message = SimpleNamespace(content=None, tool_calls=[call])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def answer_response(content):
    message = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def test_tool_loop_executes_tool_and_returns_answer(settings, caplog):
    caplog.set_level(logging.INFO, logger="voice_assistant.infrastructure.llm.deepseek")
    handler = Mock(return_value="risultati della ricerca")
    tool = make_tool(handler)

    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.side_effect = [
            tool_call_response(),
            answer_response("  Ecco le notizie.  "),
        ]
        provider = DeepSeekProvider(settings, tools=[tool])

        result = provider.generate_response(history=[], user_message="notizie?")

    assert result == "Ecco le notizie."
    handler.assert_called_once_with(query="notizie")
    assert 'Tool web_search called with {"query": "notizie"}' in caplog.text
    assert "Tool web_search completed in" in caplog.text

    calls = client.chat.completions.create.call_args_list
    assert calls[0].kwargs["tools"] == [tool.schema]
    second_messages = calls[1].kwargs["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["tool_calls"][0]["id"] == "call_1"
    assert second_messages[-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "risultati della ricerca",
    }


def test_tools_are_not_sent_when_provider_has_none(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.return_value = answer_response("Ciao")
        provider = DeepSeekProvider(settings)

        provider.generate_response(history=[], user_message="Ciao")

    assert "tools" not in client.chat.completions.create.call_args.kwargs


def test_tool_loop_stops_after_max_rounds(settings):
    tool = make_tool(Mock(return_value="ok"))

    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.side_effect = [tool_call_response()] * 10
        provider = DeepSeekProvider(settings, tools=[tool])

        with pytest.raises(RuntimeError, match="Tool call limit exceeded"):
            provider.generate_response(history=[], user_message="notizie?")


def test_unknown_tool_is_reported_to_the_model(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.side_effect = [
            tool_call_response(name="missing_tool"),
            answer_response("Non posso farlo."),
        ]
        provider = DeepSeekProvider(settings)

        result = provider.generate_response(history=[], user_message="fai qualcosa")

    assert result == "Non posso farlo."
    tool_message = client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert tool_message["role"] == "tool"
    assert "missing_tool" in tool_message["content"]


def test_failing_tool_does_not_break_the_turn(settings):
    tool = make_tool(Mock(side_effect=RuntimeError("boom")))

    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.side_effect = [
            tool_call_response(),
            answer_response("Riprova più tardi."),
        ]
        provider = DeepSeekProvider(settings, tools=[tool])

        result = provider.generate_response(history=[], user_message="notizie?")

    assert result == "Riprova più tardi."
    tool_message = client.chat.completions.create.call_args_list[1].kwargs["messages"][-1]
    assert "Errore nell'esecuzione" in tool_message["content"]


def test_system_prompt_mentions_web_search_only_when_available(settings):
    with patch("voice_assistant.infrastructure.llm.deepseek.OpenAI") as openai_mock:
        client = openai_mock.return_value
        client.chat.completions.create.return_value = answer_response("Ciao")
        provider = DeepSeekProvider(settings, tools=[make_tool(Mock(return_value="ok"))])

        provider.generate_response(history=[], user_message="Ciao")

    system_content = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "web_search" in system_content
