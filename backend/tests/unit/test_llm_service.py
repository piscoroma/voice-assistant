from unittest.mock import Mock

from voice_assistant.services.llm import LLMService


def test_generate_response_delegates_to_provider():
    provider = Mock()
    provider.generate_response.return_value = "Ciao!"

    service = LLMService(provider)
    history = [{"role": "assistant", "content": "Buongiorno"}]

    result = service.generate_response(
        history=history,
        user_message="Come stai?",
    )

    assert result == "Ciao!"
    provider.generate_response.assert_called_once_with(
        history=history,
        user_message="Come stai?",
    )
