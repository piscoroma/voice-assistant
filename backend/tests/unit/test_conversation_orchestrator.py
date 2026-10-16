from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from voice_assistant.services.conversation_orchestrator import (
    ConversationOrchestrator,
)


@pytest.fixture
def services(tmp_path):
    transcription = Mock()
    transcription.transcribe.return_value = SimpleNamespace(text="  Accendi la luce  ")

    conversation = Mock()
    conversation.get_history.return_value = [{"role": "assistant", "content": "Dimmi"}]

    llm = Mock()
    llm.generate_response.return_value = "Risposta dell'assistente"

    speech = Mock()
    speech.synthesize.return_value = tmp_path / "response.wav"

    orchestrator = ConversationOrchestrator(
        transcription_service=transcription,
        conversation_service=conversation,
        speech_synthesis_service=speech,
        llm_service=llm,
    )

    return orchestrator, transcription, conversation, llm, speech


def test_process_runs_full_pipeline(services, tmp_path):
    orchestrator, transcription, conversation, llm, speech = services
    audio_path = str(tmp_path / "request.wav")

    result = orchestrator.process(audio_path, "esp32-salotto")

    assert result == tmp_path / "response.wav"
    transcription.transcribe.assert_called_once_with(audio_path)
    conversation.get_history.assert_called_once_with("esp32-salotto")
    llm.generate_response.assert_called_once_with(
        history=[{"role": "assistant", "content": "Dimmi"}],
        user_message="Accendi la luce",
    )
    speech.synthesize.assert_called_once_with("Risposta dell'assistente")
    conversation.add_exchange.assert_called_once_with(
        "esp32-salotto",
        "Accendi la luce",
        "Risposta dell'assistente",
    )


def test_process_rejects_empty_transcription(services, tmp_path):
    orchestrator, transcription, conversation, llm, speech = services
    transcription.transcribe.return_value = SimpleNamespace(text="  ")

    with pytest.raises(ValueError, match="No speech detected"):
        orchestrator.process(str(tmp_path / "request.wav"), "esp32-salotto")

    conversation.get_history.assert_not_called()
    llm.generate_response.assert_not_called()
    speech.synthesize.assert_not_called()
    conversation.add_exchange.assert_not_called()


def test_process_does_not_save_exchange_if_llm_fails(services, tmp_path):
    orchestrator, _transcription, conversation, llm, speech = services
    llm.generate_response.side_effect = RuntimeError("LLM unavailable")

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        orchestrator.process(str(tmp_path / "request.wav"), "esp32-salotto")

    speech.synthesize.assert_not_called()
    conversation.add_exchange.assert_not_called()


def test_process_does_not_save_exchange_if_synthesis_fails(services, tmp_path):
    orchestrator, _transcription, conversation, _llm, speech = services
    speech.synthesize.side_effect = RuntimeError("Piper unavailable")

    with pytest.raises(RuntimeError, match="Piper unavailable"):
        orchestrator.process(str(tmp_path / "request.wav"), "esp32-salotto")

    conversation.add_exchange.assert_not_called()
