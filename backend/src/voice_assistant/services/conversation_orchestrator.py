import logging
from pathlib import Path

from voice_assistant.services.conversation import ConversationService
from voice_assistant.services.llm import LLMService
from voice_assistant.services.speech_synthesis import SpeechSynthesisService
from voice_assistant.services.transcription import TranscriptionService

logger = logging.getLogger(__name__)


class ConversationOrchestrator:
    def __init__(
        self,
        transcription_service: TranscriptionService,
        conversation_service: ConversationService,
        speech_synthesis_service: SpeechSynthesisService,
        llm_service: LLMService,
    ) -> None:
        self._transcription_service = transcription_service
        self._conversation_service = conversation_service
        self._speech_synthesis_service = speech_synthesis_service
        self._llm_service = llm_service

    def process(self, req_audio_path: str, esp_id: str) -> Path:
        # Transcribe the uploaded audio.
        transcription = self._transcription_service.transcribe(req_audio_path)
        user_message = transcription.text.strip()

        if not user_message:
            raise ValueError("No speech detected in the audio")

        # Retrieve the conversation history for this ESP32.
        history = self._conversation_service.get_history(esp_id)

        # Call the LLM service to generate the assistant response.
        assistant_message = self._llm_service.generate_response(
            history=history,
            user_message=user_message,
        )

        # Generate the assistant audio with Piper.
        resp_audio_path = self._speech_synthesis_service.synthesize(assistant_message)

        # Save the exchange only once the answer audio exists: if synthesis
        # fails, the user never heard the reply and it must not pollute history.
        self._conversation_service.add_exchange(
            esp_id,
            user_message,
            assistant_message,
        )

        return resp_audio_path
