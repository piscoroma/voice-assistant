import logging
from time import perf_counter

from voice_assistant.infrastructure.speech.whisper import WhisperRecognizer
from voice_assistant.schemas.transcription import TranscriptionResponse

logger = logging.getLogger(__name__)


class TranscriptionService:
    def __init__(self, recognizer: WhisperRecognizer) -> None:
        self._recognizer = recognizer

    def transcribe(self, audio_path: str) -> TranscriptionResponse:
        start = perf_counter()
        result = self._recognizer.transcribe(audio_path)
        elapsed = perf_counter() - start
        logger.info(
            "Whisper transcription completed in %.3f seconds",
            elapsed,
        )
        return TranscriptionResponse.model_validate(result)